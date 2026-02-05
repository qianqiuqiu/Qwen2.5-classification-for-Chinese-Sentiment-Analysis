"""
微调主脚本
使用 LoRA/QLoRA 微调 Qwen-1.5B 进行中文情感分析

使用方法:
    python train.py                          # 使用默认配置
    python train.py --use_qlora              # 使用 QLoRA（4-bit 量化）
    python train.py --lora_r 16              # 自定义 LoRA 秩
    python train.py --num_epochs 5           # 自定义训练轮数
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict, Any
import wandb
import glob

# 设置 HuggingFace 镜像（用于在线下载时加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training

# 导入项目配置
from configs import (
    get_lora_config,
    LORA_CONFIGS,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    QLoRAConfig,
    get_training_args,
)
from data import load_sentiment_dataset, create_tokenized_dataset, get_data_collator

# 评估指标
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_local_model_path(model_name: str) -> str:
    """
    检测并返回本地模型路径
    
    Args:
        model_name: HuggingFace 模型名称，如 "Qwen/Qwen2.5-1.5B"
    
    Returns:
        本地模型路径或原始模型名称
    """
    # 将 HuggingFace 格式转换为本地缓存路径格式
    # "Qwen/Qwen2.5-1.5B" -> "models--Qwen--Qwen2.5-1.5B"
    cache_folder = "models--" + model_name.replace("/", "--")
    
    # 在当前目录下查找
    if os.path.exists(cache_folder):
        # 查找 snapshots 目录中的模型
        snapshot_pattern = os.path.join(cache_folder, "snapshots", "*")
        snapshots = glob.glob(snapshot_pattern)
        if snapshots:
            model_path = snapshots[0]  # 使用第一个快照
            print(f"✅ 检测到本地模型: {model_path}")
            return model_path
    
    print(f"🌐 本地模型不存在，将从 HuggingFace 下载: {model_name}")
    return model_name


def get_local_dataset_path(dataset_name: str) -> tuple:
    """
    检测并返回本地数据集路径
    
    Args:
        dataset_name: HuggingFace 数据集名称，如 "lansinuote/ChnSentiCorp"
    
    Returns:
        (是否使用本地, 数据集路径)
    """
    # 构建缓存文件夹名称
    cache_folder = "datasets--" + dataset_name.replace("/", "--")
    
    # 在当前目录下查找
    if os.path.exists(cache_folder):
        # 查找 snapshots 目录
        snapshot_pattern = os.path.join(cache_folder, "snapshots", "*")
        snapshots = glob.glob(snapshot_pattern)
        if snapshots:
            dataset_path = snapshots[0]
            print(f"✅ 检测到本地数据集: {dataset_path}")
            return True, dataset_path
    
    print(f"🌐 本地数据集不存在，将从 HuggingFace 下载: {dataset_name}")
    return False, dataset_name


def get_last_checkpoint(output_dir: str) -> str:
    """
    获取最新的 checkpoint 路径
    
    Args:
        output_dir: 输出目录
    
    Returns:
        最新 checkpoint 的路径，如果不存在则返回 None
    """
    if not os.path.isdir(output_dir):
        return None
    
    # 查找所有 checkpoint 目录
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    if not checkpoints:
        return None
    
    # 返回最新的 checkpoint（按修改时间排序）
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        eval_pred: 包含预测结果和标签的元组
    
    Returns:
        指标字典
    """
    predictions, labels = eval_pred
    
    # 如果是 logits，取 argmax
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=-1)
    
    # 计算各项指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def setup_model_and_tokenizer(
    model_config: ModelConfig,
    qlora_config: QLoRAConfig,
) -> tuple:
    """
    初始化模型和分词器
    
    Args:
        model_config: 模型配置
        qlora_config: QLoRA 配置
    
    Returns:
        (model, tokenizer) 元组
    """
    
    print(f"正在加载模型: {model_config.model_name_or_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    
    # 确保有 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化（QLoRA）
    quantization_config = None
    if qlora_config.use_qlora:
        print("启用 QLoRA 4-bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config.load_in_4bit,
            bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
        )
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=model_config.num_labels,
        trust_remote_code=model_config.trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto" if qlora_config.use_qlora else None,
        torch_dtype=torch.bfloat16,
    )
    
    # 设置 pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 如果使用 QLoRA，准备模型
    if qlora_config.use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def setup_lora(
    model,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_qlora: bool = False,
):
    """
    为模型添加 LoRA 适配器
    
    Args:
        model: 基础模型
        lora_r: LoRA 秩
        lora_alpha: LoRA alpha
        lora_dropout: Dropout 概率
        use_qlora: 是否使用 QLoRA
    
    Returns:
        带有 LoRA 的模型
    """
    
    print(f"配置 LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    lora_config = get_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_qlora=use_qlora,
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def train(args: argparse.Namespace):
    """
    主训练函数
    
    Args:
        args: 命令行参数
    """
    
    # ==================== 1. 配置初始化 ====================
    # 检测本地模型
    local_model_path = get_local_model_path(args.model_name)
    
    model_config = ModelConfig(
        model_name_or_path=local_model_path,
        num_labels=2,
    )
    
    data_config = DataConfig(
        dataset_name=args.dataset,
        max_length=args.max_length,
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    qlora_config = QLoRAConfig(
        use_qlora=args.use_qlora,
    )
    
    # ==================== 2. 加载模型和分词器 ====================
    model, tokenizer = setup_model_and_tokenizer(model_config, qlora_config)
    
    # ==================== 3. 应用 LoRA ====================
    model = setup_lora(
        model=model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=args.use_qlora,
    )
    
    # ==================== 4. 加载和处理数据 ====================
    print(f"\n正在加载数据集: {data_config.dataset_name}")
    
    # 检测本地数据集
    use_local, dataset_path = get_local_dataset_path("lansinuote/ChnSentiCorp")
    dataset = load_sentiment_dataset(data_config.dataset_name, local_path=dataset_path if use_local else None)
    
    print("正在进行分词处理...")
    tokenized_dataset = create_tokenized_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=data_config.max_length,
    )
    
    print(f"训练集大小: {len(tokenized_dataset['train'])}")
    print(f"验证集大小: {len(tokenized_dataset['validation'])}")
    
    # ==================== 5. 初始化 wandb ====================
    wandb.init(
        project="qwen-sentiment-analysis",
        name=f"lora-r{args.lora_r}-{args.dataset}",
        config={
            "model_name": args.model_name,
            "dataset": args.dataset,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "use_qlora": args.use_qlora,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
        },
        tags=["LoRA", "QLoRA" if args.use_qlora else "LoRA", "sentiment-analysis"],
    )
    
    # ==================== 6. 配置训练参数 ====================
    training_args = TrainingArguments(
        **get_training_args(training_config)
    )
    
    # ==================== 7. 初始化 Trainer ====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=get_data_collator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ] if args.early_stopping else [],
    )
    
    # ==================== 8. 检测断点 ====================
    checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            # 自动检测最新 checkpoint
            checkpoint = get_last_checkpoint(args.output_dir)
            if checkpoint:
                print(f"\n检测到断点: {checkpoint}")
                print("将从断点恢复训练...\n")
            else:
                print("\n未检测到可用的 checkpoint，将从头开始训练...\n")
        else:
            # 使用指定的 checkpoint 路径
            checkpoint = args.resume_from_checkpoint
            if os.path.isdir(checkpoint):
                print(f"\n从指定断点恢复: {checkpoint}\n")
            else:
                print(f"\n警告: 指定的 checkpoint 不存在: {checkpoint}")
                print("将从头开始训练...\n")
                checkpoint = None
    
    # ==================== 9. 开始训练 ====================
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50 + "\n")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # ==================== 10. 保存模型 ====================
    print("\n保存模型...")
    
    # 保存 LoRA 适配器
    lora_save_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    
    # 保存训练结果
    trainer.save_metrics("train", train_result.metrics)
    
    # ==================== 11. 最终评估 ====================
    print("\n在测试集上进行最终评估...")
    if "test" in tokenized_dataset:
        test_results = trainer.evaluate(tokenized_dataset["test"])
        print(f"\n测试集结果：")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")
        trainer.save_metrics("test", test_results)
        
        # 记录测试结果到 wandb
        wandb.log({f"test/{key}": value for key, value in test_results.items()})
    
    print(f"\n训练完成！模型已保存到: {lora_save_path}")
    
    # 结束 wandb 运行
    wandb.finish()
    
    return trainer


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    
    parser = argparse.ArgumentParser(
        description="使用 LoRA 微调 Qwen 进行中文情感分析"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="模型名称或路径",
    )
    
    # 数据参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="ChnSentiCorp",
        choices=["ChnSentiCorp", "IMDB_Chinese"],
        help="数据集名称",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="最大序列长度",
    )
    
    # LoRA 参数
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA 秩",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="使用 QLoRA（4-bit 量化）",
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="输出目录",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批次大小",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="启用梯度检查点（节省显存）",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=True,
        help="启用早停",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help="从 checkpoint 恢复训练。'auto' 表示自动检测最新 checkpoint，也可以指定具体路径，'none' 表示不恢复",
    )
    
    args = parser.parse_args()
    
    # 处理 resume_from_checkpoint 参数
    if args.resume_from_checkpoint.lower() == "none":
        args.resume_from_checkpoint = None
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 60)
    print("中文情感分析 - LoRA 微调")
    print("=" * 60)
    print(f"\n配置信息：")
    print(f"  模型: {args.model_name}")
    print(f"  数据集: {args.dataset}")
    print(f"  LoRA r: {args.lora_r}")
    print(f"  QLoRA: {'是' if args.use_qlora else '否'}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print()
    
    train(args)
