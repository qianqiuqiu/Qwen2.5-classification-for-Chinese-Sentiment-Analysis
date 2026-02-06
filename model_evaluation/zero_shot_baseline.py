"""
Zero-shot 打分式 Baseline
ZeroShot Label-Scoring Baseline

使用 Qwen 基础模型（CausalLM）进行 zero-shot 情感分类。
方法：对每条文本拼接候选标签（"正向"/"负向"），
     计算标签 token 的 log probability，选概率更大的作为预测。

这是严格的 zero-shot：不训练任何参数，仅靠预训练模型 + prompt。
"""

import os

# 修复 torch 导入卡死问题 (Intel MKL 库冲突)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sys
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# 处理直接运行时的相对导入问题
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "model_evaluation"

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .utils import save_json, ensure_output_dir, get_device, get_torch_dtype
except ImportError:
    from model_evaluation.utils import save_json, ensure_output_dir, get_device, get_torch_dtype

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ============================================================
# Prompt 模板与候选标签
# ============================================================

PROMPT_TEMPLATE = "以下是一条中文评论，请判断它的情感倾向。\n评论：{text}\n情感倾向："

# label_id → (标签文本, 数据集中的 label 值)
# ChnSentiCorp: 0=负面, 1=正面
CANDIDATE_LABELS = {
    0: "负向",  # negative
    1: "正向",  # positive
}


def build_prompt(text: str) -> str:
    """构造 prompt（不含标签部分）"""
    return PROMPT_TEMPLATE.format(text=text)


# ============================================================
# 核心：计算标签的 log probability
# ============================================================

def compute_label_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    label_texts: List[str],
    device: torch.device,
) -> List[float]:
    """
    对一个 prompt，计算每个候选标签的 log probability。

    做法：
        1. 分别拼接 prompt + label_text
        2. 对完整序列做前向传播取 logits
        3. 只在标签 token 对应位置取 log softmax，求和

    Args:
        model: CausalLM 模型
        tokenizer: 分词器
        prompt_text: prompt 文本（不含标签）
        label_texts: 候选标签文本列表, e.g. ["正向", "负向"]
        device: 计算设备

    Returns:
        每个标签的 log probability 列表
    """
    # 先 tokenize prompt 以确定其长度
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    logprobs = []

    for label_text in label_texts:
        # 拼接完整序列：prompt + label
        full_text = prompt_text + label_text
        input_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            # logits shape: (1, seq_len, vocab_size)
            all_logits = outputs.logits[0]  # (seq_len, vocab_size)

        # 标签 token 在 input_ids 中的位置：从 prompt_len 开始到末尾
        # 对于位置 t 的 token，其预测来自位置 t-1 的 logits
        # 所以标签第一个 token 的对数概率 = logits[prompt_len - 1] 对应 input_ids[prompt_len]
        label_token_ids = input_ids[0, prompt_len:]  # 标签部分的 token ids
        label_logits = all_logits[prompt_len - 1 : -1]  # 对应位置的 logits

        # log softmax → 取对应 token 的 log prob → 求和
        log_probs = F.log_softmax(label_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=label_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        total_logprob = token_log_probs.sum().item()
        logprobs.append(total_logprob)

    return logprobs


def compute_label_logprobs_batch(
    model,
    tokenizer,
    prompts: List[str],
    label_texts: List[str],
    device: torch.device,
) -> List[List[float]]:
    """
    批量版本：对多条 prompt 分别计算各候选标签的 logprob。
    由于 prompt 长度不同会导致 padding 影响 logprob 计算，
    这里逐条处理以保证精确。

    Returns:
        List[List[float]]: 外层=样本, 内层=各标签的 logprob
    """
    all_logprobs = []
    for prompt in prompts:
        lp = compute_label_logprobs(model, tokenizer, prompt, label_texts, device)
        all_logprobs.append(lp)
    return all_logprobs


# ============================================================
# 评估主流程
# ============================================================

class ZeroShotEvaluator:
    """Zero-shot 打分式评估器"""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B",
        max_length: int = 256,
    ):
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.device = get_device()
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载 CausalLM 基础模型（不加任何分类头或 LoRA）"""
        torch_dtype = get_torch_dtype()
        print(f"使用设备: {self.device}")
        print(f"使用数据类型: {torch_dtype}")

        print(f"加载 CausalLM 模型: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        print("模型加载完成。\n")

    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        report_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        在给定数据上运行 zero-shot 打分式评估。

        Args:
            texts: 文本列表
            labels: 对应的真实标签列表 (0/1)
            report_interval: 每隔多少条打印一次进度

        Returns:
            评估结果字典
        """
        if self.model is None:
            self.load_model()

        label_texts = [CANDIDATE_LABELS[0], CANDIDATE_LABELS[1]]  # ["负向", "正向"]
        device = next(self.model.parameters()).device

        all_predictions = []
        all_positive_probs = []  # 用于 AUC-ROC
        inference_times = []
        total = len(texts)

        print(f"开始 zero-shot 打分式评估，共 {total} 条样本")
        print(f"  Prompt 模板: {PROMPT_TEMPLATE}")
        print(f"  候选标签:  label_0='{CANDIDATE_LABELS[0]}', label_1='{CANDIDATE_LABELS[1]}'")
        print()

        for idx, text in enumerate(texts):
            # 截断过长文本
            truncated = text[:self.max_length]
            prompt = build_prompt(truncated)

            start_time = time.time()
            logprobs = compute_label_logprobs(
                self.model, self.tokenizer, prompt, label_texts, device
            )
            elapsed = time.time() - start_time
            inference_times.append(elapsed)

            # logprobs[0] = "负向" 的 logprob, logprobs[1] = "正向" 的 logprob
            # 预测 = argmax
            pred = int(np.argmax(logprobs))
            all_predictions.append(pred)

            # 将 logprob 转换为概率用于 AUC-ROC
            # softmax over the two logprobs
            lp = np.array(logprobs, dtype=np.float64)
            probs = np.exp(lp - np.max(lp))  # numerically stable softmax
            probs = probs / probs.sum()
            positive_prob = float(probs[1])  # P(正向)
            all_positive_probs.append(positive_prob)

            if (idx + 1) % report_interval == 0 or idx == total - 1:
                running_acc = accuracy_score(labels[: idx + 1], all_predictions)
                print(
                    f"  [{idx + 1:>5}/{total}]  "
                    f"running_acc={running_acc:.4f}  "
                    f"last_time={elapsed * 1000:.1f}ms  "
                    f"pred={pred} label={labels[idx]}  "
                    f"logprobs=({logprobs[0]:.2f}, {logprobs[1]:.2f})"
                )

        # ----- 汇总指标 -----
        predictions = np.array(all_predictions)
        labels_arr = np.array(labels)

        accuracy = accuracy_score(labels_arr, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, predictions, average="binary"
        )
        _, recalls, _, _ = precision_recall_fscore_support(
            labels_arr, predictions, average=None
        )
        negative_recall = float(recalls[0])
        positive_recall = float(recalls[1])

        try:
            auc = roc_auc_score(labels_arr, np.array(all_positive_probs))
        except Exception:
            auc = 0.0

        total_time = sum(inference_times)
        avg_time_ms = (total_time / total) * 1000
        qps = total / total_time

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "negative_recall": negative_recall,
            "positive_recall": positive_recall,
            "auc_roc": float(auc),
            "avg_inference_time_ms": avg_time_ms,
            "qps": qps,
        }

        # 打印结果
        print(f"\n{'=' * 60}")
        print("Zero-shot 打分式 Baseline 结果")
        print(f"{'=' * 60}")
        print(f"  准确率 (Accuracy):  {accuracy:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall):    {recall:.4f}")
        print(f"  F1:                 {f1:.4f}")
        print(f"  AUC-ROC:            {auc:.4f}")
        print(f"  负面召回率:          {negative_recall:.4f}")
        print(f"  正面召回率:          {positive_recall:.4f}")
        print(f"  平均推理时间:        {avg_time_ms:.2f} ms/sample")
        print(f"  QPS:                {qps:.2f}")
        print(f"{'=' * 60}\n")

        return metrics

    def cleanup(self):
        """释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 结果保存：合并到 baseline_comparison.json
# ============================================================

def merge_results_to_baseline_comparison(metrics: Dict[str, Any]):
    """将 zero-shot 结果合并写入 baseline_comparison.json"""
    output_path = ensure_output_dir() / "baseline_comparison.json"

    # 读取已有结果
    existing = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # 添加 zero-shot 结果
    existing["qwen_zero_shot"] = {
        "key": "qwen_zero_shot",
        "name": "Qwen-1.5B Zero-shot (打分式)",
        "description": (
            "不微调，使用 CausalLM 基础模型 + prompt 模板 + label logprob 打分。"
            "严格 zero-shot：无任何参数训练，仅靠预训练知识。"
        ),
        "method": {
            "type": "label_scoring / logprob",
            "prompt_template": PROMPT_TEMPLATE,
            "candidate_labels": CANDIDATE_LABELS,
            "note": "对候选标签 token 计算 log probability，取 argmax 作为预测",
        },
        "metrics": metrics,
    }

    # 写回
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"结果已合并保存到: {output_path}")


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot 打分式 Baseline（CausalLM + label logprob）"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="基础模型名称",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="最大测试样本数（0=全部）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="文本最大字符数（截断用）",
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=100,
        help="每隔多少条打印一次进度",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("任务 A: Zero-shot 打分式 Baseline")
    print("=" * 60)
    print(f"  基础模型:   {args.base_model}")
    print(f"  最大样本数: {args.max_samples if args.max_samples > 0 else '全部'}")
    print(f"  文本截断:   {args.max_length} 字符")
    print()

    # 1. 加载数据
    print("加载数据集: ChnSentiCorp")
    dataset = load_dataset("lansinuote/ChnSentiCorp")
    eval_split = "test" if "test" in dataset else "validation"
    eval_data = dataset[eval_split]

    texts = list(eval_data["text"])
    labels = list(eval_data["label"])

    if args.max_samples > 0 and len(texts) > args.max_samples:
        print(f"限制样本数: {len(texts)} -> {args.max_samples}")
        texts = texts[: args.max_samples]
        labels = labels[: args.max_samples]

    print(f"测试样本数: {len(texts)}")
    print(f"正: {sum(labels)}, 负: {len(labels) - sum(labels)}\n")

    # 2. 评估
    evaluator = ZeroShotEvaluator(
        base_model_name=args.base_model,
        max_length=args.max_length,
    )
    evaluator.load_model()

    metrics = evaluator.evaluate(
        texts, labels, report_interval=args.report_interval
    )

    evaluator.cleanup()

    # 3. 保存结果
    merge_results_to_baseline_comparison(metrics)

    print("\n✓ 任务 A 完成！")


if __name__ == "__main__":
    main()
