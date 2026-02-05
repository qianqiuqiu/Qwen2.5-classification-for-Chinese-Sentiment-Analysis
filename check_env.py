"""
环境检查脚本 - 适用于有卡和无卡模式
运行: python check_env.py
"""
import sys
import os
import glob

def check_environment():
    """检查AutoDL环境配置"""
    print("=" * 50)
    print("🔍 AutoDL环境检查")
    print("=" * 50)
    
    # 检查本地模型和数据集
    print("\n📁 检查本地文件:")
    
    # 检查模型
    model_cache = "models--Qwen--Qwen2.5-1.5B"
    if os.path.exists(model_cache):
        snapshots = glob.glob(os.path.join(model_cache, "snapshots", "*"))
        if snapshots:
            print(f"   ✅ 本地模型: {snapshots[0]}")
        else:
            print(f"   ⚠️  模型缓存文件夹存在但无快照")
    else:
        print(f"   ⚠️  本地模型不存在，将从网络下载")
    
    # 检查数据集
    dataset_cache = "datasets--lansinuote--ChnSentiCorp"
    if os.path.exists(dataset_cache):
        snapshots = glob.glob(os.path.join(dataset_cache, "snapshots", "*"))
        if snapshots:
            print(f"   ✅ 本地数据集: {snapshots[0]}")
        else:
            print(f"   ⚠️  数据集缓存文件夹存在但无快照")
    else:
        print(f"   ⚠️  本地数据集不存在，将从网络下载")
    
    # 检查 HuggingFace 镜像设置
    hf_endpoint = os.environ.get('HF_ENDPOINT', '未设置')
    print(f"\n🌐 HuggingFace镜像: {hf_endpoint}")
    if hf_endpoint == '未设置':
        print("   💡 建议设置: export HF_ENDPOINT=https://hf-mirror.com")
    
    print("=" * 50)
    
    # 检查Python版本
    print(f"\n📌 Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        print(f"{'✅' if cuda_available else '⚠️ '} CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                # 显示显存信息
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   显存: {mem_total:.1f} GB")
        else:
            print("   ⚠️  无GPU检测（无卡开机模式）")
            print("   💡 训练前需要开卡！")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查关键依赖
    modules = {
        'transformers': 'transformers',
        'datasets': 'datasets', 
        'peft': 'peft',
        'bitsandbytes': 'bitsandbytes',
        'accelerate': 'accelerate',
        'sklearn': 'scikit-learn',
    }
    
    print("\n📦 检查项目依赖:")
    all_installed = True
    for module, package in modules.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {package}: {version}")
        except ImportError:
            print(f"   ❌ {package}: 未安装")
            all_installed = False
    
    print("\n" + "=" * 50)
    if all_installed and torch.cuda.is_available():
        print("🎉 环境配置完美！可以开始训练了！")
        print("   运行: python train.py")
    elif all_installed:
        print("✅ 依赖已安装，但需要开卡才能训练")
        print("   💡 在AutoDL控制台开卡后即可训练")
    else:
        print("⚠️  请先安装依赖:")
        print("   pip install -r requirements_autodl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
    print("=" * 50)
    
    return all_installed

if __name__ == "__main__":
    check_environment()
