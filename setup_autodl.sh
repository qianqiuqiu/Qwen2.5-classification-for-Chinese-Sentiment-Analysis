#!/bin/bash
# AutoDL环境快速配置脚本
# 使用方法: bash setup_autodl.sh

echo "🚀 开始配置AutoDL环境..."

# 检查PyTorch是否已安装
echo "📦 检查预装环境..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'GPU数量: {torch.cuda.device_count()}')
else:
    print('⚠️  当前无GPU（无卡开机模式）')
    print('💡 安装依赖后记得开卡训练！')
"

# 安装项目依赖（使用清华镜像加速）
echo ""
echo "📦 安装项目依赖..."
pip install -r requirements_autodl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps

# 验证关键包
echo ""
echo "✅ 验证安装..."
python -c "
try:
    import transformers
    import datasets
    import peft
    import bitsandbytes
    import accelerate
    print('✅ transformers:', transformers.__version__)
    print('✅ datasets:', datasets.__version__)
    print('✅ peft:', peft.__version__)
    print('✅ bitsandbytes:', bitsandbytes.__version__)
    print('✅ accelerate:', accelerate.__version__)
    print('')
    print('🎉 所有依赖安装成功！')
except Exception as e:
    print('❌ 安装出错:', e)
"

echo ""
echo "🎯 配置完成！可以开始训练了："
echo "   python train.py"
