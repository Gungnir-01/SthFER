#!/bin/bash
# ==============================================
# SthFER 面部表情识别 - 便捷运行脚本
# 项目路径: /root/autodl-tmp/SthFER
# ==============================================

set -e
PROJECT_DIR="/root/autodl-tmp/SthFER"
cd "$PROJECT_DIR"
source venv/bin/activate

show_help() {
    echo "========================================"
    echo " SthFER 面部表情识别系统"
    echo "========================================"
    echo ""
    echo "用法: bash run.sh <命令> [参数]"
    echo ""
    echo "命令:"
    echo "  setup          - 预处理数据集 (需先上传 fer2013.csv)"
    echo "  train          - 训练 Swin Transformer 模型"
    echo "  train-cnn      - 训练 CNN 模型"
    echo "  predict        - Swin Transformer 预测单张图片表情"
    echo "  gradcam        - Swin Transformer 生成 Grad-CAM 热力图"
    echo "  gradcam-cnn    - CNN 生成 Grad-CAM 热力图"
    echo "  evaluate       - 评估 Swin Transformer 模型"
    echo "  test           - 测试环境是否正常"
    echo ""
    echo "示例:"
    echo "  bash run.sh setup"
    echo "  bash run.sh train          --epochs 20 --batch_size 64"
    echo "  bash run.sh train-cnn      --epochs 50 --batch_size 64"
    echo "  bash run.sh predict        /path/to/face.jpg"
    echo "  bash run.sh gradcam        /path/to/face.jpg"
    echo "  bash run.sh gradcam-cnn    /path/to/face.jpg"
    echo ""
    echo "模型文件位置:"
    echo "  Swin Transformer: Models/Swin_Transformer/best_model.pth"
    echo "  CNN:              Models/CNN/best_model.pth"
    echo "  FER2013数据集:    fer2013/fer2013.csv"
    echo "========================================"
}

case "$1" in
    setup)
        echo ">>> 预处理 FER2013 数据集..."
        if [ ! -f "fer2013/fer2013.csv" ]; then
            echo "错误: 请先将 fer2013.csv 上传到 fer2013/ 目录"
            exit 1
        fi
        python utilities/preprocess_data.py
        echo ">>> 预处理完成!"
        ;;
    train)
        shift
        python utilities/train_model.py "$@"
        ;;
    predict)
        shift
        if [ ! -f "Models/Swin_Transformer/best_model.pth" ]; then
            echo "错误: 请先将预训练模型上传到 Models/Swin_Transformer/best_model.pth"
            exit 1
        fi
        python predict_image.py "$@"
        ;;
    gradcam)
        shift
        if [ ! -f "Models/Swin_Transformer/best_model.pth" ]; then
            echo "错误: 请先将预训练模型上传到 Models/Swin_Transformer/best_model.pth"
            exit 1
        fi
        python generate_gradcam.py "$@"
        ;;
    train-cnn)
        shift
        if [ ! -d "FER2013_processed/train" ]; then
            echo "提示: 数据集未预处理，正在自动运行 setup..."
            python utilities/preprocess_data.py
        fi
        python utilities/train_cnn.py "$@"
        ;;
    gradcam-cnn)
        shift
        if [ ! -f "Models/CNN/best_model.pth" ]; then
            echo "错误: 请先训练 CNN 模型: bash run.sh train-cnn"
            exit 1
        fi
        python generate_gradcam_cnn.py "$@"
        ;;
    evaluate)
        if [ ! -f "Models/Swin_Transformer/best_model.pth" ]; then
            echo "错误: 请先将预训练模型上传到 Models/Swin_Transformer/best_model.pth"
            exit 1
        fi
        if [ ! -d "FER2013_processed/test" ]; then
            echo "错误: 请先运行 bash run.sh setup 预处理数据集"
            exit 1
        fi
        python utilities/evaluate_model.py
        ;;
    test)
        echo ">>> 测试环境..."
        python -c "
import torch; print(f'PyTorch {torch.__version__}');
print(f'CUDA 可用: {torch.cuda.is_available()}');
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}');
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB');
import timm; print(f'timm: {timm.__version__}');
from pytorch_grad_cam import GradCAM; print('grad-cam: OK');
import cv2; print(f'OpenCV: {cv2.__version__}');
import albumentations; print(f'Albumentations: OK');
print('>>> 环境正常!')
"
        ;;
    *)
        show_help
        ;;
esac
