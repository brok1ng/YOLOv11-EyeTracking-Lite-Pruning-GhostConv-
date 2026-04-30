# EyeTrack App - YOLO11 眼动跟踪应用

## 文件说明
- `ui_main.py` - 主程序入口（PyQt5 GUI）
- `yolo_infer.py` - YOLO 推理封装类
- `utils.py` - 图像处理工具函数
- `main.py` - 备用主程序
- `config.yaml` - 环境配置模板
- `requirements.txt` - Python 依赖
- `setup.sh` - 快速设置脚本
- `models/` - 模型文件目录（需手动放置 .pt 文件）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 setup.sh
bash setup.sh

# 运行程序
python ui_main.py
```

## 模型文件
请将训练好的 .pt 模型文件放入 `models/` 目录，程序会自动识别加载。
