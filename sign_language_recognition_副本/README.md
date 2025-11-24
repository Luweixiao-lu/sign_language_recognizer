# 汉语手指字母识别软件

这是一个基于计算机视觉和机器学习的汉语手指字母（手语字母）识别软件，可以实时识别30个不同的手语字母手势（A-Z, ZH, CH, SH, NG）。

## ✨ 功能特点

- 🎥 实时摄像头手语识别
- 📚 支持30个汉语手指字母
- 🔍 可视化手部关键点检测
- 🎨 简洁易用的图形界面（桌面版 + Web版）
- 🌐 Web应用，易于共享和部署
- 🤖 基于机器学习的智能识别
- 📊 置信度评估和结果平滑处理
- ⚡ 高性能识别，帧率可达16fps以上

## 📋 系统要求

- Python 3.8+
- 摄像头（内置或外接 USB）
- macOS / Windows / Linux

## 🚀 快速开始

### 方式一：Web应用（推荐）✨

最简单的方式，适合所有用户：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动Web应用
# 或使用启动脚本
chmod +x run_web.sh && ./run_web.sh

# 3. 浏览器访问 http://localhost:5001
```

### 方式二：直接运行脚本 💻

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行识别软件
python web_app.py
# 或使用启动脚本
chmod +x run_web.sh && ./run_web.sh
```

## 📖 详细文档

**完整用户手册**: 请查看 [`用户手册.md`](用户手册.md)

手册包含：
- 📘 详细的安装指南
- 📗 完整的使用说明
- 📙 手势动作详解
- 📕 常见问题解答
- 🔧 故障排除指南

## 🎯 支持的手势

软件支持识别以下30个手语字母：
- **A-Z**（26个英文字母）
- **ZH, CH, SH, NG**（4个汉语拼音声母）

详细手势说明请参考 [`gesture_guide.md`](gesture_guide.md)

## 💡 使用提示

1. ✅ 确保摄像头正常工作
2. ✅ 在良好的光照条件下使用
3. ✅ 将手放在摄像头前，保持手势清晰
4. ✅ 每个手势保持2-3秒以便识别
5. ✅ 确保手部完全在摄像头视野内

## 📁 项目结构

```
sign_language_recognition/
├── main.py                 # 主程序（桌面版手势识别）
├── web_app.py              # Web应用（Flask）
├── data_collector.py       # 数据收集工具
├── train_model.py          # 模型训练脚本
├── hand_landmarks.py       # 手部关键点检测
├── gesture_classifier.py   # 手势分类器
├── gesture_model.pkl       # 预训练模型
├── requirements.txt        # 依赖库列表
├── run.sh                  # 桌面版启动脚本
├── run_web.sh              # Web版启动脚本
├── README.md               # 项目说明（本文件）
├── 用户手册.md            # 完整用户手册
├── gesture_guide.md        # 手势指南
├── static/                 # 静态资源（CSS, JS）
├── templates/              # HTML模板
└── training_data/          # 训练数据
```

## 🔧 技术栈

- **计算机视觉**: OpenCV, MediaPipe
- **机器学习**: scikit-learn (Random Forest)
- **Web框架**: Flask
- **数据处理**: NumPy
- **编程语言**: Python 3.8+

## 📊 性能指标

- **帧率**: 约16.9fps
- **检测时间**: 约18.86ms/帧
- **分类时间**: 约52.15ms/帧
- **准确率**: 取决于训练数据质量，一般在90%以上

## 📝 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request！

---

**详细使用说明请查看 [`用户手册.md`](用户手册.md)**