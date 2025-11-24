#!/bin/bash

# 启动手语识别Web应用

echo "=================================================="
echo "高级手语识别系统 - Web应用启动脚本"
echo "=================================================="

# 检查是否已安装所需依赖
echo "正在检查Python依赖..."

# 检查并安装Flask和其他依赖
if ! pip list | grep -q Flask; then
    echo "安装Flask依赖..."
    pip install Flask
fi

# 检查其他必要的依赖
if [ -f "requirements.txt" ]; then
    echo "安装项目依赖..."
    pip install -r requirements.txt
fi

echo "启动Web应用..."
echo "服务器将在 http://localhost:5001 上运行"
echo "按 Ctrl+C 停止服务器"
echo "=================================================="

# 运行Web应用
python web_app.py
