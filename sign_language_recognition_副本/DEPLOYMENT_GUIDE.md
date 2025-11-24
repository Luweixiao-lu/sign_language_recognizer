# Streamlit Cloud 部署指南

本指南详细说明了如何将汉语手指字母识别应用部署到Streamlit Cloud平台，使其成为一个可通过链接访问的在线应用。

## 📋 前提条件

在开始部署之前，请确保您已经准备好了以下内容：

1. **GitHub账号** - Streamlit Cloud需要从GitHub仓库部署应用
2. **项目代码** - 已完成所有必要文件（app.py, requirements.txt等）
3. **手势模型** - gesture_model.pkl文件已准备就绪
4. **Python 3.8或更高版本** - 用于本地测试

## 🔍 本地测试（重要步骤！）

在部署到云端之前，请先在本地测试应用，确保它能正常工作：

```bash
# 安装依赖
pip install -r requirements.txt

# 启动Streamlit应用
streamlit run app.py
```

检查以下功能是否正常：
- 应用是否成功启动
- 摄像头是否能正常访问
- 手势识别功能是否正常工作
- 所有UI元素是否正确显示

## 🚀 部署步骤

### 1. 将项目上传到GitHub

1. 在GitHub上创建一个新的仓库（可以设置为公开或私有）
2. 将本地项目的所有文件推送到GitHub仓库

```bash
# 初始化本地git仓库（如果尚未初始化）
git init

# 添加所有文件
git add .

# 提交更改
git commit -m "初始提交 - 汉语手指字母识别应用"

# 关联GitHub仓库
git remote add origin https://github.com/您的用户名/您的仓库名.git

# 推送到GitHub
git push -u origin main
```

### 处理大型模型文件（如果适用）

如果您的 `gesture_model.pkl` 文件较大（>100MB），请考虑使用Git LFS（Large File Storage）：

```bash
# 安装Git LFS
git lfs install

# 跟踪大型文件
git lfs track "gesture_model.pkl"

# 更新.gitattributes
git add .gitattributes
git commit -m "配置Git LFS跟踪模型文件"
git push -u origin main
```

### 2. 访问Streamlit Cloud

1. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
2. 使用您的GitHub账号登录
3. 如果是首次使用，可能需要授权Streamlit访问您的GitHub仓库

### 3. 部署应用

1. 在Streamlit Cloud仪表板中，点击右上角的 "New app" 按钮
2. 在部署表单中填写以下信息：
   - **Repository**: 开始输入您的仓库名称，系统会自动搜索并显示匹配的仓库
   - **Branch**: 选择 `main` 或 `master` 分支（取决于您的仓库设置）
   - **Main file path**: 输入 `app.py`
3. 点击 "Deploy!" 按钮开始部署过程
4. 部署过程可能需要5-15分钟，取决于依赖包的安装速度

### 4. 监控部署进度

Streamlit Cloud会开始构建和部署您的应用。这个过程通常需要几到十几分钟，具体取决于依赖包的安装速度。

您可以在部署日志中查看详细进度，包括：
- 环境设置
- 依赖包安装
- 应用启动过程

## ⚙️ 环境配置说明

### 依赖管理

应用所需的所有依赖都在 `requirements.txt` 文件中指定：

```
streamlit>=1.28.0
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
Pillow>=10.0.0
pandas>=2.0.0
```

Streamlit Cloud会自动安装这些依赖包。

### Python版本

我们在 `setup.py` 中指定了Python版本要求：

```python
python_requires=">=3.8"
```

Streamlit Cloud将使用兼容的Python版本（通常是3.8或更高版本）。

## 🔧 常见问题与解决方案

### 1. 部署失败 - 依赖安装问题

**问题**: 部署过程中因依赖冲突或安装失败而失败

**解决方案**:
- 检查依赖版本是否兼容，特别是opencv-python、mediapipe和numpy之间的版本关系
- 尝试放宽版本限制，例如将`numpy>=1.24.0,<2.0.0`改为`numpy>=1.24.0`
- 考虑移除requirements.txt中不必要的依赖包
- 对于特定错误，尝试在GitHub上查找类似问题的解决方案

### 2. 应用启动后无法访问摄像头

**问题**: 应用运行后无法访问用户摄像头

**解决方案**:
- 确保用户授予了浏览器对摄像头的访问权限（首次使用时会弹出权限请求）
- 检查应用是否通过HTTPS协议运行（浏览器通常只允许HTTPS网站访问摄像头）
- 确认您的代码使用了正确的摄像头访问方式（Streamlit中通过cv2.VideoCapture(0)）
- 尝试在不同的浏览器中测试应用

### 3. 性能问题

**问题**: 在Streamlit Cloud上运行时性能低于本地

**解决方案**:
- 优化代码以减少处理时间，特别是手部检测和分类部分
- 考虑降低视频分辨率，例如在`cv2.VideoCapture`后添加`cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`和`cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)`
- 调整平滑窗口大小，减小`smooth_window`的值
- 考虑使用批处理方式处理视频帧，而非每帧都处理

### 4. 模型加载问题

**问题**: 无法加载 `gesture_model.pkl` 文件

**解决方案**:
- 确保模型文件已正确上传到GitHub仓库（检查文件大小是否正确）
- 检查您的代码中是否使用了正确的相对路径访问模型文件
- 如果模型文件大于100MB，必须使用Git LFS进行跟踪
- 尝试使用`st.cache_resource`装饰器优化模型加载性能

### 5. 浏览器兼容性问题

**问题**: 应用在某些浏览器上无法正常运行

**解决方案**:
- 推荐使用Chrome、Firefox或Edge等现代浏览器
- 确保浏览器已更新到最新版本
- 检查浏览器是否支持WebRTC（用于摄像头访问）
- 清除浏览器缓存后重试

### 6. Streamlit Cloud 资源限制问题

**问题**: 应用因资源限制而崩溃或性能下降

**解决方案**:
- 减少应用的内存使用，优化数据结构
- 限制历史数据的保留量
- 对于免费版本，注意应用会在30分钟不活跃后进入休眠状态
- 考虑升级到Streamlit Cloud的付费计划以获得更多资源

## 📡 访问您的应用

部署成功后，您可以通过以下方式访问您的应用：

1. 在Streamlit Cloud控制台中，找到您的应用
2. 点击应用名称或URL链接
3. 您的应用URL格式通常为：`https://[应用名]-[随机字符串]-[随机数字].streamlit.app`

**分享应用链接**：
- 复制应用URL并分享给其他人
- 任何人都可以通过该链接访问您的应用，无需安装任何软件
- 首次访问时可能需要等待应用从休眠状态启动（通常10-20秒）

## 🔄 更新已部署的应用

如果您需要更新应用，可以按照以下步骤操作：

1. 对本地代码进行修改
2. 提交更改到GitHub仓库：

```bash
git add .
git commit -m "更新应用功能"
git push origin main
```

3. Streamlit Cloud会自动检测到更新并重新部署应用
4. 您可以在Streamlit Cloud控制台中监控更新进度

## 📱 移动设备兼容性

该应用支持在移动设备上使用，但请注意以下几点：

- 确保移动设备具有摄像头功能
- 使用现代移动浏览器（如Chrome、Safari等）
- 可能需要在浏览器设置中授予摄像头访问权限
- 移动设备上的性能可能会有所下降
- 建议在Wi-Fi网络下使用，以获得最佳体验

## 🔒 隐私与安全

- 应用仅在用户浏览器中进行手势识别处理
- 不会存储或传输用户的视频数据
- 所有处理都在客户端完成

## 📊 监控与维护

您可以通过Streamlit Cloud控制台监控应用的使用情况：

1. 查看应用的运行状态
2. 检查日志以排查问题
3. 重启应用（如有必要）
4. 更新部署的代码版本

## 📝 注意事项

- Streamlit Cloud的免费版本有一些限制，如应用活动时间和资源使用
- 应用长期不活跃可能会进入休眠状态，首次访问需要等待应用启动
- 对于生产环境，考虑升级到Streamlit Cloud的付费计划

---

通过以上步骤，您的汉语手指字母识别应用应该可以成功部署到Streamlit Cloud平台，并可以通过链接访问使用。如果遇到任何问题，请参考常见问题部分或查看Streamlit官方文档。