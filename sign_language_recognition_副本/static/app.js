// 高级手语识别系统 - 增强功能

// 全局变量
const App = {
    state: {
        isDarkMode: false,
        isMobile: window.innerWidth < 768,
        lastPredictions: [],
        confidenceThreshold: 0.6,
        showLandmarks: true,
        currentMode: 'recognition',
        currentGesture: null,
        cameraActive: false,
        stats: {
            totalPredictions: 0,
            correctPredictions: 0,
            accuracy: 0,
            lastUpdated: null
        }
    },
    
    // 初始化应用
    init() {
        console.log('高级手语识别系统初始化...');
        
        // 绑定事件监听器
        this.bindEvents();
        
        // 检查设备和浏览器兼容性
        this.checkCompatibility();
        
        // 设置响应式布局
        this.setupResponsive();
        
        // 启动状态更新循环
        this.startStateUpdates();
        
        // 加载用户设置
        this.loadUserSettings();
    },
    
    // 绑定事件监听器
    bindEvents() {
        // 模式切换按钮
        document.getElementById('mode-demo')?.addEventListener('click', () => this.switchMode('demo'));
        document.getElementById('mode-learning')?.addEventListener('click', () => this.switchMode('learning'));
        document.getElementById('mode-recognition')?.addEventListener('click', () => this.switchMode('recognition'));
        
        // 主题切换
        document.getElementById('theme-toggle')?.addEventListener('click', () => this.toggleTheme());
        
        // 手势卡片点击
        document.querySelectorAll('.gesture-card')?.forEach(card => {
            card.addEventListener('click', (e) => {
                const gesture = e.currentTarget.querySelector('span:first-child').textContent;
                this.selectGesture(gesture);
            });
        });
        
        // 清除历史记录
        document.getElementById('clear-history')?.addEventListener('click', () => this.clearHistory());
        
        // 刷新摄像头
        document.getElementById('refresh-camera')?.addEventListener('click', () => this.refreshCamera());
        
        // 关键点显示切换
        document.getElementById('toggle-landmarks')?.addEventListener('click', () => this.toggleLandmarks());
        
        // 窗口大小改变
        window.addEventListener('resize', () => this.handleResize());
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    },
    
    // 检查浏览器兼容性
    checkCompatibility() {
        // 检查是否支持MediaDevices API
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showNotification('您的浏览器不支持摄像头功能，请使用现代浏览器如Chrome、Firefox或Edge', 'error');
            return false;
        }
        
        // 检查WebRTC支持
        if (!window.RTCPeerConnection) {
            this.showNotification('您的浏览器不支持WebRTC，某些功能可能受限', 'warning');
        }
        
        return true;
    },
    
    // 设置响应式布局
    setupResponsive() {
        this.state.isMobile = window.innerWidth < 768;
        
        if (this.state.isMobile) {
            // 移动设备布局调整
            this.adjustForMobile();
        } else {
            // 桌面布局调整
            this.adjustForDesktop();
        }
    },
    
    // 处理窗口大小改变
    handleResize() {
        const wasMobile = this.state.isMobile;
        this.state.isMobile = window.innerWidth < 768;
        
        // 如果设备类型发生变化，重新调整布局
        if (wasMobile !== this.state.isMobile) {
            this.setupResponsive();
        }
    },
    
    // 移动设备布局调整
    adjustForMobile() {
        console.log('调整为移动设备布局');
        // 可以在这里添加移动设备特定的CSS类或布局调整
    },
    
    // 桌面设备布局调整
    adjustForDesktop() {
        console.log('调整为桌面设备布局');
        // 可以在这里添加桌面设备特定的CSS类或布局调整
    },
    
    // 切换模式
    async switchMode(mode) {
        if (this.state.currentMode === mode) return;
        
        this.state.currentMode = mode;
        
        try {
            await this.sendSettingsUpdate({ mode: mode });
            this.updateModeUI();
            
            // 显示模式切换通知
            let modeText;
            switch(mode) {
                case 'recognition':
                    modeText = '识别模式';
                    break;
                case 'learning':
                    modeText = '学习模式';
                    break;
                case 'demo':
                    modeText = '演示模式';
                    break;
                default:
                    modeText = '未知模式';
            }
            this.showNotification(`已切换到${modeText}`, 'success');
            
        } catch (error) {
            console.error('切换模式失败:', error);
            this.showNotification('切换模式失败', 'error');
        }
    },
    
    // 更新模式UI
    updateModeUI() {
        const recognitionBtn = document.getElementById('mode-recognition');
        const learningBtn = document.getElementById('mode-learning');
        const demoBtn = document.getElementById('mode-demo');
        
        if (!recognitionBtn || !learningBtn || !demoBtn) return;
        
        // 重置所有按钮样式
        const resetButtonStyles = (button) => {
            button.classList.remove('bg-primary', 'text-white');
            button.classList.add('bg-slate-200', 'text-slate-700');
        };
        
        resetButtonStyles(recognitionBtn);
        resetButtonStyles(learningBtn);
        resetButtonStyles(demoBtn);
        
        // 根据当前模式设置活动按钮样式
        const activateButton = (button) => {
            button.classList.remove('bg-slate-200', 'text-slate-700');
            button.classList.add('bg-primary', 'text-white');
        };
        
        switch(this.state.currentMode) {
            case 'recognition':
                activateButton(recognitionBtn);
                break;
            case 'learning':
                activateButton(learningBtn);
                break;
            case 'demo':
                activateButton(demoBtn);
                break;
        }
    },
    
    // 选择手势
    async selectGesture(gesture) {
        this.state.currentGesture = gesture;
        
        try {
            await this.sendSettingsUpdate({
                mode: 'learning',
                current_gesture: gesture
            });
            
            // 高亮选中的卡片
            this.highlightGestureCard(gesture);
            
            // 显示学习指南
            this.showGestureGuide(gesture);
            
            // 显示通知
            this.showNotification(`开始学习手势: ${gesture}`, 'info');
            
        } catch (error) {
            console.error('选择手势失败:', error);
            this.showNotification('选择手势失败', 'error');
        }
    },
    
    // 高亮手势卡片
    highlightGestureCard(gesture) {
        document.querySelectorAll('.gesture-card').forEach(card => {
            const cardGesture = card.querySelector('span:first-child').textContent;
            if (cardGesture === gesture) {
                card.classList.add('border-primary', 'bg-primary/5');
                card.classList.add('animate-bounce-slow');
                setTimeout(() => {
                    card.classList.remove('animate-bounce-slow');
                }, 2000);
            } else {
                card.classList.remove('border-primary', 'bg-primary/5');
            }
        });
    },
    
    // 显示手势指南
    showGestureGuide(gesture) {
        // 为不同手势提供更详细的学习指南
        const gestureGuides = {
            'A': {
                description: '将手伸直，拇指自然弯曲，手掌朝向侧面。',
                tips: '保持手指伸直，拇指不要完全伸直。'
            },
            'B': {
                description: '将所有手指伸直分开，手掌朝下。',
                tips: '手指尽量分开，形成扇形。'
            },
            'C': {
                description: '形成C形，拇指和食指相触，其他手指弯曲。',
                tips: '指尖不要完全闭合，保持C形的开口。'
            },
            'D': {
                description: '食指伸直，其他手指握拳，手掌朝向侧面。',
                tips: '食指指向正前方，拇指自然弯曲。'
            },
            'E': {
                description: '将所有手指弯曲形成E形，拇指放在食指和中指之间。',
                tips: '手指不要完全弯曲，保持E的形状。'
            },
            'F': {
                description: '拇指和食指相触形成O形，其他手指伸直。',
                tips: 'O形要清晰，其他手指伸直分开。'
            },
            'G': {
                description: '食指伸直，中指伸直与食指垂直形成G形。',
                tips: '食指和中指形成约90度角。'
            },
            'H': {
                description: '食指和中指伸直分开，其他手指握拳。',
                tips: '伸直的手指保持平行，距离适中。'
            },
            'I': {
                description: '小指伸直，其他手指握拳。',
                tips: '小指尽量伸直，指向正前方。'
            },
            'J': {
                description: '食指伸直，其他手指握拳，然后水平旋转手腕形成J形。',
                tips: '手腕旋转要自然流畅。'
            },
            'K': {
                description: '食指和中指伸直分开，拇指与中指相触。',
                tips: '保持食指和中指成V形，拇指与中指接触。'
            },
            'L': {
                description: '食指和拇指伸直形成L形，其他手指握拳。',
                tips: 'L形要成直角，手指伸直。'
            },
            'M': {
                description: '拇指和小指向内弯曲，其他三指伸直分开。',
                tips: '形成M的波纹形状。'
            },
            'N': {
                description: '食指和中指伸直并靠拢，其他手指握拳。',
                tips: '伸直的手指紧密贴合。'
            },
            'O': {
                description: '所有手指弯曲并指尖相触，形成O形。',
                tips: '保持圆形，不要让手指分开。'
            },
            'P': {
                description: '拇指和食指形成圆形，其他手指伸直。',
                tips: '手掌朝向侧面，圆形要清晰。'
            },
            'Q': {
                description: '拇指和食指形成圆形，然后向下微微倾斜手腕。',
                tips: '手腕倾斜角度约30度。'
            },
            'R': {
                description: '食指和中指弯曲，其他手指握拳，拇指与食指相触。',
                tips: '形成R的形状，手指弯曲但不完全闭合。'
            },
            'S': {
                description: '所有手指弯曲并相互接触，形成拳头。',
                tips: '不要握得太紧，保持自然。'
            },
            'T': {
                description: '拇指与食指相触，其他手指伸直。',
                tips: '形成T形，保持手掌侧面朝向。'
            },
            'U': {
                description: '食指和中指伸直并向上分开，拇指位于下方。',
                tips: '形成U形，手指伸直。'
            },
            'V': {
                description: '食指和中指伸直并分开形成V形。',
                tips: 'V形要明显，手指伸直。'
            },
            'W': {
                description: '食指、中指和无名指伸直并分开形成W形。',
                tips: '形成波浪状的W形。'
            },
            'X': {
                description: '食指交叉于中指之上，其他手指握拳。',
                tips: '交叉点要清晰可见。'
            },
            'Y': {
                description: '拇指和小指伸直分开，其他手指弯曲。',
                tips: '形成Y形，保持手指伸直。'
            },
            'Z': {
                description: '食指伸直，然后水平画出Z字形。',
                tips: '动作要连贯流畅。'
            },
            '0': {
                description: '所有手指弯曲成拳头。',
                tips: '保持自然的握拳姿态。'
            },
            '1': {
                description: '食指伸直，其他手指握拳。',
                tips: '食指指向正前方。'
            },
            '2': {
                description: '食指和中指伸直分开，其他手指握拳。',
                tips: '形成V形，表示数字2。'
            },
            '3': {
                description: '食指、中指和无名指伸直分开，其他手指握拳。',
                tips: '手指伸直，分开均匀。'
            },
            '4': {
                description: '食指、中指、无名指和小指伸直分开，拇指弯曲。',
                tips: '形成数字4的手势。'
            },
            '5': {
                description: '所有手指伸直分开，手掌展开。',
                tips: '手指尽量分开，形成数字5。'
            },
            '6': {
                description: '拇指、食指和中指形成O形，其他手指伸直。',
                tips: '保持O形清晰。'
            },
            '7': {
                description: '拇指和食指伸直分开，其他手指握拳，指向斜上方。',
                tips: '形成数字7的手势。'
            },
            '8': {
                description: '食指和中指伸直并弯曲形成8字形。',
                tips: '动作要连贯。'
            },
            '9': {
                description: '食指弯曲形成9字形，其他手指握拳。',
                tips: '弯曲的食指要清晰可见。'
            }
        };
        
        const guide = gestureGuides[gesture] || {
            description: `标准姿势展示手势 ${gesture}`,
            tips: '保持手部稳定，在摄像头范围内清晰展示。'
        };
        
        // 更新指南区域
        const guideElement = document.getElementById('gesture-guide');
        if (guideElement) {
            guideElement.innerHTML = `
                <div class="p-4 bg-primary/5 rounded-lg">
                    <h4 class="font-bold text-primary mb-2">手势 ${gesture} 学习指南</h4>
                    <p class="text-sm text-slate-700 mb-3">${guide.description}</p>
                    <div class="bg-white p-2 rounded border border-primary/20">
                        <p class="text-xs text-slate-600">
                            <i class="fa fa-lightbulb-o text-yellow-500 mr-1"></i>
                            <strong>小贴士：</strong> ${guide.tips}
                        </p>
                    </div>
                </div>
            `;
        }
    },
    
    // 切换主题
    toggleTheme() {
        this.state.isDarkMode = !this.state.isDarkMode;
        
        if (this.state.isDarkMode) {
            document.body.classList.add('dark-mode');
            if (document.getElementById('theme-toggle')) {
                const icon = document.getElementById('theme-toggle').querySelector('i');
                icon.classList.remove('fa-moon-o');
                icon.classList.add('fa-sun-o');
            }
        } else {
            document.body.classList.remove('dark-mode');
            if (document.getElementById('theme-toggle')) {
                const icon = document.getElementById('theme-toggle').querySelector('i');
                icon.classList.remove('fa-sun-o');
                icon.classList.add('fa-moon-o');
            }
        }
        
        // 保存主题设置
        this.saveUserSettings();
    },
    
    // 清除历史记录
    async clearHistory() {
        try {
            await this.sendApiRequest('/api/clear_history', 'POST');
            this.state.lastPredictions = [];
            this.updateHistoryTable();
            this.showNotification('历史记录已清除', 'success');
        } catch (error) {
            console.error('清除历史记录失败:', error);
            this.showNotification('清除历史记录失败', 'error');
        }
    },
    
    // 刷新摄像头
    refreshCamera() {
        const videoFrame = document.getElementById('video-frame');
        if (videoFrame) {
            videoFrame.src = '/video_feed?' + new Date().getTime();
            this.showNotification('摄像头已刷新', 'info');
        }
    },
    
    // 切换关键点显示
    async toggleLandmarks() {
        this.state.showLandmarks = !this.state.showLandmarks;
        
        try {
            await this.sendSettingsUpdate({ show_landmarks: this.state.showLandmarks });
            
            // 更新按钮图标
            const toggleBtn = document.getElementById('toggle-landmarks');
            if (toggleBtn) {
                const icon = toggleBtn.querySelector('i');
                if (this.state.showLandmarks) {
                    icon.classList.remove('fa-circle-o');
                    icon.classList.add('fa-dot-circle-o');
                    this.showNotification('已显示手部关键点', 'info');
                } else {
                    icon.classList.remove('fa-dot-circle-o');
                    icon.classList.add('fa-circle-o');
                    this.showNotification('已隐藏手部关键点', 'info');
                }
            }
        } catch (error) {
            console.error('切换关键点显示失败:', error);
            this.showNotification('切换关键点显示失败', 'error');
        }
    },
    
    // 启动状态更新循环
    startStateUpdates() {
        setInterval(() => this.updateSystemState(), 100);
    },
    
    // 更新系统状态
    async updateSystemState() {
        try {
            const data = await this.sendApiRequest('/api/state', 'GET');
            
            // 更新UI显示
            this.updatePredictionDisplay(data.last_prediction, data.last_confidence);
            
            // 更新历史记录
            if (data.prediction_history && data.prediction_history.length > this.state.lastPredictions.length) {
                this.state.lastPredictions = data.prediction_history;
                this.updateHistoryTable();
                
                // 更新统计信息
                this.updateStats(data.last_prediction, data.last_confidence);
            }
            
            // 更新模式状态
            if (data.mode !== this.state.currentMode) {
                this.state.currentMode = data.mode;
                this.updateModeUI();
            }
            
            // 更新其他状态
            this.state.showLandmarks = data.show_landmarks;
            this.state.confidenceThreshold = data.confidence_threshold;
            this.state.currentGesture = data.current_gesture;
            
        } catch (error) {
            console.error('更新系统状态失败:', error);
        }
    },
    
    // 更新预测显示
    updatePredictionDisplay(prediction, confidence) {
        const predictionElement = document.getElementById('current-prediction');
        const confidenceElement = document.getElementById('current-confidence');
        
        if (!predictionElement || !confidenceElement) return;
        
        if (prediction && confidence >= this.state.confidenceThreshold) {
            predictionElement.textContent = `识别结果: ${prediction}`;
            predictionElement.className = 'text-xl font-bold text-primary';
            
            confidenceElement.textContent = `置信度: ${(confidence * 100).toFixed(0)}%`;
            
            // 根据置信度设置颜色
            if (confidence > 0.8) {
                confidenceElement.className = 'text-sm opacity-80 text-green-600';
            } else if (confidence > 0.6) {
                confidenceElement.className = 'text-sm opacity-80 text-blue-600';
            } else {
                confidenceElement.className = 'text-sm opacity-80 text-yellow-600';
            }
            
            // 添加动画效果
            predictionElement.classList.add('animate-fade-in');
            setTimeout(() => {
                predictionElement.classList.remove('animate-fade-in');
            }, 500);
            
        } else if (prediction) {
            predictionElement.textContent = '置信度过低';
            predictionElement.className = 'text-xl font-bold text-yellow-600';
            confidenceElement.textContent = `置信度: ${(confidence * 100).toFixed(0)}%`;
            confidenceElement.className = 'text-sm opacity-80 text-yellow-600';
        } else {
            predictionElement.textContent = '等待识别...';
            predictionElement.className = 'text-xl font-bold text-slate-500';
            confidenceElement.textContent = '置信度: 0%';
            confidenceElement.className = 'text-sm opacity-80 text-slate-500';
        }
    },
    
    // 更新历史记录表
    updateHistoryTable() {
        const tableBody = document.getElementById('history-table-body');
        const historyCount = document.getElementById('history-count');
        
        if (!tableBody || !historyCount) return;
        
        if (this.state.lastPredictions.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="py-8 text-center text-slate-500">
                        <i class="fa fa-history text-3xl mb-2 block opacity-30"></i>
                        暂无识别历史
                    </td>
                </tr>
            `;
            historyCount.textContent = '0 条记录';
            return;
        }
        
        historyCount.textContent = `${this.state.lastPredictions.length} 条记录`;
        
        let rows = '';
        this.state.lastPredictions.slice(-10).forEach((item, index) => {
            const timestamp = new Date().toLocaleTimeString();
            const confidenceClass = item[1] > 0.7 ? 'text-green-600' : item[1] > 0.5 ? 'text-orange-600' : 'text-red-600';
            
            rows += `
                <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                    <td class="py-3 px-4 text-sm text-slate-500">${index + 1}</td>
                    <td class="py-3 px-4 font-medium">${item[0]}</td>
                    <td class="py-3 px-4 text-sm ${confidenceClass}">${(item[1] * 100).toFixed(0)}%</td>
                    <td class="py-3 px-4 text-sm text-slate-500">${timestamp}</td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = rows;
    },
    
    // 更新统计信息
    updateStats(prediction, confidence) {
        if (prediction && confidence >= this.state.confidenceThreshold) {
            this.state.stats.totalPredictions++;
            
            // 如果在学习模式下，且识别结果与当前学习的手势匹配，计为正确
            if (this.state.currentMode === 'learning' && prediction === this.state.currentGesture) {
                this.state.stats.correctPredictions++;
            }
            
            // 更新准确率
            this.state.stats.accuracy = this.state.stats.correctPredictions / this.state.stats.totalPredictions;
            this.state.stats.lastUpdated = new Date();
            
            // 更新UI显示（如果有统计区域）
            this.updateStatsUI();
        }
    },
    
    // 更新统计UI
    updateStatsUI() {
        const statsElement = document.getElementById('recognition-stats');
        if (!statsElement) return;
        
        statsElement.innerHTML = `
            <div class="grid grid-cols-3 gap-4 text-center">
                <div class="p-3 bg-slate-50 rounded-lg">
                    <p class="text-sm text-slate-500">总识别次数</p>
                    <p class="text-lg font-bold">${this.state.stats.totalPredictions}</p>
                </div>
                <div class="p-3 bg-slate-50 rounded-lg">
                    <p class="text-sm text-slate-500">正确识别</p>
                    <p class="text-lg font-bold text-green-600">${this.state.stats.correctPredictions}</p>
                </div>
                <div class="p-3 bg-slate-50 rounded-lg">
                    <p class="text-sm text-slate-500">准确率</p>
                    <p class="text-lg font-bold text-primary">${(this.state.stats.accuracy * 100).toFixed(0)}%</p>
                </div>
            </div>
        `;
    },
    
    // 发送API请求
    async sendApiRequest(endpoint, method = 'GET', data = null) {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(endpoint, options);
        
        if (!response.ok) {
            throw new Error(`API错误: ${response.status}`);
        }
        
        return await response.json();
    },
    
    // 发送设置更新
    async sendSettingsUpdate(settings) {
        return await this.sendApiRequest('/api/settings', 'POST', settings);
    },
    
    // 显示通知
    showNotification(message, type = 'info') {
        // 检查是否已存在通知容器，如果没有则创建
        let notificationContainer = document.getElementById('notification-container');
        if (!notificationContainer) {
            notificationContainer = document.createElement('div');
            notificationContainer.id = 'notification-container';
            notificationContainer.className = 'fixed bottom-4 right-4 z-50 flex flex-col gap-2';
            document.body.appendChild(notificationContainer);
        }
        
        // 创建通知元素
        const notification = document.createElement('div');
        
        // 设置样式和内容
        let bgColor = 'bg-blue-500';
        let iconClass = 'fa-info-circle';
        
        switch (type) {
            case 'success':
                bgColor = 'bg-green-500';
                iconClass = 'fa-check-circle';
                break;
            case 'error':
                bgColor = 'bg-red-500';
                iconClass = 'fa-exclamation-circle';
                break;
            case 'warning':
                bgColor = 'bg-yellow-500';
                iconClass = 'fa-exclamation-triangle';
                break;
        }
        
        notification.className = `${bgColor} text-white px-4 py-3 rounded-lg shadow-lg transform transition-all duration-300 translate-y-2 opacity-0 flex items-center gap-3`;
        notification.innerHTML = `
            <i class="fa ${iconClass}"></i>
            <span>${message}</span>
        `;
        
        // 添加到容器
        notificationContainer.appendChild(notification);
        
        // 显示动画
        setTimeout(() => {
            notification.classList.remove('translate-y-2', 'opacity-0');
        }, 10);
        
        // 自动关闭
        setTimeout(() => {
            notification.classList.add('translate-y-2', 'opacity-0');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    },
    
    // 处理键盘快捷键
    handleKeyboardShortcuts(e) {
        // 只在没有聚焦输入元素时处理快捷键
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            return;
        }
        
        switch (e.key.toLowerCase()) {
            case 'r':
                if (e.ctrlKey || e.metaKey) return; // 避免与浏览器刷新冲突
                this.switchMode('recognition');
                break;
            case 'l':
                if (e.ctrlKey || e.metaKey) return;
                this.switchMode('learning');
                break;
            case 't':
                this.toggleTheme();
                break;
            case 'h':
                this.toggleLandmarks();
                break;
            case 'c':
                if (e.ctrlKey || e.metaKey) return;
                this.clearHistory();
                break;
            case 'f5':
            case ' ': // 空格键刷新摄像头
                if (e.key === ' ') {
                    e.preventDefault();
                    this.refreshCamera();
                }
                break;
        }
    },
    
    // 保存用户设置
    saveUserSettings() {
        try {
            localStorage.setItem('signLanguageSettings', JSON.stringify({
                darkMode: this.state.isDarkMode,
                showLandmarks: this.state.showLandmarks,
                confidenceThreshold: this.state.confidenceThreshold
            }));
        } catch (error) {
            console.error('保存设置失败:', error);
        }
    },
    
    // 加载用户设置
    loadUserSettings() {
        try {
            const savedSettings = localStorage.getItem('signLanguageSettings');
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);
                
                // 应用保存的设置
                if (settings.darkMode) {
                    this.state.isDarkMode = true;
                    this.toggleTheme(); // 应用暗色主题
                }
                
                if (typeof settings.showLandmarks === 'boolean') {
                    this.state.showLandmarks = settings.showLandmarks;
                }
                
                if (typeof settings.confidenceThreshold === 'number') {
                    this.state.confidenceThreshold = settings.confidenceThreshold;
                    // 更新滑块值
                    const slider = document.getElementById('confidence-slider');
                    if (slider) {
                        slider.value = settings.confidenceThreshold;
                        const valueDisplay = document.getElementById('threshold-value');
                        if (valueDisplay) {
                            valueDisplay.textContent = `${Math.round(settings.confidenceThreshold * 100)}%`;
                        }
                    }
                }
            }
        } catch (error) {
            console.error('加载设置失败:', error);
        }
    }
};

// 手势卡片生成函数
function generateGestureCards() {
    const gestureContainer = document.getElementById('gesture-container');
    if (!gestureContainer) return;
    
    // 字母手势（A-Z）
    const letters = Array.from({length: 26}, (_, i) => String.fromCharCode(65 + i));
    // 数字手势（0-9）
    const numbers = Array.from({length: 10}, (_, i) => String(i));
    
    // 合并所有手势
    const allGestures = [...letters, ...numbers];
    
    let cardsHtml = '';
    allGestures.forEach(gesture => {
        cardsHtml += `
            <div class="gesture-card border border-slate-200 rounded-lg p-4 text-center cursor-pointer transition-all hover:border-primary hover:bg-primary/5 hover:scale-[1.02]">
                <span class="text-2xl font-bold block mb-1">${gesture}</span>
                <span class="text-xs text-slate-500">点击学习</span>
            </div>
        `;
    });
    
    gestureContainer.innerHTML = cardsHtml;
    
    // 为生成的卡片绑定事件
    document.querySelectorAll('.gesture-card').forEach(card => {
        card.addEventListener('click', (e) => {
            const gesture = e.currentTarget.querySelector('span:first-child').textContent;
            App.selectGesture(gesture);
        });
    });
}

// 添加手势演示功能
function addGestureDemonstration() {
    const demoBtn = document.getElementById('demo-gestures');
    if (demoBtn) {
        demoBtn.addEventListener('click', () => {
            App.showNotification('手势演示功能正在开发中', 'info');
        });
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    // 延迟初始化以确保所有DOM元素都已加载
    setTimeout(() => {
        // 生成手势卡片
        generateGestureCards();
        
        // 添加手势演示功能
        addGestureDemonstration();
        
        // 初始化应用
        App.init();
    }, 100);
});

// 扩展动画效果
CSS.registerProperty && CSS.registerProperty({
    name: '--animate-delay',
    syntax: '<time>',
    inherits: false,
    initialValue: '0ms'
});

// 性能监控函数
function startPerformanceMonitoring() {
    if (window.performance && window.performance.now) {
        let lastFrameTime = performance.now();
        
        function updateFPS() {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = Math.round(1000 / deltaTime);
            lastFrameTime = currentTime;
            
            // 更新FPS显示（如果有显示区域）
            const fpsElement = document.getElementById('fps-counter');
            if (fpsElement) {
                fpsElement.textContent = `FPS: ${fps}`;
                
                // 根据FPS设置颜色
                if (fps >= 50) {
                    fpsElement.className = 'text-xs text-green-600';
                } else if (fps >= 30) {
                    fpsElement.className = 'text-xs text-yellow-600';
                } else {
                    fpsElement.className = 'text-xs text-red-600';
                }
            }
            
            requestAnimationFrame(updateFPS);
        }
        
        requestAnimationFrame(updateFPS);
    }
}

// 启动性能监控
if (typeof window !== 'undefined') {
    window.addEventListener('load', startPerformanceMonitoring);
}

// 为移动设备添加触摸优化
function addMobileOptimizations() {
    if (App && App.state && App.state.isMobile) {
        // 为触摸事件添加延迟以改善交互体验
        document.body.classList.add('touch-action-manipulation');
        
        // 为按钮添加更大的点击区域
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            if (button.offsetWidth < 44 || button.offsetHeight < 44) {
                button.classList.add('mobile-friendly-button');
            }
        });
    }
}

// 当设备类型改变时应用移动优化
if (window.addEventListener) {
    window.addEventListener('resize', addMobileOptimizations);
}
