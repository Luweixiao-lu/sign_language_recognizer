import streamlit as st
import cv2
import numpy as np
import time
from hand_landmarks import HandLandmarkDetector
from gesture_classifier import GestureClassifier
import threading
import queue

# 设置页面配置
st.set_page_config(
    page_title="汉语手指字母识别",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化检测器和分类器
@st.cache_resource
def init_models():
    detector = HandLandmarkDetector()
    classifier = GestureClassifier()
    return detector, classifier

detector, classifier = init_models()

# 应用状态管理
class AppState:
    def __init__(self):
        self.mode = 'recognition'  # recognition or learning
        self.current_gesture = None
        self.prediction_history = []
        self.last_prediction = None
        self.last_confidence = 0.0
        self.show_landmarks = True
        self.confidence_threshold = 0.6
        self.gesture_statistics = {}
        self.performance_metrics = {
            'frame_rate': 0,
            'detection_time': 0,
            'classification_time': 0
        }
        self.hand_detected = False
        self.smooth_predictions = []
        self.smooth_window = 5
        self.history_size = 20
        self.is_running = False
        self.is_recording = False
        self.demo_gesture = None

state = AppState()

# 视频流处理线程
class VideoStreamProcessor:
    def __init__(self):
        self.video_queue = queue.Queue(maxsize=10)
        self.running = False
        self.cap = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
    
    def process_video(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("无法打开摄像头")
            return
        
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            # 计算帧率
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                state.performance_metrics['frame_rate'] = round(30 / elapsed, 1) if elapsed > 0 else 0
                start_time = time.time()
                frame_count = 0
            
            # 读取一帧
            success, frame = self.cap.read()
            if not success:
                break
            
            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 检测手部关键点
            detection_start = time.time()
            landmarks, annotated_frame = detector.detect(frame)
            state.performance_metrics['detection_time'] = round((time.time() - detection_start) * 1000, 2)
            
            # 更新手部检测状态
            state.hand_detected = landmarks is not None
            
            # 识别手势
            prediction = None
            confidence = 0.0
            
            if landmarks is not None:
                # 提取特征
                features = detector.extract_features(landmarks)
                
                if features is not None:
                    # 预测手势
                    classification_start = time.time()
                    prediction, confidence = classifier.predict(features)
                    state.performance_metrics['classification_time'] = round((time.time() - classification_start) * 1000, 2)
                    
                    # 保存预测结果
                    state.last_prediction = prediction
                    state.last_confidence = confidence
                    
                    # 预测平滑处理
                    state.smooth_predictions.append((prediction, confidence))
                    if len(state.smooth_predictions) > state.smooth_window:
                        state.smooth_predictions.pop(0)
                    
                    # 基于滑动窗口的平滑预测
                    if len(state.smooth_predictions) >= state.smooth_window:
                        # 计算窗口内的预测统计
                        gesture_counts = {}
                        for gest, conf in state.smooth_predictions:
                            if conf > state.confidence_threshold:
                                gesture_counts[gest] = gesture_counts.get(gest, 0) + 1
                        
                        # 选择最常见的手势
                        if gesture_counts:
                            smoothed_prediction = max(gesture_counts, key=gesture_counts.get)
                            
                            # 更新统计信息
                            state.gesture_statistics[smoothed_prediction] = state.gesture_statistics.get(smoothed_prediction, 0) + 1
                            
                            # 添加到历史记录
                            state.prediction_history.append((smoothed_prediction, confidence))
                            if len(state.prediction_history) > state.history_size:
                                state.prediction_history.pop(0)
            
            # 如果不需要显示关键点，使用原始帧
            if not state.show_landmarks:
                annotated_frame = frame.copy()
            
            # 显示结果
            if prediction is not None and confidence > state.confidence_threshold:
                # 使用平滑后的预测结果
                if len(state.smooth_predictions) >= state.smooth_window:
                    gesture_counts = {}
                    for gest, conf in state.smooth_predictions:
                        if conf > state.confidence_threshold:
                            gesture_counts[gest] = gesture_counts.get(gest, 0) + 1
                    
                    if gesture_counts:
                        smoothed_prediction = max(gesture_counts, key=gesture_counts.get)
                        prediction = smoothed_prediction
                
                # 显示预测结果
                text = f"手势: {prediction}"
                confidence_text = f"置信度: {confidence:.1%}"
                
                # 根据置信度选择颜色
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                
                # 显示文本
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(annotated_frame, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 将BGR转换为RGB
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # 更新队列
            if self.video_queue.full():
                self.video_queue.get()
            self.video_queue.put(rgb_frame)

# 初始化视频处理器
video_processor = VideoStreamProcessor()

# 页面标题
st.title("👋 汉语手指字母识别")

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    
    # 显示/隐藏手部关键点
    state.show_landmarks = st.checkbox("显示手部关键点", value=True)
    
    # 置信度阈值滑块
    state.confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.05
    )
    
    # 平滑窗口大小
    state.smooth_window = st.slider(
        "平滑窗口大小",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    # 应用模式
    state.mode = st.radio(
        "应用模式",
        ['recognition', 'learning'],
        index=0
    )
    
    # 学习模式下的手势选择
    if state.mode == 'learning':
        state.current_gesture = st.selectbox(
            "选择要学习的手势",
            classifier.LABELS if hasattr(classifier, 'LABELS') else []
        )
    
    # 开始/停止按钮
    if st.button("开始识别", type="primary"):
        if not state.is_running:
            state.is_running = True
            video_processor.start()
    
    if st.button("停止识别"):
        if state.is_running:
            state.is_running = False
            video_processor.stop()
    
    # 清除历史按钮
    if st.button("清除历史"):
        state.prediction_history = []
        state.gesture_statistics = {}

# 主内容区
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("实时视频流")
    video_placeholder = st.empty()
    
with col2:
    st.subheader("识别结果")
    result_placeholder = st.empty()
    
    st.subheader("性能指标")
    metrics_placeholder = st.empty()
    
    st.subheader("识别统计")
    stats_placeholder = st.empty()

# 主循环
while True:
    # 显示视频流
    if state.is_running and not video_processor.video_queue.empty():
        frame = video_processor.video_queue.get()
        video_placeholder.image(frame, use_column_width=True)
    
    # 显示识别结果
    with result_placeholder:
        if state.last_prediction is not None:
            st.markdown(f"### 🎯 最近识别: {state.last_prediction}")
            st.progress(state.last_confidence)
            st.text(f"置信度: {state.last_confidence:.1%}")
            
            # 显示预测历史
            if state.prediction_history:
                st.text("\n预测历史:")
                history_data = [(i+1, gest, f"{conf:.1%}") for i, (gest, conf) in enumerate(reversed(state.prediction_history[-5:]))]
                st.table(history_data)
        else:
            st.info("等待手势识别...")
    
    # 显示性能指标
    with metrics_placeholder:
        col1, col2, col3 = st.columns(3)
        col1.metric("帧率", f"{state.performance_metrics['frame_rate']} FPS")
        col2.metric("检测时间", f"{state.performance_metrics['detection_time']} ms")
        col3.metric("分类时间", f"{state.performance_metrics['classification_time']} ms")
    
    # 显示统计信息
    with stats_placeholder:
        if state.gesture_statistics:
            st.bar_chart(state.gesture_statistics)
        else:
            st.info("暂无统计数据")
    
    # 检查应用是否在运行
    if not state.is_running:
        # 显示占位图像
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "点击开始识别按钮启动", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        video_placeholder.image(placeholder_img, use_column_width=True)
    
    # 等待一小段时间
    time.sleep(0.01)
