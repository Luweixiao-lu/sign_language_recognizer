"""
手语识别Web应用
使用Flask作为后端，提供视频流和实时识别功能
"""
import os
import cv2
import numpy as np
import traceback
from flask import Flask, render_template, Response, jsonify, request
from hand_landmarks import HandLandmarkDetector
from gesture_classifier import GestureClassifier

# 创建Flask应用
app = Flask(__name__)

# 初始化识别器组件
detector = HandLandmarkDetector()
classifier = GestureClassifier()

# 全局状态
class AppState:
    def __init__(self):
        self.mode = 'recognition'  # 'recognition' 或 'learning'
        self.current_gesture = None  # 当前学习的手势
        self.prediction_history = []
        self.history_size = 20
        self.confidence_threshold = 0.6
        self.show_landmarks = True
        self.last_prediction = None
        self.last_confidence = 0.0
        self.gesture_statistics = {}
        self.smooth_predictions = []
        self.smooth_window = 3
        self.performance_metrics = {
            'frame_rate': 0,
            'detection_time': 0,
            'classification_time': 0
        }
        self.hand_detected = False
        self.is_recording = False
        self.demo_gesture = None
        self.is_running = False  # 识别是否正在运行

# 应用状态
state = AppState()

# 生成视频流的函数
def generate_frames():
    import time
    
    # 等待识别启动信号
    while not state.is_running:
        # 创建一个黑色占位图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 将图像编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        # 返回MJPEG流格式
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.1)  # 避免CPU占用过高
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while state.is_running:
            # 计算帧率
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                state.performance_metrics['frame_rate'] = round(30 / elapsed, 1) if elapsed > 0 else 0
                start_time = time.time()
                frame_count = 0
            
            # 读取一帧
            success, frame = cap.read()
            if not success:
                break
            
            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 检测手部关键点
            detection_start = time.time()
            landmarks, annotated_frame = detector.detect(frame)
            state.performance_metrics['detection_time'] = round((time.time() - detection_start) * 1000, 2)  # 毫秒
            
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
                    state.performance_metrics['classification_time'] = round((time.time() - classification_start) * 1000, 2)  # 毫秒
                    
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
                            
                            # 根据模式进行不同处理
                            if state.mode == 'recognition' and confidence > state.confidence_threshold:
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
                
                # 不同模式下的显示效果 - 不显示文字
                if state.mode == 'learning' and state.current_gesture:
                    match = prediction == state.current_gesture
                    # 不再显示匹配信息文字
                
                # 不再显示预测结果和置信度文字
            elif prediction is not None:
                # 不再显示低置信度文字
                pass
            else:
                # 不再显示状态文字
                pass
            
            # 不再显示模式信息文字
            
            # 不再显示性能信息文字
            
            # 将图像编码为JPEG格式
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            # 返回MJPEG流格式
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"视频流处理错误: {e}")
        traceback.print_exc()
    finally:
        cap.release()

# 路由定义
@app.route('/')
def index():
    return render_template('index.html', gestures=classifier.LABELS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state', methods=['GET'])
def get_state():
    # 计算统计数据
    total_predictions = sum(state.gesture_statistics.values())
    
    return jsonify({
        'mode': state.mode,
        'current_gesture': state.current_gesture,
        'prediction_history': state.prediction_history,
        'last_prediction': state.last_prediction,
        'last_confidence': state.last_confidence,
        'show_landmarks': state.show_landmarks,
        'confidence_threshold': state.confidence_threshold,
        'gesture_statistics': state.gesture_statistics,
        'total_predictions': total_predictions,
        'performance_metrics': state.performance_metrics,
        'hand_detected': state.hand_detected,
        'is_recording': state.is_recording,
        'demo_gesture': state.demo_gesture
    })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    
    if 'mode' in data:
        state.mode = data['mode']
    
    if 'current_gesture' in data:
        state.current_gesture = data['current_gesture']
    
    if 'show_landmarks' in data:
        state.show_landmarks = bool(data['show_landmarks'])
    
    if 'confidence_threshold' in data:
        state.confidence_threshold = float(data['confidence_threshold'])
    
    if 'smooth_window' in data:
        state.smooth_window = max(1, min(10, int(data['smooth_window'])))
    
    if 'is_recording' in data:
        state.is_recording = bool(data['is_recording'])
    
    if 'demo_gesture' in data:
        state.demo_gesture = data['demo_gesture']
    
    return jsonify({'status': 'success'})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    state.prediction_history = []
    return jsonify({'status': 'success'})

@app.route('/api/reset_statistics', methods=['POST'])
def reset_statistics():
    state.gesture_statistics = {}
    return jsonify({'status': 'success'})

@app.route('/api/all_gestures', methods=['GET'])
def get_all_gestures():
    return jsonify({
        'gestures': classifier.LABELS,
        'total_gestures': len(classifier.LABELS)
    })

@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    """切换识别状态的路由"""
    try:
        data = request.json
        if 'is_running' in data:
            state.is_running = bool(data['is_running'])
            print(f"识别状态已更新: {'启动' if state.is_running else '停止'}")
            return jsonify({'status': 'success', 'message': f'识别已{"启动" if state.is_running else "停止"}'})
        else:
            return jsonify({'status': 'error', 'message': '缺少is_running参数'}), 400
    except Exception as e:
        print(f"切换识别状态出错: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 启动应用
if __name__ == '__main__':
    # 确保templates和static目录存在
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(host='0.0.0.0', port=5001, debug=True)
