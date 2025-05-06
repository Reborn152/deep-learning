import os
import cv2
import torch
import numpy as np
import logging
import time
import queue
import faiss
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
from ultralytics import YOLO
from models import SwinIR, ViT_FaceNet

# ----------------------
# 配置文件
# ----------------------
CONFIG = {
    "sr_model_path": "weights/swinir_sr4x.pth",
    "face_db_path": "database/faces/",
    "vote_threshold": 1,
    "frame_buffer_size": 15,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "enroll_samples": 5,
    "quality_threshold": 120,
    "faiss_index_path": "database/faiss_index.bin",
    "processing_size": (640, 480),
    "min_face_size": 80,  # 新增最小人脸尺寸
    "blur_threshold": 80,  # 新增模糊阈值
    "max_enroll_time": 120  # 注册超时时间
}

# ----------------------
# 日志配置
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('attendance.log'), logging.StreamHandler()]
)

# 抑制OpenCV的ICC警告
os.environ["OPENCV_IO_IGNORE_ICC_PROFILE"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


# ----------------------
# 核心系统组件
# ----------------------
class FaceProcessingPipeline:
    def __init__(self):
        self.detection_queue = Queue(maxsize=5)
        self.recognition_queue = Queue(maxsize=5)
        self.lock = Lock()

        # 初始化模型
        self.detector = YOLO('weights/yolo11n.pt').to(CONFIG['device'])  # 修正模型名称
        self.recognizer = ViT_FaceNet().to(CONFIG['device'])

        # 修正SwinIR参数
        self.sr_model = SwinIR(
            upscale=4,
            in_chans=3,
            embed_dim=180  # 使用官方推荐的48维
        ).to(CONFIG['device'])

        # 优化权重加载
        state_dict = torch.load(CONFIG['sr_model_path'])
        if 'params' in state_dict:
            state_dict = state_dict['params']
        # 去除前缀（兼容不同训练方式）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.sr_model.load_state_dict(state_dict, strict=False)

        # 特征数据库
        self.faiss_index, self.id_map = self._init_database()
        self.attendance_records = {}

    def _init_database(self):
        if os.path.exists(CONFIG['faiss_index_path']):
            index = faiss.read_index(CONFIG['faiss_index_path'])
            with open(os.path.join(CONFIG['face_db_path'], 'id_map.txt'), 'r') as f:
                id_map = [line.strip() for line in f]
            return index, id_map
        return faiss.IndexFlatIP(512), []

    def detection_worker(self):
        while True:
            frame = self.detection_queue.get()
            if frame is None:
                break

            resized = cv2.resize(frame, CONFIG['processing_size'])
            results = self.detector.track(
                resized,
                persist=True,
                classes=0,  # 只检测人脸类
                conf=0.7,
                iou=0.5,
                verbose=False
            )[0]
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)

            if len(boxes) > 0:
                self.recognition_queue.put((frame, boxes))

                # 在 FaceProcessingPipeline 类的 recognition_worker 方法中
    def recognition_worker(self):
        while True:
            data = self.recognition_queue.get()
            if data is None:
                break

            frame, boxes = data
            for box in boxes:
                x1, y1, x2, y2 = box
                face_roi = frame[y1:y2, x1:x2]

                 # 超分辨率增强
            if max(x2 - x1, y2 - y1) < 100:
             with torch.no_grad():
                lr_tensor = ViT_FaceNet.preprocess(face_roi).unsqueeze(0).to(CONFIG['device'])
                sr_tensor = self.sr_model(lr_tensor)
                face_roi = (sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8)


                # 特征提取
                tensor = ViT_FaceNet.preprocess(face_roi).unsqueeze(0).to(CONFIG['device'])
                with torch.no_grad():
                    feature = self.pipeline.recognizer(tensor).cpu().numpy()
                feature.append(feature)

                # 数据库检索
                distances, indices = self.faiss_index.search(feature, 1)
                student_id = "unknown"  # 给 student_id 赋默认值
                if distances[0][0] > 0.6:
                    student_id = self.id_map[indices[0][0]]
                    self._record_attendance(student_id)

                # 绘制结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{student_id} {distances[0][0]:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _record_attendance(self, student_id):
        timestamp = datetime.now().isoformat()
        self.attendance_records.setdefault(student_id, []).append(timestamp)
        logging.info(f"记录 {student_id} 的考勤，当前记录数量: {len(self.attendance_records[student_id])}")

        if len(self.attendance_records[student_id]) >= CONFIG['vote_threshold']:
            logging.info(f"学生 {student_id} 考勤已确认")
            del self.attendance_records[student_id]


class StudentEnrollment:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()

    def enroll(self, student_id):
        features = []
        collected = 0

        while collected < CONFIG['enroll_samples'] and (time.time() - self.start_time) < CONFIG['max_enroll_time']:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 人脸检测和质量筛选
            resized = cv2.resize(frame, CONFIG['processing_size'])
            results = self.pipeline.detector(resized, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)

            if len(boxes) == 1:
                x1, y1, x2, y2 = boxes[0]
                face_roi = frame[y1:y2, x1:x2]

                # 质量检查
                if (x2 - x1) < CONFIG['min_face_size'] or cv2.Laplacian(face_roi, cv2.CV_64F).var() < CONFIG[
                    'blur_threshold']:
                    continue

                # 特征提取
                tensor = ViT_FaceNet.preprocess(face_roi).unsqueeze(0).to(CONFIG['device'])
                with torch.no_grad():
                    feature = self.pipeline.recognizer(tensor).cpu().numpy()
                features.append(feature)
                collected += 1

                # 显示采集进度
                cv2.putText(frame, f"Collecting: {collected}/{CONFIG['enroll_samples']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Enrollment', frame)
                cv2.waitKey(300)  # 间隔300ms采集下一张

        # 保存有效特征
        if features:
            avg_feature = np.mean(features, axis=0)
            with self.pipeline.lock:
                self.pipeline.faiss_index.add(avg_feature)
                self.pipeline.id_map.append(student_id)
                faiss.write_index(self.pipeline.faiss_index, CONFIG['faiss_index_path'])
                with open(os.path.join(CONFIG['face_db_path'], 'id_map.txt'), 'a') as f:
                    f.write(f"{student_id}\n")
            logging.info(f"学号 {student_id} 注册成功")

        self.cap.release()
        cv2.destroyAllWindows()


# ----------------------
# 运行入口
# ----------------------
if __name__ == "__main__":
    pipeline = FaceProcessingPipeline()

    mode = input("请选择模式 [1]考勤 [2]注册: ").strip()

    if mode == "2":
        student_id = input("请输入学号: ").strip()
        if not student_id.isalnum():
            print("无效学号! 只能包含数字")
            exit()
        StudentEnrollment(pipeline).enroll(student_id)
    else:
        detect_thread = Thread(target=pipeline.detection_worker)
        recog_thread = Thread(target=pipeline.recognition_worker)
        detect_thread.start()
        recog_thread.start()

        try:
            cap = cv2.VideoCapture(0)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Attendance System', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

                try:
                    # 使用非阻塞方式放入队列
                    pipeline.detection_queue.put(frame, block=False)
                    frame_count += 1
                    if frame_count % 10 == 0:
                        logging.info(f"已处理 {frame_count} 帧")
                except queue.Full:  # 修改为正确的异常捕获
                    # 队列满时跳过当前帧
                    continue

        except KeyboardInterrupt:
            logging.info("用户中断程序，正在退出...")
        finally:
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            pipeline.detection_queue.put(None)
            pipeline.recognition_queue.put(None)
            detect_thread.join()
            recog_thread.join()