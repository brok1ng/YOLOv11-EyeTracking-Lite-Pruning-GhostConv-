import cv2
import numpy as np
from ultralytics import YOLO

class YOLOInfer:
    def __init__(self, model_path=None, device="cuda"):
        self.model = None
        self.device = device
        self.class_names = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        print(f"[YOLOInfer] 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names if hasattr(self.model, "names") else None

    def denoise_func(self, img, method=None):
        if method is None or method == "none":
            return img
        elif method == "gaussian":
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(img, 5)
        elif method == "mean":
            return cv2.blur(img, (5, 5))
        else:
            return img

    def infer_image(self, img_path, denoise_method=None):
        print(f"[YOLOInfer] infer_image on {img_path}, denoise_method={denoise_method}")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法打开图片文件: {img_path}")
        img = self.denoise_func(img, method=denoise_method)
        results = self.model.predict(img, device=self.device, verbose=False)
        annotated = self.draw_boxes_noid(results[0], img)
        return results[0], annotated

    def infer_video(self, video_path, on_result=None, denoise_method=None, tracker="bytetrack.yaml"):
        print(f"[YOLOInfer] infer_video on {video_path}, denoise_method={denoise_method}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {video_path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[YOLOInfer] 视频读取结束或出错")
                break
            frame_idx += 1
            print(f"[YOLOInfer] 处理第 {frame_idx} 帧")
            frame = self.denoise_func(frame, method=denoise_method)
            try:
                results, annotated = self.track(frame, tracker=tracker)
            except Exception as e:
                print(f"[YOLOInfer] track异常: {e}")
                continue
            if on_result:
                on_result(frame, annotated, results)
        cap.release()

    def track(self, frame, tracker="bytetrack.yaml"):
        print("[YOLOInfer] 调用模型 track 方法")
        results = self.model.track(
            frame,
            tracker=tracker,
            persist=True,
            imgsz=640,
            device=self.device,
            verbose=False
        )
        print("[YOLOInfer] track 返回，准备draw_boxes_noid")
        annotated = self.draw_boxes_noid(results[0], frame)
        return results[0], annotated

    def draw_boxes_noid(self, results, frame, color=(255, 0, 0)):
        img = frame.copy()
        if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = confs[i]
                cls = clss[i]
                label = str(cls) if self.class_names is None else self.class_names[cls]
                text = f"{label} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
                cv2.putText(img, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        else:
            print("[YOLOInfer] draw_boxes_noid: 没有检测到任何目标")
        return img