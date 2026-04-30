import cv2
from ultralytics import YOLO
import numpy as np

def draw_boxes_noid(results, frame, class_names=None, color=(255, 0, 0)):
    """
    可视化检测结果：只显示label和置信度，不显示track id
    """
    img = frame.copy()
    if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()   # (N, 4)
        confs = results.boxes.conf.cpu().numpy()   # (N,)
        clss = results.boxes.cls.cpu().numpy().astype(int)  # (N,)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = confs[i]
            cls = clss[i]
            label = str(cls) if class_names is None else class_names[cls]
            text = f"{label} {conf:.2f}"
            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # 画文字
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return img

def detect_and_save_fixed_frame(model, video_path, fixed_frame=100, device="cuda", tracker="bytetrack.yaml"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0

    # 加载类别名（如果可用）
    class_names = model.names if hasattr(model, "names") else None

    print(f"开始检测并在第{fixed_frame}帧截取和保存图片 {video_path}。按 'q' 键退出显示。")

    notified = False  # 只通知一次
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(
            frame,
            tracker=tracker,
            persist=True,
            imgsz=640,
            device=device,
            verbose=False
        )
        annotated_frame = draw_boxes_noid(results[0], frame, class_names)

        # 在左上角显示当前帧数


        # 截取并保存指定帧
        if frame_count == fixed_frame and not notified:
            save_path_raw = video_path.replace('.mp4', f'_fixed_frame_{fixed_frame}_raw.jpg')
            save_path_anno = video_path.replace('.mp4', f'_fixed_frame_{fixed_frame}_det.jpg')
            cv2.imwrite(save_path_raw, frame)
            cv2.imwrite(save_path_anno, annotated_frame)
            print(f"已截取并保存第{fixed_frame}帧（原图）到：{save_path_raw}")
            print(f"已截取并保存第{fixed_frame}帧（含检测结果）到：{save_path_anno}")
            notified = True

        cv2.imshow(f"Processed: {video_path}", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("检测到退出...")
            break

    cap.release()
    cv2.destroyAllWindows()
    if not notified:
        print(f"视频帧数不足{fixed_frame}，未能保存。")

if __name__ == "__main__":
    video_paths = [
        r"C:\Desktop\graduate_design\label\0001.mp4"
    ]
    # 选择你的模型权重
    #model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\exp_2548\weights\best.pt")  # Baseline
    #model = YOLO( r"C:\Desktop\graduate_design\ultralytics-main\runs_ghost_new\stage3_full\weights\best.pt")  # Ghost-Backbone

    #model = YOLO( r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\weights_25413\fintune\weights\best.pt")
    model = YOLO(r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\exp_2548\weights\fintune_pruneandghost\weights\best.pt")# Pruned-Ghost

    fixed_frame = 9325
    for video in video_paths:
        detect_and_save_fixed_frame(model, video, fixed_frame=fixed_frame)