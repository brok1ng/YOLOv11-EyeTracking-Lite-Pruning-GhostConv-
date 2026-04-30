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

def detect_and_save_min_conf_frame(model, video_path, device="cuda", tracker="bytetrack.yaml"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return 0

    frame_count = 0
    min_conf = float('inf')
    min_conf_frame = None
    min_conf_annotated = None
    min_conf_frame_idx = -1

    saved_frame_indices = []

    class_names = model.names if hasattr(model, "names") else None

    ret, frame = cap.read()
    if not ret:
        print("视频为空或读取失败")
        return 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到第一帧
    print(f"开始检测并保存置信度最小帧 {video_path}。按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            tracker=tracker,
            persist=True,
            imgsz=640,
            device=device,
            verbose=False
        )

        # 获取当前帧最小置信度（忽略空boxes帧）
        frame_min_conf = float('inf')
        if hasattr(results[0], "boxes") and results[0].boxes is not None and len(results[0].boxes) > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            frame_min_conf = float(np.min(confs))
            # 若当前帧的最小置信度低于历史最小，则保存
            if frame_min_conf < min_conf:
                min_conf = frame_min_conf
                min_conf_frame = frame.copy()
                min_conf_annotated = draw_boxes_noid(results[0], frame, class_names)
                min_conf_frame_idx = frame_count + 1  # 帧号（从1开始）
                saved_frame_indices = [min_conf_frame_idx]
            elif frame_min_conf == min_conf and min_conf < float('inf'):
                saved_frame_indices.append(frame_count + 1)

        # 在左上角显示当前帧数
        frame_count += 1
        annotated_frame = draw_boxes_noid(results[0], frame, class_names)
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(f"Processed: {video_path}", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("检测到退出...")
            break

        if frame_count % 50 == 0:
            print(f"{video_path} 已处理 {frame_count} 帧...")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n===== {video_path} 检测完成，总帧数: {frame_count} =====")
    print(f"最小置信度: {min_conf:.4f}" if min_conf != float('inf') else "无检测帧")
    print(f"最小置信度帧号（可能有多帧）: {saved_frame_indices}")
    print(f"所截取的帧数: {len(saved_frame_indices)}")

    # 保存置信度最小帧
    if min_conf_frame is not None and min_conf_annotated is not None:
        save_path_raw = video_path.replace('.mp4', f'_minconf_{min_conf:.3f}_raw.jpg')
        save_path_anno = video_path.replace('.mp4', f'_minconf_{min_conf:.3f}_det.jpg')
        cv2.imwrite(save_path_raw, min_conf_frame)
        cv2.imwrite(save_path_anno, min_conf_annotated)
        print(f"已保存最小置信度帧（原图）到：{save_path_raw}")
        print(f"已保存最小置信度帧（含检测结果）到：{save_path_anno}")
    else:
        print("未找到有效的检测帧，未保存图片。")
    return frame_count

if __name__ == "__main__":
    video_paths = [
          r"C:\Desktop\graduate_design\label\0001.mp4"
    ]
    # 选择你的模型权重
    # model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\exp_2548\weights\best.pt")  # Baseline
    #model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\runs_ghost_new\stage3_full\weights\best.pt")  # Ghost-Backbone
    #model = YOLO( r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\weights_25413\fintune\weights\best.pt")
    model = YOLO(r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\exp_2548\weights\fintune_pruneandghost\weights\best.pt")# Pruned-Ghost
    for video in video_paths:
        detect_and_save_min_conf_frame(model, video)