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

def detect_and_save_nth_lowest_object_conf_frame(model, video_path, n=1, conf_threshold=0.2, device="cuda", tracker="bytetrack.yaml"):
    """
    截取检测置信度大于阈值conf_threshold的第 n 小目标所在帧（全视频所有目标排序）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return 0

    frame_count = 0
    obj_conf_list = []  # 存储每个目标的置信度及相关帧信息

    class_names = model.names if hasattr(model, "names") else None

    ret, frame = cap.read()
    if not ret:
        print("视频为空或读取失败")
        return 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到第一帧
    print(f"开始检测并保存置信度大于{conf_threshold}的第{n}小目标所在帧 {video_path}。按 'q' 键退出。")

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
            verbose=False,
            conf=0.01  # 降低阈值，保留低置信度目标
        )

        if hasattr(results[0], "boxes") and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                # 只保留置信度大于阈值的目标
                if confs[i] > conf_threshold:
                    obj_conf_list.append({
                        "conf": confs[i],
                        "frame": frame.copy(),
                        "annotated": draw_boxes_noid(results[0], frame, class_names),
                        "frame_idx": frame_count + 1,
                        "box": boxes[i],
                        "cls": clss[i]
                    })
            # 可打印当前帧所有目标置信度
            print(f"Frame {frame_count + 1}: all confs = {confs}")

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

    if len(obj_conf_list) < n or n <= 0:
        print(f"有效检测目标（置信度>{conf_threshold}）不足{n}个，未能保存。")
        return frame_count

    # 根据置信度升序排列，选第n小（即第n低的置信度）
    obj_conf_list_sorted = sorted(obj_conf_list, key=lambda x: x["conf"])
    target_info = obj_conf_list_sorted[n-1]

    min_conf = target_info["conf"]
    min_conf_frame = target_info["frame"]
    min_conf_annotated = target_info["annotated"]
    min_conf_frame_idx = target_info["frame_idx"]
    min_conf_box = target_info["box"]
    min_conf_cls = target_info["cls"]

    print(f"大于阈值{conf_threshold}的第{n}小置信度: {min_conf:.4f}，对应帧号: {min_conf_frame_idx}，类别: {min_conf_cls}，box: {min_conf_box}")

    # 保存第n小置信度目标所在帧
    save_path_raw = video_path.replace('.mp4', f'_nthminobjconf_{n}_thres{conf_threshold}_{min_conf:.3f}_raw.jpg')
    save_path_anno = video_path.replace('.mp4', f'_nthminobjconf_{n}_thres{conf_threshold}_{min_conf:.3f}_det.jpg')
    cv2.imwrite(save_path_raw, min_conf_frame)
    cv2.imwrite(save_path_anno, min_conf_annotated)
    print(f"已保存大于阈值{conf_threshold}的第{n}小置信度目标所在帧（原图）到：{save_path_raw}")
    print(f"已保存大于阈值{conf_threshold}的第{n}小置信度目标所在帧（含检测结果）到：{save_path_anno}")

    return frame_count

if __name__ == "__main__":
    video_paths = [
        r"C:\Desktop\graduate_design\label\output_noisy_videos\gaussian_35\0001_gaussian_35.mp4"
    ]
    model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\runs_ghost_new\stage3_full\weights\best.pt")  # Ghost-Backbone
    n = 5 # 取置信度大于阈值的第n小的目标所在帧
    conf_threshold = 0.2 # 只考虑置信度大于0.2的目标
    for video in video_paths:
        detect_and_save_nth_lowest_object_conf_frame(model, video, n=n, conf_threshold=conf_threshold)