import time
import cv2
from ultralytics import YOLO

def measure_video_inference_time_and_display(model, video_path, device="cuda", tracker="bytetrack.yaml"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return 0, 0.0

    frame_count = 0
    total_time = 0.0
    warmup = 5

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("视频为空或读取失败")
        return 0, 0.0

    # 模型预热
    print(f"正在对 {video_path} 进行模型预热...")
    for _ in range(warmup):
        _ = model.track(
            frame,
            tracker=tracker,
            persist=True,
            imgsz=640,
            device=device,
            verbose=False
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到第一帧
    print(f"开始统计并显示 {video_path} 的实时性... 按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        results = model.track(
            frame,
            tracker=tracker,
            persist=True,
            imgsz=640,
            device=device,
            verbose=False
        )
        infer_time = time.time() - start
        total_time += infer_time
        frame_count += 1

        # 可视化
        annotated_frame = results[0].plot() if hasattr(results[0], 'plot') else frame
        cv2.imshow(f"Processed: {video_path}", annotated_frame)
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("检测到退出...")
            break

        if frame_count % 50 == 0:
            print(f"{video_path} 已处理 {frame_count} 帧...")

    cap.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print(f"{video_path} 未能处理任何帧。")
        return 0, 0.0

    avg_time = total_time / frame_count
    print(f"\n===== {video_path} 实时性分析结果 =====")
    print(f"总帧数: {frame_count}")
    print(f"平均单帧推理耗时: {avg_time * 1000:.2f} ms")
    print(f"平均FPS: {1 / avg_time:.2f}")
    return frame_count, total_time

if __name__ == "__main__":
    video_paths = [

        r"C:\Desktop\graduate_design\label\leftlevel4.mp4"
    ]
    # 选择你的模型权重

    #model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\exp_2548\weights\best.pt")  # Baseline
    #model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\runs_ghost_new\stage3_full\weights\best.pt")  # Ghost-Backbone
    model = YOLO(r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\weights_25413\fintune\weights\best.pt")
    #model = YOLO(r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\exp_2548\weights\fintune_pruneandghost\weights\best.pt")# Pruned-Ghost
    all_frames = 0
    all_time = 0.0
    for video in video_paths:
        f, t = measure_video_inference_time_and_display(model, video)
        all_frames += f
        all_time += t

    if all_frames > 0:
        print("\n===== 所有视频总体实时性统计 =====")
        print(f"总帧数: {all_frames}")
        print(f"平均单帧推理耗时: {all_time / all_frames * 1000:.2f} ms")
        print(f"平均FPS: {all_frames / all_time:.2f}")
    else:
        print("所有视频都未能成功推理。")