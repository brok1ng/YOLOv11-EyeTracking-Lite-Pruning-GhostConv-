import time
import numpy as np
from ultralytics import YOLO
import cv2
import glob
import torch

def measure_inference_time(model, image_paths, device="cuda", warmup=10, repeat=1000):
    print(f"图片总数: {len(image_paths)}")
    imgs = [cv2.imread(p) for p in image_paths]

    # 预热
    print("正在进行模型预热...")
    for img in imgs[:warmup]:
        _ = model(img, device=device, verbose=False)

    times = []
    print("开始正式计时...")
    for i in range(repeat):
        img = imgs[i % len(imgs)]
        torch.cuda.synchronize()  # 确保开始前GPU空闲
        start = time.time()
        _ = model(img, device=device, verbose=False)
        torch.cuda.synchronize()  # 等待GPU推理结束
        end = time.time()
        times.append(end - start)
        print(f"第{i+1}次推理耗时: {times[-1]:.4f} 秒")

    times = np.array(times)
    print("\n===== 实时性分析结果 =====")
    print(f"平均单帧推理耗时: {np.mean(times) * 1000:.2f} ms")
    print(f"最小单帧推理耗时: {np.min(times) * 1000:.2f} ms")
    print(f"最大单帧推理耗时: {np.max(times) * 1000:.2f} ms")
    print(f"平均FPS: {1 / np.mean(times):.2f}")

if __name__ == "__main__":
    # 替换为你的模型路径
    # model = YOLO(r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\exp_2548\weights\fintune_pruneandghost\weights\best.pt")
    # model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\exp_2548\weights\best.pt")  # Baseline
    model = YOLO(r"C:\Desktop\graduate_design\ultralytics-main\runs_ghost_new\stage3_full\weights\best.pt")  # Ghost-Backbone
    # model = YOLO(r"C:\Desktop\graduate_design\labels_my-project-name_2025-01-14-09-49-56\yolov11_prune_distillation-main\weights_25413\fintune\weights\best.pt")  # Pruned-Ghost

    image_folder = r"C:\Desktop\graduate_design\label\0001_copy\420"
    image_paths = glob.glob(f"{image_folder}/*.png")
    if len(image_paths) == 0:
        print("未找到测试图片，请检查路径！")
        exit(1)
    measure_inference_time(model, image_paths)