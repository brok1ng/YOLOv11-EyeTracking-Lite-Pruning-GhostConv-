import os
import cv2
import numpy as np

def add_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def adjust_brightness(img, factor=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[..., 2] = hsv[..., 2] * factor
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(img, factor=1.0):
    img = img.astype(np.float32)
    mean = np.mean(img)
    img = (img - mean) * factor + mean
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_pepper_noise(image, proportion):
    output = image.copy()
    num_noise = int(proportion * image.shape[0] * image.shape[1])
    H, W = image.shape[:2]
    Y = np.random.randint(0, H, num_noise)
    X = np.random.randint(0, W, num_noise)
    values = np.random.choice([0, 255], num_noise)
    if image.ndim == 3:
        output[Y, X] = np.stack([values, values, values], axis=1)
    else:
        output[Y, X] = values
    return output

def process_video_single_noise_to_folders(src_video_path, dst_folder,
                                          noise_type='blur',
                                          strengths=None):
    """
    对单个视频添加噪声并输出到不同子文件夹中
    strengths: 列表，每个元素表示一种强度（如ksize, factor, sigma, amount等）
    noise_type: 'blur', 'brightness', 'contrast', 'gaussian', 'sp'
    """
    if strengths is None:
        print("请设置噪声强度列表 strengths")
        return

    # 创建输出子文件夹
    for s in strengths:
        subfolder = os.path.join(dst_folder, f"{noise_type}_{s}")
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    # 读取视频
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {src_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    basename = os.path.splitext(os.path.basename(src_video_path))[0]

    # 为每种强度新建一个VideoWriter
    writers = {}
    for s in strengths:
        out_video_path = os.path.join(dst_folder, f"{noise_type}_{s}", f"{basename}_{noise_type}_{s}.mp4")
        writers[s] = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for s in strengths:
            if noise_type == 'blur':
                noisy_frame = add_blur(frame, ksize=int(s))
            elif noise_type == 'brightness':
                noisy_frame = adjust_brightness(frame, factor=float(s))
            elif noise_type == 'contrast':
                noisy_frame = adjust_contrast(frame, factor=float(s))
            elif noise_type == 'gaussian':
                noisy_frame = add_gaussian_noise(frame, sigma=float(s))
            elif noise_type in ['sp', 'salt_pepper']:
                noisy_frame = add_salt_pepper_noise(frame, proportion=float(s))
            else:
                print(f"未知噪声类型: {noise_type}")
                continue
            writers[s].write(noisy_frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    for s in strengths:
        writers[s].release()
    print("视频处理完成。")

if __name__ == "__main__":
    # 设置路径
    source_video = r"C:\Desktop\graduate_design\label\0001.mp4"
    output_folder = r"C:\Desktop\graduate_design\label\output_noisy_videos"

    # 选择噪声类型和强度
    selected_noise = 'sp'  # 可选: 'blur', 'brightness', 'contrast', 'gaussian', 'sp'
    if selected_noise == 'blur':
        strengths = [83]
    elif selected_noise == 'brightness':
        strengths = [0.7, 1.0, 1.3]
    elif selected_noise == 'contrast':
        strengths = [0.5, 1.0, 2.0]
    elif selected_noise == 'gaussian':
        strengths = [35]
    elif selected_noise in ['sp', 'salt_pepper']:
        strengths = [0.1]
    else:
        strengths = [1]

    process_video_single_noise_to_folders(
        source_video,
        output_folder,
        noise_type=selected_noise,
        strengths=strengths
    )