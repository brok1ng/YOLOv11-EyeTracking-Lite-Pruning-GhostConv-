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
    '''
    给彩色或灰度图片添加黑白椒盐噪声（所有通道一致）
    '''
    output = image.copy()
    num_noise = int(proportion * image.shape[0] * image.shape[1])
    H, W = image.shape[:2]
    # 随机生成噪声点坐标
    Y = np.random.randint(0, H, num_noise)
    X = np.random.randint(0, W, num_noise)
    # 随机生成0或255
    values = np.random.choice([0, 255], num_noise)
    # 彩色图片
    if image.ndim == 3:
        output[Y, X] = np.stack([values, values, values], axis=1)
        # 或者更通用一点：
        # output[Y, X] = np.tile(values[:, None], (1, image.shape[2]))
    else:
        output[Y, X] = values
    return output


def process_images_single_noise_to_folders(src_folder, dst_folder,
                                           noise_type='blur',
                                           strengths=None):
    """
    strengths: 列表，每个元素表示一种强度（如ksize, factor, sigma, amount等）
    noise_type: 'blur', 'brightness', 'contrast', 'gaussian', 'sp'
    """
    if strengths is None:
        print("请设置噪声强度列表 strengths")
        return

    for s in strengths:
        # 为每个强度新建一个子文件夹
        subfolder = os.path.join(dst_folder, f"{noise_type}_{s}")
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    for fname in os.listdir(src_folder):
        if fname.lower().endswith('.png'):
            img_path = os.path.join(src_folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {img_path}, skipping.")
                continue

            for s in strengths:
                if noise_type == 'blur':
                    noisy_img = add_blur(img, ksize=int(s))
                elif noise_type == 'brightness':
                    noisy_img = adjust_brightness(img, factor=float(s))
                elif noise_type == 'contrast':
                    noisy_img = adjust_contrast(img, factor=float(s))
                elif noise_type == 'gaussian':
                    noisy_img = add_gaussian_noise(img, sigma=float(s))
                elif noise_type in ['sp', 'salt_pepper']:
                    noisy_img = add_salt_pepper_noise(img, proportion=float(s))
                else:
                    print(f"未知噪声类型: {noise_type}")
                    continue

                subfolder = os.path.join(dst_folder, f"{noise_type}_{s}")
                out_name = fname
                cv2.imwrite(os.path.join(subfolder, out_name), noisy_img)
                print(f"Processed: {fname}, noise: {noise_type}, strength: {s} -> {subfolder}/{out_name}")


if __name__ == "__main__":
    # 设置路径
    source_folder = r"C:\Desktop\graduate_design\label\0001_copy\420"
    output_folder = r"C:\Desktop\graduate_design\label\0001_copy\420\output_noisy_pngs"

    # 选择噪声类型和强度
    selected_noise = 'gaussian'  # 可选: 'blur', 'brightness', 'contrast', 'gaussian', 'sp'
    if selected_noise == 'blur':
        strengths = [3,23,43,63]  # 卷积核大小，必须是奇数
    elif selected_noise == 'brightness':
        strengths = [0.7, 1.0, 1.3]  # 亮度因子
    elif selected_noise == 'contrast':
        strengths = [0.5, 1.0, 2.0]  # 对比度因子
    elif selected_noise == 'gaussian':
        strengths = [35]  # 高斯噪声sigma
    elif selected_noise in ['sp', 'salt_pepper']:
        strengths = [ 0.05,0.06,0.07,0.08,0.09,0.1]  # 椒盐噪声比例
    else:
        strengths = [1]

    process_images_single_noise_to_folders(
        source_folder,
        output_folder,
        noise_type=selected_noise,
        strengths=strengths
    )