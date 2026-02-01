"""
图像基础预处理模块
功能：实现图片读取、灰度转换、高斯模糊去噪、直方图均衡化
输出处理前后对比图
"""

import cv2
import numpy as np
import os


def read_image(image_path):
    """
    读取图像文件
    
    参数:
        image_path: 图像文件路径
    返回:
        image: 读取的BGR格式图像，失败返回None
    """
    # 使用cv2.imread读取图像，默认返回BGR格式
    image = cv2.imread(image_path)
    if image is None:
        print(f"[错误] 无法读取图像: {image_path}")
        return None
    print(f"[成功] 图像读取成功，尺寸: {image.shape}")
    return image


def convert_to_grayscale(image):
    """
    将BGR图像转换为灰度图像
    
    参数:
        image: BGR格式彩色图像
    返回:
        gray: 灰度图像
    """
    # cv2.COLOR_BGR2GRAY: BGR转灰度的颜色空间转换码
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("[处理] 灰度转换完成")
    return gray


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    应用高斯模糊进行去噪处理
    
    参数:
        image: 输入图像（灰度或彩色）
        kernel_size: 高斯核大小，必须是奇数，默认(5,5)
        sigma: 标准差，0表示自动计算
    返回:
        blurred: 模糊后的图像
    """
    # 确保核大小为奇数
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        kernel_size = (kernel_size[0] + 1, kernel_size[1] + 1)
    
    # cv2.GaussianBlur: 高斯模糊，有效去除高斯噪声
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    print(f"[处理] 高斯模糊完成，核大小: {kernel_size}")
    return blurred


def apply_histogram_equalization(image):
    """
    应用直方图均衡化增强对比度
    
    参数:
        image: 输入图像（灰度或彩色）
    返回:
        equalized: 均衡化后的图像
    """
    # 判断输入是灰度图还是彩色图
    if len(image.shape) == 2:
        # 灰度图像直接均衡化
        equalized = cv2.equalizeHist(image)
    else:
        # 彩色图像：转换到YUV空间，对Y通道均衡化，保持色彩
        # YUV: Y-亮度, U/V-色度
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    print("[处理] 直方图均衡化完成")
    return equalized


def create_comparison_image(original, processed_list, titles):
    """
    创建处理前后对比图
    
    参数:
        original: 原始图像
        processed_list: 处理后的图像列表
        titles: 每个图像的标题
    返回:
        comparison: 拼接后的对比图
    """
    # 统一转换为BGR格式以便拼接
    display_images = [original]
    
    for img in processed_list:
        if len(img.shape) == 2:
            # 灰度图转BGR
            img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_display = img
        display_images.append(img_display)
    
    # 统一尺寸
    height, width = original.shape[:2]
    resized_images = []
    for img in display_images:
        resized = cv2.resize(img, (width, height))
        resized_images.append(resized)
    
    # 水平拼接
    comparison = np.hstack(resized_images)
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 计算每个标题的位置
    section_width = width
    for i, title in enumerate(titles):
        x = int(i * section_width + 10)
        y = 30
        # 添加白色背景使文字更清晰
        cv2.putText(comparison, title, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(comparison, title, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return comparison


def save_image(image, output_path):
    """
    保存图像到指定路径
    
    参数:
        image: 要保存的图像
        output_path: 保存路径
    返回:
        bool: 保存是否成功
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result = cv2.imwrite(output_path, image)
    if result:
        print(f"[成功] 图像已保存: {output_path}")
    else:
        print(f"[错误] 图像保存失败: {output_path}")
    return result


def run_preprocessing_pipeline(image_path, output_dir="output"):
    """
    运行完整的预处理流程
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    返回:
        dict: 包含所有处理结果的字典
    """
    print("\n" + "="*50)
    print("【图像基础预处理流程】")
    print("="*50)
    
    # 1. 读取图像
    original = read_image(image_path)
    if original is None:
        return None
    
    # 2. 灰度转换
    gray = convert_to_grayscale(original)
    
    # 3. 高斯模糊去噪
    blurred = apply_gaussian_blur(gray, kernel_size=(5, 5))
    
    # 4. 直方图均衡化
    equalized = apply_histogram_equalization(gray)
    
    # 5. 创建对比图
    comparison = create_comparison_image(
        original,
        [gray, blurred, equalized],
        ["Original", "Grayscale", "Gaussian Blur", "Equalized"]
    )
    
    # 6. 保存结果
    results = {
        "original": original,
        "grayscale": gray,
        "blurred": blurred,
        "equalized": equalized,
        "comparison": comparison
    }
    
    # 保存各阶段结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_image(gray, f"{output_dir}/{base_name}_01_grayscale.jpg")
    save_image(blurred, f"{output_dir}/{base_name}_02_blurred.jpg")
    save_image(equalized, f"{output_dir}/{base_name}_03_equalized.jpg")
    save_image(comparison, f"{output_dir}/{base_name}_04_comparison.jpg")
    
    print("="*50)
    print("【预处理流程完成】")
    print("="*50 + "\n")
    
    return results


if __name__ == "__main__":
    # 测试代码
    import sys
    
    # 如果提供了命令行参数，使用指定图像
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 使用默认测试图像
        image_path = "images/test_image.jpg"
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"[提示] 图像不存在: {image_path}")
        print("[提示] 请先运行 generate_test_images.py 生成测试图像")
    else:
        run_preprocessing_pipeline(image_path)
