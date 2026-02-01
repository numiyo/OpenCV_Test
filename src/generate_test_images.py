"""
测试图像生成模块
功能：生成各类测试图像，用于验证算法效果
包括：基础测试图、颜色测试图、几何图形测试图、数字测试图、装甲板测试图
"""

import cv2
import numpy as np
import os


def create_directory(path):
    """创建目录"""
    os.makedirs(path, exist_ok=True)


def generate_basic_test_image(output_path):
    """
    生成基础预处理测试图像
    包含不同亮度区域和噪声，用于测试预处理算法
    """
    # 创建基础图像
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # 添加渐变背景
    for y in range(400):
        for x in range(600):
            image[y, x] = [
                int(50 + (x / 600) * 100),   # B
                int(100 + (y / 400) * 80),   # G
                int(150 - (x / 600) * 50)    # R
            ]
    
    # 添加几何形状
    # 圆形
    cv2.circle(image, (150, 100), 50, (255, 255, 255), -1)
    cv2.circle(image, (450, 100), 50, (0, 0, 0), -1)
    
    # 矩形
    cv2.rectangle(image, (100, 250), (200, 350), (255, 0, 0), -1)
    cv2.rectangle(image, (250, 250), (350, 350), (0, 255, 0), -1)
    cv2.rectangle(image, (400, 250), (500, 350), (0, 0, 255), -1)
    
    # 添加高斯噪声区域
    noise_region = np.random.normal(128, 30, (100, 100, 3)).astype(np.uint8)
    image[50:150, 250:350] = noise_region
    
    cv2.imwrite(output_path, image)
    print(f"[生成] 基础测试图像: {output_path}")
    return image


def generate_color_test_image(output_path):
    """
    生成颜色识别测试图像
    包含红、蓝、绿色块，用于测试颜色阈值分割
    """
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # 灰色背景
    image[:] = (128, 128, 128)
    
    # 红色块（多个不同大小和位置）
    cv2.circle(image, (100, 100), 40, (0, 0, 255), -1)
    cv2.rectangle(image, (50, 200), (150, 300), (0, 0, 200), -1)
    cv2.ellipse(image, (100, 350), (30, 20), 0, 0, 360, (50, 0, 255), -1)
    
    # 蓝色块
    cv2.circle(image, (300, 100), 45, (255, 0, 0), -1)
    cv2.rectangle(image, (250, 200), (350, 280), (200, 0, 0), -1)
    cv2.ellipse(image, (300, 350), (25, 35), 45, 0, 360, (255, 50, 0), -1)
    
    # 绿色块
    cv2.circle(image, (500, 100), 35, (0, 255, 0), -1)
    cv2.rectangle(image, (450, 200), (550, 320), (0, 200, 0), -1)
    
    # 添加一些干扰色（黄色、紫色）
    cv2.circle(image, (200, 350), 20, (0, 255, 255), -1)  # 黄色
    cv2.circle(image, (400, 350), 20, (255, 0, 255), -1)  # 紫色
    
    cv2.imwrite(output_path, image)
    print(f"[生成] 颜色测试图像: {output_path}")
    return image


def generate_shape_number_test_image(output_path):
    """
    生成几何图形和数字识别测试图像
    包含各种几何形状和数字0-9
    """
    image = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # 白色背景
    image[:] = (255, 255, 255)
    
    # 几何形状区域
    # 三角形
    triangle = np.array([[100, 150], [50, 230], [150, 230]], np.int32)
    cv2.fillPoly(image, [triangle], (0, 0, 255))
    
    # 正方形
    cv2.rectangle(image, (200, 150), (280, 230), (0, 255, 0), -1)
    
    # 矩形
    cv2.rectangle(image, (350, 160), (480, 220), (255, 0, 0), -1)
    
    # 圆形
    cv2.circle(image, (580, 190), 40, (255, 0, 255), -1)
    
    # 五边形
    pentagon = np.array([
        [100, 350],
        [70, 400],
        [90, 450],
        [130, 450],
        [150, 400]
    ], np.int32)
    cv2.fillPoly(image, [pentagon], (255, 255, 0))
    
    # 六边形
    hexagon = np.array([
        [250, 340], [290, 340], [310, 380],
        [290, 420], [250, 420], [230, 380]
    ], np.int32)
    cv2.fillPoly(image, [hexagon], (0, 255, 255))
    
    # 椭圆
    cv2.ellipse(image, (420, 380), (50, 30), 30, 0, 360, (128, 0, 128), -1)
    
    # 数字区域 (0-9)
    digit_y = 480
    for i, digit in enumerate("0123456789"):
        x = 50 + i * 65
        cv2.putText(image, digit, (x, digit_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    
    cv2.imwrite(output_path, image)
    print(f"[生成] 几何图形和数字测试图像: {output_path}")
    return image


def generate_armor_test_image(output_path):
    """
    生成装甲板检测测试图像
    模拟机器人装甲板的发光灯条
    """
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 深色背景
    image[:] = (30, 30, 30)
    
    # 绘制模拟装甲板
    def draw_armor_panel(img, center, angle, color, scale=1.0):
        """绘制单个装甲板"""
        cx, cy = center
        
        # 装甲板尺寸参数
        bar_width = int(15 * scale)
        bar_height = int(60 * scale)
        bar_spacing = int(50 * scale)
        
        # 计算旋转后的灯条位置
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 左灯条中心
        lx = cx - bar_spacing / 2 * cos_a
        ly = cy - bar_spacing / 2 * sin_a
        
        # 右灯条中心
        rx = cx + bar_spacing / 2 * cos_a
        ry = cy + bar_spacing / 2 * sin_a
        
        # 绘制灯条（发光效果）
        for offset in range(3, 0, -1):
            glow_intensity = int(100 / offset)
            glow_color = tuple(min(255, c + glow_intensity) for c in color)
            
            # 左灯条发光
            pts_l = np.array([
                [lx - bar_width/2 - offset, ly - bar_height/2 - offset],
                [lx + bar_width/2 + offset, ly - bar_height/2 - offset],
                [lx + bar_width/2 + offset, ly + bar_height/2 + offset],
                [lx - bar_width/2 - offset, ly + bar_height/2 + offset]
            ], np.int32)
            
            # 旋转灯条
            center_l = np.array([lx, ly])
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            pts_l = np.array([center_l + rotation_matrix @ (p - center_l) for p in pts_l], np.int32)
            cv2.fillPoly(img, [pts_l], glow_color)
            
            # 右灯条发光
            pts_r = np.array([
                [rx - bar_width/2 - offset, ry - bar_height/2 - offset],
                [rx + bar_width/2 + offset, ry - bar_height/2 - offset],
                [rx + bar_width/2 + offset, ry + bar_height/2 + offset],
                [rx - bar_width/2 - offset, ry + bar_height/2 + offset]
            ], np.int32)
            
            center_r = np.array([rx, ry])
            pts_r = np.array([center_r + rotation_matrix @ (p - center_r) for p in pts_r], np.int32)
            cv2.fillPoly(img, [pts_r], glow_color)
        
        # 绘制灯条核心
        # 左灯条
        pts_l = np.array([
            [lx - bar_width/2, ly - bar_height/2],
            [lx + bar_width/2, ly - bar_height/2],
            [lx + bar_width/2, ly + bar_height/2],
            [lx - bar_width/2, ly + bar_height/2]
        ], np.int32)
        pts_l = np.array([center_l + rotation_matrix @ (p - center_l) for p in pts_l], np.int32)
        cv2.fillPoly(img, [pts_l], color)
        
        # 右灯条
        pts_r = np.array([
            [rx - bar_width/2, ry - bar_height/2],
            [rx + bar_width/2, ry - bar_height/2],
            [rx + bar_width/2, ry + bar_height/2],
            [rx - bar_width/2, ry + bar_height/2]
        ], np.int32)
        pts_r = np.array([center_r + rotation_matrix @ (p - center_r) for p in pts_r], np.int32)
        cv2.fillPoly(img, [pts_r], color)
        
        # 绘制装甲板中心标记
        cv2.circle(img, (int(cx), int(cy)), 3, (0, 255, 0), -1)
    
    # 绘制多个装甲板
    # 蓝色装甲板（不同角度和位置）
    draw_armor_panel(image, (150, 120), 0, (255, 100, 0), 1.0)
    draw_armor_panel(image, (450, 120), 10, (255, 100, 0), 0.9)
    draw_armor_panel(image, (320, 240), -5, (255, 100, 0), 1.1)
    
    # 红色装甲板
    draw_armor_panel(image, (150, 360), 5, (0, 50, 255), 0.95)
    draw_armor_panel(image, (480, 360), -8, (0, 50, 255), 1.0)
    
    # 添加一些干扰元素
    # 随机小光点
    for _ in range(10):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        cv2.circle(image, (x, y), 3, (100, 100, 100), -1)
    
    cv2.imwrite(output_path, image)
    print(f"[生成] 装甲板测试图像: {output_path}")
    return image


def generate_lighting_test_images(output_dir):
    """
    生成不同光照条件的测试图像（用于鲁棒性测试）
    """
    create_directory(output_dir)
    
    # 基础图像
    base_image = np.zeros((300, 400, 3), dtype=np.uint8)
    base_image[:] = (128, 128, 128)
    
    # 添加测试目标
    cv2.circle(base_image, (200, 150), 50, (0, 0, 255), -1)
    cv2.rectangle(base_image, (100, 100), (150, 200), (255, 0, 0), -1)
    
    # 正常光照
    cv2.imwrite(os.path.join(output_dir, "lighting_normal.jpg"), base_image)
    
    # 强光
    bright = cv2.convertScaleAbs(base_image, alpha=1.5, beta=50)
    cv2.imwrite(os.path.join(output_dir, "lighting_bright.jpg"), bright)
    
    # 弱光
    dark = cv2.convertScaleAbs(base_image, alpha=0.5, beta=-30)
    cv2.imwrite(os.path.join(output_dir, "lighting_dark.jpg"), dark)
    
    # 不均匀光照
    gradient = base_image.copy().astype(float)
    for y in range(300):
        for x in range(400):
            factor = 0.5 + (x / 400) * 0.8
            gradient[y, x] = np.clip(base_image[y, x] * factor, 0, 255)
    gradient = gradient.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "lighting_gradient.jpg"), gradient)
    
    # 逆光（背景亮前景暗）
    backlit = base_image.copy()
    backlit[:] = (200, 200, 200)
    cv2.circle(backlit, (200, 150), 50, (50, 50, 100), -1)
    cv2.imwrite(os.path.join(output_dir, "lighting_backlit.jpg"), backlit)
    
    print(f"[生成] 光照测试图像组: {output_dir}")


def generate_occlusion_test_images(output_dir):
    """
    生成遮挡测试图像（用于鲁棒性测试）
    """
    create_directory(output_dir)
    
    # 基础图像
    base_image = np.zeros((300, 400, 3), dtype=np.uint8)
    base_image[:] = (128, 128, 128)
    
    # 完整目标
    cv2.circle(base_image, (200, 150), 50, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "occlusion_none.jpg"), base_image)
    
    # 部分遮挡（25%）
    occ_25 = base_image.copy()
    cv2.rectangle(occ_25, (200, 100), (250, 200), (128, 128, 128), -1)
    cv2.imwrite(os.path.join(output_dir, "occlusion_25.jpg"), occ_25)
    
    # 部分遮挡（50%）
    occ_50 = base_image.copy()
    cv2.rectangle(occ_50, (200, 100), (250, 200), (128, 128, 128), -1)
    cv2.rectangle(occ_50, (150, 150), (200, 200), (128, 128, 128), -1)
    cv2.imwrite(os.path.join(output_dir, "occlusion_50.jpg"), occ_50)
    
    # 部分遮挡（75%）
    occ_75 = base_image.copy()
    cv2.rectangle(occ_75, (150, 100), (250, 200), (128, 128, 128), -1)
    cv2.circle(occ_75, (175, 125), 10, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "occlusion_75.jpg"), occ_75)
    
    print(f"[生成] 遮挡测试图像组: {output_dir}")


def generate_all_test_images(base_dir="images"):
    """
    生成所有测试图像
    
    参数:
        base_dir: 图像保存基础目录
    """
    print("\n" + "="*50)
    print("【生成测试图像】")
    print("="*50 + "\n")
    
    create_directory(base_dir)
    
    # 基础必做任务测试图
    generate_basic_test_image(os.path.join(base_dir, "basic_test.jpg"))
    generate_color_test_image(os.path.join(base_dir, "color_test.jpg"))
    generate_shape_number_test_image(os.path.join(base_dir, "shape_number_test.jpg"))
    
    # 进阶任务测试图
    generate_armor_test_image(os.path.join(base_dir, "armor_test.jpg"))
    
    # 鲁棒性测试图
    generate_lighting_test_images(os.path.join(base_dir, "lighting"))
    generate_occlusion_test_images(os.path.join(base_dir, "occlusion"))
    
    print("\n" + "="*50)
    print("【测试图像生成完成】")
    print(f"图像保存位置: {os.path.abspath(base_dir)}")
    print("="*50 + "\n")


if __name__ == "__main__":
    generate_all_test_images()
