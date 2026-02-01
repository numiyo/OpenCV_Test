"""
颜色阈值色块识别模块
功能：基于HSV空间实现红/蓝色目标分割，结合形态学操作去噪
筛选有效轮廓并标注坐标、面积
"""

import cv2
import numpy as np
import os


# HSV颜色空间阈值定义
# H: 色调(0-179), S: 饱和度(0-255), V: 明度(0-255)
# 红色在HSV空间中跨越0度，需要两个范围
COLOR_RANGES = {
    "red": {
        "lower1": np.array([0, 100, 100]),    # 红色下限1
        "upper1": np.array([10, 255, 255]),   # 红色上限1
        "lower2": np.array([160, 100, 100]),  # 红色下限2（跨越0度）
        "upper2": np.array([179, 255, 255]),  # 红色上限2
        "color": (0, 0, 255)                   # BGR显示颜色
    },
    "blue": {
        "lower": np.array([100, 100, 100]),   # 蓝色下限
        "upper": np.array([130, 255, 255]),   # 蓝色上限
        "color": (255, 0, 0)                   # BGR显示颜色
    },
    "green": {
        "lower": np.array([40, 50, 50]),      # 绿色下限
        "upper": np.array([80, 255, 255]),    # 绿色上限
        "color": (0, 255, 0)                   # BGR显示颜色
    }
}


def convert_to_hsv(image):
    """
    将BGR图像转换为HSV颜色空间
    
    参数:
        image: BGR格式图像
    返回:
        hsv: HSV格式图像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv


def create_color_mask(hsv_image, color_name):
    """
    创建指定颜色的二值掩码
    
    参数:
        hsv_image: HSV格式图像
        color_name: 颜色名称 ("red", "blue", "green")
    返回:
        mask: 二值掩码图像
    """
    color_info = COLOR_RANGES.get(color_name)
    if color_info is None:
        print(f"[错误] 不支持的颜色: {color_name}")
        return None
    
    if color_name == "red":
        # 红色需要合并两个范围的掩码
        mask1 = cv2.inRange(hsv_image, color_info["lower1"], color_info["upper1"])
        mask2 = cv2.inRange(hsv_image, color_info["lower2"], color_info["upper2"])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv_image, color_info["lower"], color_info["upper"])
    
    return mask


def apply_morphology(mask, kernel_size=5, operations=["open", "close"]):
    """
    应用形态学操作去噪
    
    参数:
        mask: 输入二值掩码
        kernel_size: 结构元素大小
        operations: 操作列表，可选 "open", "close", "erode", "dilate"
    返回:
        processed_mask: 处理后的掩码
    """
    # 创建结构元素（核）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    result = mask.copy()
    
    for op in operations:
        if op == "open":
            # 开运算：先腐蚀后膨胀，去除小噪点
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            # 闭运算：先膨胀后腐蚀，填补小孔洞
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif op == "erode":
            # 腐蚀：缩小前景区域
            result = cv2.erode(result, kernel, iterations=1)
        elif op == "dilate":
            # 膨胀：扩大前景区域
            result = cv2.dilate(result, kernel, iterations=1)
    
    return result


def find_contours(mask, min_area=100, max_area=None):
    """
    查找并筛选有效轮廓
    
    参数:
        mask: 二值掩码图像
        min_area: 最小面积阈值
        max_area: 最大面积阈值（None表示无限制）
    返回:
        valid_contours: 符合条件的轮廓列表
        contour_info: 轮廓信息列表（包含中心点、面积等）
    """
    # 查找轮廓
    # cv2.RETR_EXTERNAL: 只检测外层轮廓
    # cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直、对角线段
    contours, hierarchy = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    valid_contours = []
    contour_info = []
    
    for i, contour in enumerate(contours):
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 面积筛选
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        
        # 计算轮廓中心点（矩）
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0
        
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算外接圆
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        
        valid_contours.append(contour)
        contour_info.append({
            "id": i + 1,
            "center": (cx, cy),
            "area": area,
            "bbox": (x, y, w, h),
            "circle_center": (int(circle_x), int(circle_y)),
            "circle_radius": int(radius)
        })
    
    # 按面积降序排序
    sorted_pairs = sorted(
        zip(valid_contours, contour_info), 
        key=lambda x: x[1]["area"], 
        reverse=True
    )
    
    if sorted_pairs:
        valid_contours, contour_info = zip(*sorted_pairs)
        valid_contours = list(valid_contours)
        contour_info = list(contour_info)
    
    return valid_contours, contour_info


def draw_contour_info(image, contours, contour_info, color=(0, 255, 0)):
    """
    在图像上绘制轮廓和信息标注
    
    参数:
        image: 原始图像
        contours: 轮廓列表
        contour_info: 轮廓信息列表
        color: 绘制颜色
    返回:
        annotated: 标注后的图像
    """
    annotated = image.copy()
    
    for i, (contour, info) in enumerate(zip(contours, contour_info)):
        # 绘制轮廓
        cv2.drawContours(annotated, [contour], -1, color, 2)
        
        # 绘制中心点
        cx, cy = info["center"]
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        
        # 绘制外接矩形
        x, y, w, h = info["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 添加文字信息
        text = f"ID:{info['id']} ({cx},{cy}) A:{int(info['area'])}"
        cv2.putText(
            annotated, 
            text, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
    
    return annotated


def detect_color_targets(image, target_colors=["red", "blue"], output_dir="output"):
    """
    检测指定颜色的目标
    
    参数:
        image: 输入BGR图像
        target_colors: 要检测的颜色列表
        output_dir: 输出目录
    返回:
        results: 检测结果字典
    """
    print("\n" + "="*50)
    print("【颜色阈值色块识别】")
    print("="*50)
    
    results = {
        "original": image,
        "detections": {}
    }
    
    # 转换为HSV
    hsv = convert_to_hsv(image)
    
    for color_name in target_colors:
        print(f"\n[检测] 正在检测 {color_name} 色块...")
        
        # 创建颜色掩码
        mask = create_color_mask(hsv, color_name)
        if mask is None:
            continue
        
        # 形态学操作去噪
        mask_clean = apply_morphology(mask, kernel_size=5, operations=["open", "close"])
        
        # 查找轮廓
        contours, info = find_contours(mask_clean, min_area=100)
        
        print(f"[结果] 发现 {len(contours)} 个 {color_name} 色块")
        
        # 绘制结果
        color = COLOR_RANGES[color_name]["color"]
        annotated = draw_contour_info(image, contours, info, color)
        
        # 保存结果
        results["detections"][color_name] = {
            "mask": mask,
            "mask_clean": mask_clean,
            "contours": contours,
            "info": info,
            "annotated": annotated
        }
        
        # 打印详细信息
        for item in info:
            print(f"  - 目标 {item['id']}: 中心{item['center']}, 面积{item['area']:.1f}")
    
    # 创建综合对比图
    comparison = create_color_comparison(image, results["detections"])
    results["comparison"] = comparison
    
    print("="*50)
    print("【颜色识别完成】")
    print("="*50 + "\n")
    
    return results


def create_color_comparison(original, detections):
    """
    创建颜色检测综合对比图
    
    参数:
        original: 原始图像
        detections: 检测结果字典
    返回:
        comparison: 对比图
    """
    rows = []
    
    # 第一行：原始图 + 各颜色掩码
    masks = [original]
    mask_titles = ["Original"]
    
    for color_name, data in detections.items():
        # 将掩码转为3通道以便显示
        mask_display = cv2.cvtColor(data["mask_clean"], cv2.COLOR_GRAY2BGR)
        masks.append(mask_display)
        mask_titles.append(f"{color_name.capitalize()} Mask")
    
    # 填充空白使每行数量一致
    while len(masks) < 4:
        masks.append(np.zeros_like(original))
        mask_titles.append("")
    
    # 统一尺寸并拼接
    height, width = original.shape[:2]
    mask_row = []
    for img in masks[:4]:
        mask_row.append(cv2.resize(img, (width, height)))
    rows.append(np.hstack(mask_row))
    
    # 第二行：检测结果标注
    annotated_images = [original]
    annotated_titles = ["Original"]
    
    for color_name, data in detections.items():
        annotated_images.append(data["annotated"])
        annotated_titles.append(f"{color_name.capitalize()} Detection")
    
    while len(annotated_images) < 4:
        annotated_images.append(np.zeros_like(original))
        annotated_titles.append("")
    
    annotated_row = []
    for img in annotated_images[:4]:
        annotated_row.append(cv2.resize(img, (width, height)))
    rows.append(np.hstack(annotated_row))
    
    # 垂直拼接两行
    comparison = np.vstack(rows)
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 第一行标题
    for i, title in enumerate(mask_titles[:4]):
        if title:
            x = i * width + 10
            y = 30
            cv2.putText(comparison, title, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(comparison, title, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    # 第二行标题
    row2_offset = height
    for i, title in enumerate(annotated_titles[:4]):
        if title:
            x = i * width + 10
            y = row2_offset + 30
            cv2.putText(comparison, title, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(comparison, title, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return comparison


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "images/color_test.jpg"
    
    if not os.path.exists(image_path):
        print(f"[提示] 图像不存在: {image_path}")
        print("[提示] 请先运行 generate_test_images.py 生成测试图像")
    else:
        image = cv2.imread(image_path)
        results = detect_color_targets(image)
        
        # 保存结果
        os.makedirs("output", exist_ok=True)
        for color_name, data in results["detections"].items():
            cv2.imwrite(f"output/color_{color_name}_mask.jpg", data["mask_clean"])
            cv2.imwrite(f"output/color_{color_name}_result.jpg", data["annotated"])
        cv2.imwrite("output/color_comparison.jpg", results["comparison"])
