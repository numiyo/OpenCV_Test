"""
简单特征识别模块
功能：识别图片中矩形、圆形等几何图形
识别0-9印刷体数字（不使用OCR库，基于轮廓/模板匹配实现）
"""

import cv2
import numpy as np
import os


# 数字模板（用于模板匹配）
# 这里使用简单的7段数码管风格的模板定义
DIGIT_TEMPLATES = {}


def preprocess_for_recognition(image):
    """
    预处理图像用于识别
    
    参数:
        image: 输入图像
    返回:
        binary: 二值化图像
        edges: 边缘图像
    """
    # 灰度转换
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 反转，使目标为白色
        11, 2
    )
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    return binary, edges


def detect_geometric_shapes(image, binary=None):
    """
    检测几何图形（矩形、圆形、三角形等）
    
    参数:
        image: 原始图像
        binary: 二值图像（可选）
    返回:
        shapes: 检测到的图形列表
        annotated: 标注后的图像
    """
    print("\n[几何图形检测] 开始...")
    
    if binary is None:
        binary, _ = preprocess_for_recognition(image)
    
    # 形态学操作优化轮廓
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    shapes = []
    annotated = image.copy()
    
    for i, contour in enumerate(contours):
        # 面积过滤
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 多边形逼近
        epsilon = 0.04 * perimeter  # 逼近精度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 获取外接矩形信息
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # 形状分类
        shape_type = classify_shape(approx, area, perimeter, aspect_ratio)
        
        # 计算中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        shape_info = {
            "id": len(shapes) + 1,
            "type": shape_type,
            "center": (cx, cy),
            "area": area,
            "bbox": (x, y, w, h),
            "contour": contour,
            "approx": approx
        }
        shapes.append(shape_info)
        
        # 绘制标注
        color = get_shape_color(shape_type)
        cv2.drawContours(annotated, [approx], -1, color, 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        
        # 添加标签
        label = f"{shape_type}"
        cv2.putText(
            annotated, 
            label, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )
        
        print(f"  - 图形 {shape_info['id']}: {shape_type}, 中心({cx},{cy}), 面积{area:.1f}")
    
    print(f"[几何图形检测] 完成，共检测到 {len(shapes)} 个图形\n")
    return shapes, annotated


def classify_shape(approx, area, perimeter, aspect_ratio):
    """
    根据轮廓特征分类形状
    
    参数:
        approx: 多边形逼近结果
        area: 面积
        perimeter: 周长
        aspect_ratio: 宽高比
    返回:
        shape_type: 形状类型字符串
    """
    vertices = len(approx)
    
    # 圆形度计算：4π*面积/周长²，圆形接近1
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # 判断是矩形还是正方形
        if 0.9 <= aspect_ratio <= 1.1:
            return "Square"
        else:
            return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    elif vertices > 6:
        # 根据圆形度判断是否为圆形
        if circularity > 0.75:
            return "Circle"
        else:
            return "Ellipse" if aspect_ratio < 0.8 or aspect_ratio > 1.2 else "Polygon"
    else:
        return "Unknown"


def get_shape_color(shape_type):
    """
    获取形状对应的显示颜色
    
    参数:
        shape_type: 形状类型
    返回:
        color: BGR颜色元组
    """
    colors = {
        "Triangle": (0, 255, 255),    # 黄色
        "Rectangle": (0, 255, 0),     # 绿色
        "Square": (255, 255, 0),      # 青色
        "Pentagon": (255, 0, 255),    # 紫色
        "Hexagon": (255, 165, 0),     # 橙色
        "Circle": (0, 0, 255),        # 红色
        "Ellipse": (255, 0, 0),       # 蓝色
        "Polygon": (128, 128, 128),   # 灰色
        "Unknown": (255, 255, 255)    # 白色
    }
    return colors.get(shape_type, (255, 255, 255))


def create_digit_templates():
    """
    创建数字0-9的模板图像（用于模板匹配）
    
    返回:
        templates: 数字模板字典 {数字: 模板图像}
    """
    templates = {}
    template_w, template_h = 50, 70  # 模板尺寸 (宽, 高)
    
    # 使用简单字体创建数字模板
    for digit in range(10):
        # 创建空白图像 (高, 宽)
        template = np.zeros((template_h, template_w), dtype=np.uint8)
        
        # 绘制数字
        digit_str = str(digit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 调整字体大小和位置使数字居中
        (text_width, text_height), _ = cv2.getTextSize(digit_str, font, 2, 3)
        x = (template_w - text_width) // 2
        y = (template_h + text_height) // 2
        
        cv2.putText(template, digit_str, (x, y), font, 2, 255, 3)
        
        templates[digit] = template
    
    return templates


def recognize_digits(image, binary=None, templates=None):
    """
    识别图像中的印刷体数字（基于轮廓和模板匹配）
    
    参数:
        image: 原始图像
        binary: 二值图像（可选）
        templates: 数字模板字典（可选）
    返回:
        digits: 识别结果列表
        annotated: 标注后的图像
    """
    print("\n[数字识别] 开始...")
    
    # 创建模板
    if templates is None:
        templates = create_digit_templates()
    
    if binary is None:
        binary, _ = preprocess_for_recognition(image)
    
    # 查找数字轮廓
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    digits = []
    annotated = image.copy()
    
    # 按x坐标排序（从左到右）
    contour_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # 过滤噪声（根据宽高和面积）
        if w > 10 and h > 20 and area > 100:
            contour_boxes.append((x, y, w, h, contour))
    
    contour_boxes.sort(key=lambda b: b[0])  # 按x排序
    
    for x, y, w, h, contour in contour_boxes:
        # 提取ROI
        roi = binary[y:y+h, x:x+w]
        
        # 调整尺寸匹配模板 (确保尺寸一致)
        template_size = (50, 70)
        roi_resized = cv2.resize(roi, template_size)
        
        # 模板匹配
        best_match = -1
        best_score = float('inf')
        
        for digit, template in templates.items():
            # 确保模板尺寸正确
            if template.shape != template_size:
                template = cv2.resize(template, template_size)
            
            # 使用平方差匹配
            result = cv2.matchTemplate(roi_resized, template, cv2.TM_SQDIFF_NORMED)
            score = np.min(result)
            
            if score < best_score:
                best_score = score
                best_match = digit
        
        # 阈值判断（避免误识别）
        if best_score < 0.3:  # 匹配阈值
            digit_info = {
                "digit": best_match,
                "confidence": 1 - best_score,
                "bbox": (x, y, w, h)
            }
            digits.append(digit_info)
            
            # 绘制标注
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                annotated, 
                str(best_match), 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
            
            print(f"  - 识别数字: {best_match}, 置信度: {1-best_score:.2f}")
    
    print(f"[数字识别] 完成，共识别 {len(digits)} 个数字\n")
    return digits, annotated


def recognize_digits_contour_based(image, binary=None):
    """
    基于轮廓特征的备用数字识别方法
    使用轮廓特征（端点、孔洞数等）识别数字
    
    参数:
        image: 原始图像
        binary: 二值图像（可选）
    返回:
        digits: 识别结果列表
        annotated: 标注后的图像
    """
    print("\n[数字识别-轮廓法] 开始...")
    
    if binary is None:
        binary, _ = preprocess_for_recognition(image)
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        binary.copy(), 
        cv2.RETR_TREE,  # 使用TREE模式获取层级关系（用于检测孔洞）
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    digits = []
    annotated = image.copy()
    
    if hierarchy is None:
        return digits, annotated
    
    # 分析每个轮廓
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤噪声
        if w < 10 or h < 20 or w > 200 or h > 200:
            continue
        
        # 宽高比检查
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 3:
            continue
        
        # 获取层级信息
        h_item = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]
        
        # 特征提取
        features = extract_digit_features(contour, binary, x, y, w, h)
        
        # 基于特征识别
        digit = classify_digit_by_features(features)
        
        if digit is not None:
            digit_info = {
                "digit": digit,
                "features": features,
                "bbox": (x, y, w, h)
            }
            digits.append(digit_info)
            
            # 绘制标注
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                annotated, 
                str(digit), 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
            
            print(f"  - 识别数字: {digit}, 特征: {features}")
    
    print(f"[数字识别-轮廓法] 完成，共识别 {len(digits)} 个数字\n")
    return digits, annotated


def extract_digit_features(contour, binary, x, y, w, h):
    """
    提取数字的轮廓特征
    
    参数:
        contour: 数字轮廓
        binary: 二值图像
        x, y, w, h: 外接矩形
    返回:
        features: 特征字典
    """
    # 提取ROI
    roi = binary[y:y+h, x:x+w]
    
    # 1. 孔洞数（使用轮廓层级或连通域分析）
    roi_inv = cv2.bitwise_not(roi)
    hole_contours, _ = cv2.findContours(roi_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_count = len([c for c in hole_contours if cv2.contourArea(c) > 10])
    
    # 2. 端点数（使用骨架化后检测端点）
    # 简化：检测上下左右的投影
    top_proj = np.sum(roi[0:5, :]) > 0
    bottom_proj = np.sum(roi[-5:, :]) > 0
    left_proj = np.sum(roi[:, 0:5]) > 0
    right_proj = np.sum(roi[:, -5:]) > 0
    
    # 3. 重心位置
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid_x_ratio = (cx - x) / w if w > 0 else 0.5
        centroid_y_ratio = (cy - y) / h if h > 0 else 0.5
    else:
        centroid_x_ratio = 0.5
        centroid_y_ratio = 0.5
    
    features = {
        "hole_count": hole_count,
        "top_open": not top_proj,
        "bottom_open": not bottom_proj,
        "left_open": not left_proj,
        "right_open": not right_proj,
        "centroid_x": centroid_x_ratio,
        "centroid_y": centroid_y_ratio,
        "aspect_ratio": h / w if w > 0 else 1
    }
    
    return features


def classify_digit_by_features(features):
    """
    基于特征识别数字
    
    参数:
        features: 特征字典
    返回:
        digit: 识别的数字或None
    """
    hole_count = features["hole_count"]
    cx, cy = features["centroid_x"], features["centroid_y"]
    
    # 基于孔洞数初步分类
    if hole_count == 0:
        # 无孔洞：1, 2, 3, 4, 5, 7
        if features["aspect_ratio"] > 2:
            return 1
        elif cy < 0.4:
            return 7
        elif features["bottom_open"] and not features["top_open"]:
            return 2
        elif not features["bottom_open"] and features["top_open"]:
            return 5
        else:
            return 3 if cx > 0.5 else 4
    elif hole_count == 1:
        # 一个孔洞：0, 4, 6, 9
        if cy > 0.6:
            return 6
        elif cy < 0.4:
            return 9
        elif features["aspect_ratio"] > 1.5:
            return 0
        else:
            return 4
    elif hole_count == 2:
        # 两个孔洞：8
        return 8
    
    return None


def run_shape_recognition(image_path, output_dir="output"):
    """
    运行完整的形状和数字识别流程
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    print("\n" + "="*50)
    print("【简单特征识别】")
    print("="*50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"[错误] 无法读取图像: {image_path}")
        return None
    
    # 预处理
    binary, edges = preprocess_for_recognition(image)
    
    # 几何图形识别
    shapes, shape_annotated = detect_geometric_shapes(image, binary)
    
    # 数字识别（使用模板匹配）
    templates = create_digit_templates()
    digits, digit_annotated = recognize_digits(image, binary, templates)
    
    # 数字识别（备用轮廓法）
    digits_contour, digit_contour_annotated = recognize_digits_contour_based(image, binary)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    cv2.imwrite(f"{output_dir}/{base_name}_shapes.jpg", shape_annotated)
    cv2.imwrite(f"{output_dir}/{base_name}_digits.jpg", digit_annotated)
    cv2.imwrite(f"{output_dir}/{base_name}_digits_contour.jpg", digit_contour_annotated)
    
    # 创建对比图
    comparison = create_recognition_comparison(image, shape_annotated, digit_annotated)
    cv2.imwrite(f"{output_dir}/{base_name}_recognition_comparison.jpg", comparison)
    
    results = {
        "shapes": shapes,
        "digits": digits,
        "shape_annotated": shape_annotated,
        "digit_annotated": digit_annotated
    }
    
    print("="*50)
    print("【特征识别完成】")
    print("="*50 + "\n")
    
    return results


def create_recognition_comparison(original, shape_result, digit_result):
    """
    创建识别结果对比图
    
    参数:
        original: 原始图像
        shape_result: 图形识别结果
        digit_result: 数字识别结果
    返回:
        comparison: 对比图
    """
    height, width = original.shape[:2]
    
    # 统一尺寸
    orig_resized = cv2.resize(original, (width, height))
    shape_resized = cv2.resize(shape_result, (width, height))
    digit_resized = cv2.resize(digit_result, (width, height))
    
    # 水平拼接
    comparison = np.hstack([orig_resized, shape_resized, digit_resized])
    
    # 添加标题
    titles = ["Original", "Shape Detection", "Digit Recognition"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    for i, title in enumerate(titles):
        x = i * width + 10
        y = 30
        cv2.putText(comparison, title, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(comparison, title, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return comparison


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "images/shape_number_test.jpg"
    
    if not os.path.exists(image_path):
        print(f"[提示] 图像不存在: {image_path}")
        print("[提示] 请先运行 generate_test_images.py 生成测试图像")
    else:
        run_shape_recognition(image_path)
