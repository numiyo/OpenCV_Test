"""
多条件目标精定位模块（进阶）
功能：融合颜色、轮廓、角点检测，实现模拟装甲板目标的精确定位
输出旋转角度、中心坐标
"""

import cv2
import numpy as np
import os
import math


class ArmorDetector:
    """
    装甲板检测器类
    融合多种检测方法实现装甲板精确定位
    """
    
    def __init__(self):
        # HSV颜色阈值（装甲板通常为红蓝两色）
        self.color_ranges = {
            "red": {
                "lower1": np.array([0, 120, 100]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([160, 120, 100]),
                "upper2": np.array([179, 255, 255])
            },
            "blue": {
                "lower": np.array([100, 120, 100]),
                "upper": np.array([130, 255, 255])
            }
        }
        
        # 装甲板几何参数（根据实际场景调整）
        self.armor_params = {
            "min_aspect_ratio": 1.5,    # 最小宽高比
            "max_aspect_ratio": 4.0,    # 最大宽高比
            "min_area": 200,            # 最小面积
            "max_area": 5000,           # 最大面积
            "max_angle_diff": 15,       # 两灯条最大角度差
            "max_height_diff_ratio": 0.3,  # 最大高度差比例
            "max_center_dist_ratio": 4.0   # 最大中心距与灯条高度比
        }
    
    def preprocess(self, image):
        """
        预处理图像
        
        参数:
            image: 输入BGR图像
        返回:
            gray: 灰度图
            blurred: 模糊图
            hsv: HSV图
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return gray, blurred, hsv
    
    def detect_color_regions(self, hsv, color="blue"):
        """
        检测指定颜色的区域
        
        参数:
            hsv: HSV格式图像
            color: 目标颜色
        返回:
            mask: 颜色掩码
        """
        if color == "red":
            mask1 = cv2.inRange(hsv, self.color_ranges["red"]["lower1"], 
                               self.color_ranges["red"]["upper1"])
            mask2 = cv2.inRange(hsv, self.color_ranges["red"]["lower2"], 
                               self.color_ranges["red"]["upper2"])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, self.color_ranges["blue"]["lower"], 
                              self.color_ranges["blue"]["upper"])
        
        # 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def find_light_bars(self, mask, gray):
        """
        查找灯条（装甲板的发光部分）
        
        参数:
            mask: 颜色掩码
            gray: 灰度图
        返回:
            light_bars: 灯条列表
        """
        # 在掩码中查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        light_bars = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.armor_params["min_area"] or area > self.armor_params["max_area"]:
                continue
            
            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # 获取矩形参数
            center, (width, height), angle = rect
            
            # 确保width是长边
            if width < height:
                width, height = height, width
                angle = angle + 90 if angle < 0 else angle - 90
            
            # 宽高比检查
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < self.armor_params["min_aspect_ratio"] or \
               aspect_ratio > self.armor_params["max_aspect_ratio"]:
                continue
            
            # 计算灯条中心亮度（验证是否为发光体）
            mask_roi = np.zeros_like(mask)
            cv2.drawContours(mask_roi, [contour], -1, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask_roi)[0]
            
            if mean_brightness < 50:  # 亮度阈值
                continue
            
            light_bar = {
                "center": (int(center[0]), int(center[1])),
                "width": width,
                "height": height,
                "angle": angle,
                "area": area,
                "box": box,
                "contour": contour,
                "brightness": mean_brightness
            }
            light_bars.append(light_bar)
        
        # 按x坐标排序
        light_bars.sort(key=lambda x: x["center"][0])
        
        return light_bars
    
    def match_armor_pairs(self, light_bars):
        """
        匹配灯条对，识别装甲板
        
        参数:
            light_bars: 灯条列表
        返回:
            armors: 装甲板列表
        """
        armors = []
        n = len(light_bars)
        
        for i in range(n):
            for j in range(i + 1, n):
                bar1 = light_bars[i]
                bar2 = light_bars[j]
                
                # 角度差检查
                angle_diff = abs(bar1["angle"] - bar2["angle"])
                if angle_diff > self.armor_params["max_angle_diff"]:
                    continue
                
                # 高度差检查
                height_diff = abs(bar1["height"] - bar2["height"])
                avg_height = (bar1["height"] + bar2["height"]) / 2
                if height_diff / avg_height > self.armor_params["max_height_diff_ratio"]:
                    continue
                
                # 中心距检查
                dx = bar2["center"][0] - bar1["center"][0]
                dy = bar2["center"][1] - bar1["center"][1]
                center_dist = math.sqrt(dx * dx + dy * dy)
                
                if center_dist / avg_height > self.armor_params["max_center_dist_ratio"]:
                    continue
                
                # 计算装甲板参数
                armor_center = (
                    int((bar1["center"][0] + bar2["center"][0]) / 2),
                    int((bar1["center"][1] + bar2["center"][1]) / 2)
                )
                
                # 计算装甲板角度
                armor_angle = math.degrees(math.atan2(dy, dx))
                
                # 计算装甲板尺寸
                armor_width = center_dist
                armor_height = avg_height
                
                # 计算置信度
                confidence = self.calculate_confidence(bar1, bar2, angle_diff, height_diff, center_dist)
                
                armor = {
                    "left_bar": bar1,
                    "right_bar": bar2,
                    "center": armor_center,
                    "angle": armor_angle,
                    "width": armor_width,
                    "height": armor_height,
                    "confidence": confidence,
                    "box": self.get_armor_box(bar1, bar2)
                }
                armors.append(armor)
        
        # 按置信度排序
        armors.sort(key=lambda x: x["confidence"], reverse=True)
        
        # 非极大值抑制（去除重叠的装甲板）
        armors = self.nms(armors)
        
        return armors
    
    def calculate_confidence(self, bar1, bar2, angle_diff, height_diff, center_dist):
        """
        计算装甲板匹配的置信度
        
        参数:
            bar1, bar2: 两个灯条
            angle_diff: 角度差
            height_diff: 高度差
            center_dist: 中心距
        返回:
            confidence: 置信度 (0-1)
        """
        # 角度相似度
        angle_score = 1 - angle_diff / self.armor_params["max_angle_diff"]
        
        # 高度相似度
        avg_height = (bar1["height"] + bar2["height"]) / 2
        height_score = 1 - height_diff / (avg_height * self.armor_params["max_height_diff_ratio"])
        
        # 距离合理性（根据实际装甲板尺寸）
        dist_ratio = center_dist / avg_height
        optimal_ratio = 2.5  # 最优距离比
        dist_score = 1 - abs(dist_ratio - optimal_ratio) / optimal_ratio
        dist_score = max(0, dist_score)
        
        # 综合置信度
        confidence = (angle_score * 0.4 + height_score * 0.3 + dist_score * 0.3)
        
        return confidence
    
    def nms(self, armors, threshold=0.5):
        """
        非极大值抑制
        
        参数:
            armors: 装甲板列表
            threshold: IoU阈值
        返回:
            filtered: 过滤后的装甲板列表
        """
        if not armors:
            return []
        
        filtered = []
        for armor in armors:
            is_suppressed = False
            for kept in filtered:
                iou = self.calculate_iou(armor, kept)
                if iou > threshold:
                    is_suppressed = True
                    break
            if not is_suppressed:
                filtered.append(armor)
        
        return filtered
    
    def calculate_iou(self, armor1, armor2):
        """
        计算两个装甲板的IoU
        
        参数:
            armor1, armor2: 装甲板
        返回:
            iou: IoU值
        """
        # 简化为中心点距离判断
        dx = armor1["center"][0] - armor2["center"][0]
        dy = armor1["center"][1] - armor2["center"][1]
        dist = math.sqrt(dx * dx + dy * dy)
        
        avg_size = (armor1["width"] + armor2["width"]) / 4
        
        if dist < avg_size:
            return 1 - dist / avg_size
        return 0
    
    def get_armor_box(self, bar1, bar2):
        """
        获取装甲板的四个顶点
        
        参数:
            bar1, bar2: 两个灯条
        返回:
            box: 装甲板顶点数组
        """
        # 获取灯条的四个顶点
        box1 = bar1["box"]
        box2 = bar2["box"]
        
        # 按y坐标排序，取上下各两个点
        points1 = sorted(box1, key=lambda p: p[1])
        points2 = sorted(box2, key=lambda p: p[1])
        
        # 组合成装甲板四边形
        # 左灯条的下上点 + 右灯条的上下的点
        armor_box = np.array([
            points1[0],   # 左下
            points1[-1],  # 左上
            points2[-1],  # 右上
            points2[0]    # 右下
        ])
        
        return armor_box
    
    def detect_corners(self, image, armor):
        """
        使用Shi-Tomasi角点检测精确定位装甲板角点
        
        参数:
            image: 原始图像
            armor: 装甲板信息
        返回:
            corners: 角点列表
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 在装甲板区域内检测角点
        x1 = max(0, int(armor["center"][0] - armor["width"] / 2 - 10))
        y1 = max(0, int(armor["center"][1] - armor["height"] / 2 - 10))
        x2 = min(gray.shape[1], int(armor["center"][0] + armor["width"] / 2 + 10))
        y2 = min(gray.shape[0], int(armor["center"][1] + armor["height"] / 2 + 10))
        
        roi = gray[y1:y2, x1:x2]
        
        # Shi-Tomasi角点检测
        corners = cv2.goodFeaturesToTrack(
            roi, 
            maxCorners=4,
            qualityLevel=0.1,
            minDistance=10
        )
        
        if corners is not None:
            corners = corners.reshape(-1, 2)
            # 转换回全局坐标
            corners += [x1, y1]
            corners = corners.astype(int)
        else:
            # 如果没有检测到角点，使用盒子顶点
            corners = armor["box"]
        
        return corners
    
    def detect(self, image, target_color="blue"):
        """
        执行完整的装甲板检测流程
        
        参数:
            image: 输入图像
            target_color: 目标颜色
        返回:
            results: 检测结果
        """
        # 预处理
        gray, blurred, hsv = self.preprocess(image)
        
        # 颜色检测
        color_mask = self.detect_color_regions(hsv, target_color)
        
        # 查找灯条
        light_bars = self.find_light_bars(color_mask, gray)
        
        # 匹配装甲板
        armors = self.match_armor_pairs(light_bars)
        
        # 角点精确定位
        for armor in armors:
            corners = self.detect_corners(image, armor)
            armor["corners"] = corners
        
        return {
            "light_bars": light_bars,
            "armors": armors,
            "color_mask": color_mask
        }
    
    def visualize(self, image, results):
        """
        可视化检测结果
        
        参数:
            image: 原始图像
            results: 检测结果
        返回:
            vis_image: 可视化图像
        """
        vis_image = image.copy()
        
        # 绘制灯条
        for bar in results["light_bars"]:
            cv2.drawContours(vis_image, [bar["box"]], -1, (0, 255, 255), 2)
            cv2.circle(vis_image, bar["center"], 3, (0, 255, 0), -1)
        
        # 绘制装甲板
        for i, armor in enumerate(results["armors"]):
            # 绘制装甲板框
            cv2.drawContours(vis_image, [armor["box"]], -1, (0, 0, 255), 2)
            
            # 绘制角点
            if "corners" in armor:
                for corner in armor["corners"]:
                    cv2.circle(vis_image, tuple(corner), 5, (255, 0, 0), -1)
            
            # 绘制中心点
            cv2.circle(vis_image, armor["center"], 5, (0, 255, 0), -1)
            
            # 绘制角度线
            angle_rad = math.radians(armor["angle"])
            line_length = 50
            end_x = int(armor["center"][0] + line_length * math.cos(angle_rad))
            end_y = int(armor["center"][1] + line_length * math.sin(angle_rad))
            cv2.line(vis_image, armor["center"], (end_x, end_y), (255, 255, 0), 2)
            
            # 添加文字信息
            text = f"ID:{i+1} C:{armor['center']} A:{armor['angle']:.1f}"
            cv2.putText(vis_image, text, 
                       (armor["center"][0] - 50, armor["center"][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            conf_text = f"Conf:{armor['confidence']:.2f}"
            cv2.putText(vis_image, conf_text,
                       (armor["center"][0] - 50, armor["center"][1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 添加统计信息
        info_text = f"Light Bars: {len(results['light_bars'])}, Armors: {len(results['armors'])}"
        cv2.putText(vis_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image


def run_armor_detection(image_path, output_dir="output", target_color="blue"):
    """
    运行装甲板检测流程
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
        target_color: 目标颜色
    """
    print("\n" + "="*50)
    print("【多条件目标精定位 - 装甲板检测】")
    print("="*50)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"[错误] 无法读取图像: {image_path}")
        return None
    
    # 创建检测器
    detector = ArmorDetector()
    
    # 执行检测
    results = detector.detect(image, target_color)
    
    # 可视化
    vis_image = detector.visualize(image, results)
    
    # 打印结果
    print(f"\n[检测结果]")
    print(f"  - 检测到 {len(results['light_bars'])} 个灯条")
    print(f"  - 匹配到 {len(results['armors'])} 个装甲板")
    
    for i, armor in enumerate(results["armors"]):
        print(f"\n  装甲板 {i+1}:")
        print(f"    - 中心坐标: {armor['center']}")
        print(f"    - 旋转角度: {armor['angle']:.2f}°")
        print(f"    - 尺寸: {armor['width']:.1f} x {armor['height']:.1f}")
        print(f"    - 置信度: {armor['confidence']:.2f}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    cv2.imwrite(f"{output_dir}/{base_name}_armor_mask.jpg", results["color_mask"])
    cv2.imwrite(f"{output_dir}/{base_name}_armor_result.jpg", vis_image)
    
    # 创建对比图
    comparison = create_armor_comparison(image, results["color_mask"], vis_image)
    cv2.imwrite(f"{output_dir}/{base_name}_armor_comparison.jpg", comparison)
    
    print("\n" + "="*50)
    print("【装甲板检测完成】")
    print("="*50 + "\n")
    
    return results


def create_armor_comparison(original, mask, result):
    """
    创建装甲板检测对比图
    
    参数:
        original: 原图
        mask: 颜色掩码
        result: 检测结果
    返回:
        comparison: 对比图
    """
    height, width = original.shape[:2]
    
    # 转换掩码为3通道
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 统一尺寸
    orig_resized = cv2.resize(original, (width, height))
    mask_resized = cv2.resize(mask_display, (width, height))
    result_resized = cv2.resize(result, (width, height))
    
    # 水平拼接
    comparison = np.hstack([orig_resized, mask_resized, result_resized])
    
    # 添加标题
    titles = ["Original", "Color Mask", "Detection Result"]
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
        image_path = "images/armor_test.jpg"
    
    target_color = sys.argv[2] if len(sys.argv) > 2 else "blue"
    
    if not os.path.exists(image_path):
        print(f"[提示] 图像不存在: {image_path}")
        print("[提示] 请先运行 generate_test_images.py 生成测试图像")
    else:
        run_armor_detection(image_path, target_color=target_color)
