"""
交互调参工具模块（进阶）
功能：基于OpenCV Trackbar创建可视化调参界面，实时展示参数调整效果
支持颜色阈值调参、形态学参数调参、Canny边缘检测调参等
"""

import cv2
import numpy as np
import os


class ParameterTuner:
    """
    参数调节器类
    提供可视化界面实时调节参数
    """
    
    def __init__(self):
        self.window_name = "Parameter Tuner"
        self.image = None
        self.original_image = None
        self.current_mode = "color"  # color, edge, morphology, threshold
        
        # 参数值存储
        self.params = {
            "color": {
                "h_min": 0,
                "h_max": 179,
                "s_min": 100,
                "s_max": 255,
                "v_min": 100,
                "v_max": 255
            },
            "edge": {
                "canny_low": 50,
                "canny_high": 150,
                "blur_kernel": 5,
                "sobel_ksize": 3
            },
            "morphology": {
                "kernel_size": 5,
                "operation": 0,  # 0:open, 1:close, 2:erode, 3:dilate
                "iterations": 1
            },
            "threshold": {
                "thresh_type": 0,  # 0:binary, 1:binary_inv, 2:trunc, 3:tozero, 4:tozero_inv
                "thresh_value": 127,
                "max_value": 255,
                "adaptive": 0  # 0:global, 1:mean, 2:gaussian
            }
        }
    
    def create_trackbars(self, mode):
        """
        创建指定模式的Trackbar
        
        参数:
            mode: 调参模式
        """
        self.current_mode = mode
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        if mode == "color":
            # HSV颜色阈值调节
            cv2.createTrackbar("H Min", self.window_name, 
                             self.params["color"]["h_min"], 179, 
                             lambda x: self.update_param("color", "h_min", x))
            cv2.createTrackbar("H Max", self.window_name, 
                             self.params["color"]["h_max"], 179, 
                             lambda x: self.update_param("color", "h_max", x))
            cv2.createTrackbar("S Min", self.window_name, 
                             self.params["color"]["s_min"], 255, 
                             lambda x: self.update_param("color", "s_min", x))
            cv2.createTrackbar("S Max", self.window_name, 
                             self.params["color"]["s_max"], 255, 
                             lambda x: self.update_param("color", "s_max", x))
            cv2.createTrackbar("V Min", self.window_name, 
                             self.params["color"]["v_min"], 255, 
                             lambda x: self.update_param("color", "v_min", x))
            cv2.createTrackbar("V Max", self.window_name, 
                             self.params["color"]["v_max"], 255, 
                             lambda x: self.update_param("color", "v_max", x))
        
        elif mode == "edge":
            # Canny边缘检测调参
            cv2.createTrackbar("Canny Low", self.window_name, 
                             self.params["edge"]["canny_low"], 255, 
                             lambda x: self.update_param("edge", "canny_low", x))
            cv2.createTrackbar("Canny High", self.window_name, 
                             self.params["edge"]["canny_high"], 255, 
                             lambda x: self.update_param("edge", "canny_high", x))
            cv2.createTrackbar("Blur Kernel", self.window_name, 
                             self.params["edge"]["blur_kernel"], 21, 
                             lambda x: self.update_param("edge", "blur_kernel", x))
            cv2.createTrackbar("Sobel KSize", self.window_name, 
                             self.params["edge"]["sobel_ksize"], 7, 
                             lambda x: self.update_param("edge", "sobel_ksize", x))
        
        elif mode == "morphology":
            # 形态学操作调参
            cv2.createTrackbar("Kernel Size", self.window_name, 
                             self.params["morphology"]["kernel_size"], 21, 
                             lambda x: self.update_param("morphology", "kernel_size", x))
            cv2.createTrackbar("Operation", self.window_name, 
                             self.params["morphology"]["operation"], 3, 
                             lambda x: self.update_param("morphology", "operation", x))
            cv2.createTrackbar("Iterations", self.window_name, 
                             self.params["morphology"]["iterations"], 10, 
                             lambda x: self.update_param("morphology", "iterations", x))
        
        elif mode == "threshold":
            # 阈值调节
            cv2.createTrackbar("Thresh Type", self.window_name, 
                             self.params["threshold"]["thresh_type"], 4, 
                             lambda x: self.update_param("threshold", "thresh_type", x))
            cv2.createTrackbar("Thresh Value", self.window_name, 
                             self.params["threshold"]["thresh_value"], 255, 
                             lambda x: self.update_param("threshold", "thresh_value", x))
            cv2.createTrackbar("Max Value", self.window_name, 
                             self.params["threshold"]["max_value"], 255, 
                             lambda x: self.update_param("threshold", "max_value", x))
            cv2.createTrackbar("Adaptive", self.window_name, 
                             self.params["threshold"]["adaptive"], 2, 
                             lambda x: self.update_param("threshold", "adaptive", x))
    
    def update_param(self, mode, param_name, value):
        """
        更新参数值
        
        参数:
            mode: 参数模式
            param_name: 参数名
            value: 新值
        """
        self.params[mode][param_name] = value
        self.refresh_display()
    
    def refresh_display(self):
        """
        刷新显示
        """
        if self.image is None:
            return
        
        result = self.process_image()
        
        # 创建对比显示
        if len(result.shape) == 2:
            result_display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            result_display = result
        
        # 确保尺寸一致
        h, w = self.original_image.shape[:2]
        result_display = cv2.resize(result_display, (w, h))
        
        # 水平拼接
        comparison = np.hstack([self.original_image, result_display])
        
        # 添加信息文字
        info_text = f"Mode: {self.current_mode.upper()}"
        cv2.putText(comparison, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加参数信息
        y_offset = 60
        for key, value in self.params[self.current_mode].items():
            param_text = f"{key}: {value}"
            cv2.putText(comparison, param_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        cv2.imshow(self.window_name, comparison)
    
    def process_image(self):
        """
        根据当前模式和参数处理图像
        
        返回:
            result: 处理结果
        """
        if self.current_mode == "color":
            return self.process_color()
        elif self.current_mode == "edge":
            return self.process_edge()
        elif self.current_mode == "morphology":
            return self.process_morphology()
        elif self.current_mode == "threshold":
            return self.process_threshold()
        
        return self.image
    
    def process_color(self):
        """
        HSV颜色阈值处理
        """
        # 转换为HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # 获取参数
        h_min = self.params["color"]["h_min"]
        h_max = self.params["color"]["h_max"]
        s_min = self.params["color"]["s_min"]
        s_max = self.params["color"]["s_max"]
        v_min = self.params["color"]["v_min"]
        v_max = self.params["color"]["v_max"]
        
        # 创建掩码
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # 应用掩码
        result = cv2.bitwise_and(self.image, self.image, mask=mask)
        
        return result
    
    def process_edge(self):
        """
        Canny边缘检测处理
        """
        # 灰度转换
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 获取参数
        canny_low = self.params["edge"]["canny_low"]
        canny_high = self.params["edge"]["canny_high"]
        blur_kernel = self.params["edge"]["blur_kernel"]
        
        # 确保核大小为奇数
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        return edges
    
    def process_morphology(self):
        """
        形态学操作处理
        """
        # 灰度转换
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 获取参数
        kernel_size = self.params["morphology"]["kernel_size"]
        operation = self.params["morphology"]["operation"]
        iterations = self.params["morphology"]["iterations"]
        
        # 确保核大小至少为1
        kernel_size = max(1, kernel_size)
        
        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # 执行操作
        operations = [cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_ERODE, cv2.MORPH_DILATE]
        op = operations[min(operation, len(operations) - 1)]
        
        result = cv2.morphologyEx(binary, op, kernel, iterations=iterations)
        
        return result
    
    def process_threshold(self):
        """
        阈值处理
        """
        # 灰度转换
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        # 获取参数
        thresh_type = self.params["threshold"]["thresh_type"]
        thresh_value = self.params["threshold"]["thresh_value"]
        max_value = self.params["threshold"]["max_value"]
        adaptive = self.params["threshold"]["adaptive"]
        
        # 阈值类型
        thresh_types = [
            cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV,
            cv2.THRESH_TRUNC,
            cv2.THRESH_TOZERO,
            cv2.THRESH_TOZERO_INV
        ]
        ttype = thresh_types[min(thresh_type, len(thresh_types) - 1)]
        
        # 自适应阈值
        if adaptive == 0:
            # 全局阈值
            _, result = cv2.threshold(gray, thresh_value, max_value, ttype)
        elif adaptive == 1:
            # 自适应均值
            result = cv2.adaptiveThreshold(gray, max_value, 
                                          cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          ttype, 11, 2)
        else:
            # 自适应高斯
            result = cv2.adaptiveThreshold(gray, max_value, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          ttype, 11, 2)
        
        return result
    
    def run(self, image_path, mode="color"):
        """
        运行调参工具
        
        参数:
            image_path: 图像路径
            mode: 调参模式 (color, edge, morphology, threshold)
        """
        # 读取图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"[错误] 无法读取图像: {image_path}")
            return
        
        self.original_image = self.image.copy()
        
        print("\n" + "="*50)
        print("【交互调参工具】")
        print("="*50)
        print(f"模式: {mode}")
        print("操作说明:")
        print("  - 拖动Trackbar调节参数")
        print("  - 按 'S' 保存当前参数")
        print("  - 按 'Q' 或 ESC 退出")
        print("="*50 + "\n")
        
        # 创建Trackbar
        self.create_trackbars(mode)
        
        # 初始显示
        self.refresh_display()
        
        # 事件循环
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC 或 Q
                break
            elif key == ord('s'):  # 保存参数
                self.save_params()
        
        cv2.destroyAllWindows()
        print("[提示] 调参工具已关闭")
    
    def save_params(self):
        """
        保存当前参数到文件
        """
        filename = f"params_{self.current_mode}.txt"
        filepath = os.path.join("output", filename)
        
        os.makedirs("output", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {self.current_mode.upper()} Parameters\n")
            for key, value in self.params[self.current_mode].items():
                f.write(f"{key}: {value}\n")
        
        print(f"[保存] 参数已保存到: {filepath}")


def print_usage():
    """
    打印使用说明
    """
    print("""
========================================
    OpenCV 交互调参工具
========================================

使用方法:
    python parameter_tuner.py <图像路径> [模式]

可用模式:
    color       - HSV颜色阈值调参
    edge        - Canny边缘检测调参
    morphology  - 形态学操作调参
    threshold   - 阈值处理调参

示例:
    python parameter_tuner.py test.jpg color
    python parameter_tuner.py test.jpg edge

快捷键:
    S - 保存当前参数
    Q/ESC - 退出

========================================
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "color"
    
    if not os.path.exists(image_path):
        print(f"[错误] 图像不存在: {image_path}")
        sys.exit(1)
    
    valid_modes = ["color", "edge", "morphology", "threshold"]
    if mode not in valid_modes:
        print(f"[错误] 无效模式: {mode}")
        print(f"可用模式: {', '.join(valid_modes)}")
        sys.exit(1)
    
    tuner = ParameterTuner()
    tuner.run(image_path, mode)
