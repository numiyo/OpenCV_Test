"""
OpenCV图像识别考核 - 主程序入口
整合所有模块，提供统一的运行接口
"""

import cv2
import numpy as np
import os
import sys
import argparse

# 导入各功能模块
from basic_preprocessing import run_preprocessing_pipeline
from color_detection import detect_color_targets
from shape_number_recognition import run_shape_recognition
from armor_detection import run_armor_detection
from parameter_tuner import ParameterTuner
from generate_test_images import generate_all_test_images


def print_banner():
    """打印程序欢迎信息"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           OpenCV 图像识别考核 - 综合测试平台                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """打印功能菜单"""
    print("""
【功能菜单】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基础必做任务:
  1. 图像基础预处理
  2. 颜色阈值色块识别
  3. 几何图形与数字识别

进阶选做任务:
  4. 装甲板精定位（多条件目标检测）
  5. 交互调参工具

辅助功能:
  6. 生成测试图像
  7. 运行全部测试
  8. 鲁棒性测试

其他:
  0. 退出程序
  h. 显示帮助

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def run_robustness_test(image_dir="images", output_dir="output"):
    """
    运行鲁棒性测试
    针对不同光照、遮挡场景测试算法，统计识别准确率
    
    参数:
        image_dir: 测试图像目录
        output_dir: 输出目录
    """
    print("\n" + "="*50)
    print("【鲁棒性测试】")
    print("="*50 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 光照测试
    lighting_dir = os.path.join(image_dir, "lighting")
    if os.path.exists(lighting_dir):
        print("[光照测试] 开始...")
        lighting_results = []
        
        for filename in sorted(os.listdir(lighting_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(lighting_dir, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # 使用颜色检测作为测试基准
                from color_detection import detect_color_targets
                results = detect_color_targets(image, target_colors=["red"], output_dir=output_dir)
                
                detection_count = len(results["detections"].get("red", {}).get("contours", []))
                lighting_results.append({
                    "filename": filename,
                    "detections": detection_count
                })
                
                print(f"  {filename}: 检测到 {detection_count} 个目标")
        
        # 保存光照测试结果
        with open(os.path.join(output_dir, "robustness_lighting.txt"), 'w', encoding='utf-8') as f:
            f.write("光照鲁棒性测试结果\n")
            f.write("="*50 + "\n\n")
            for result in lighting_results:
                f.write(f"{result['filename']}: {result['detections']} 个目标\n")
        
        print("[光照测试] 完成\n")
    
    # 遮挡测试
    occlusion_dir = os.path.join(image_dir, "occlusion")
    if os.path.exists(occlusion_dir):
        print("[遮挡测试] 开始...")
        occlusion_results = []
        
        for filename in sorted(os.listdir(occlusion_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(occlusion_dir, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # 使用几何图形检测作为测试基准
                from shape_number_recognition import detect_geometric_shapes, preprocess_for_recognition
                binary, _ = preprocess_for_recognition(image)
                shapes, _ = detect_geometric_shapes(image, binary)
                
                occlusion_results.append({
                    "filename": filename,
                    "shapes": len(shapes)
                })
                
                print(f"  {filename}: 检测到 {len(shapes)} 个图形")
        
        # 保存遮挡测试结果
        with open(os.path.join(output_dir, "robustness_occlusion.txt"), 'w', encoding='utf-8') as f:
            f.write("遮挡鲁棒性测试结果\n")
            f.write("="*50 + "\n\n")
            for result in occlusion_results:
                f.write(f"{result['filename']}: {result['shapes']} 个图形\n")
        
        print("[遮挡测试] 完成\n")
    
    print("="*50)
    print("【鲁棒性测试完成】")
    print(f"结果保存位置: {output_dir}")
    print("="*50 + "\n")


def run_all_tests(image_dir="images", output_dir="output"):
    """
    运行全部测试
    
    参数:
        image_dir: 测试图像目录
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("【运行全部测试】")
    print("="*60 + "\n")
    
    # 1. 基础预处理测试
    basic_image = os.path.join(image_dir, "basic_test.jpg")
    if os.path.exists(basic_image):
        print("\n>>> 测试1: 图像基础预处理")
        run_preprocessing_pipeline(basic_image, output_dir)
    
    # 2. 颜色识别测试
    color_image = os.path.join(image_dir, "color_test.jpg")
    if os.path.exists(color_image):
        print("\n>>> 测试2: 颜色阈值色块识别")
        image = cv2.imread(color_image)
        detect_color_targets(image, target_colors=["red", "blue"], output_dir=output_dir)
    
    # 3. 形状和数字识别测试
    shape_image = os.path.join(image_dir, "shape_number_test.jpg")
    if os.path.exists(shape_image):
        print("\n>>> 测试3: 几何图形与数字识别")
        run_shape_recognition(shape_image, output_dir)
    
    # 4. 装甲板检测测试
    armor_image = os.path.join(image_dir, "armor_test.jpg")
    if os.path.exists(armor_image):
        print("\n>>> 测试4: 装甲板精定位")
        run_armor_detection(armor_image, output_dir, target_color="blue")
    
    print("\n" + "="*60)
    print("【全部测试完成】")
    print(f"结果保存位置: {os.path.abspath(output_dir)}")
    print("="*60 + "\n")


def interactive_mode():
    """
    交互模式
    """
    print_banner()
    print_menu()
    
    while True:
        choice = input("请输入选项 (0-8/h): ").strip().lower()
        
        if choice == '0':
            print("\n感谢使用，再见！")
            break
        
        elif choice == 'h':
            print_menu()
        
        elif choice == '1':
            image_path = input("请输入图像路径 (默认: images/basic_test.jpg): ").strip()
            if not image_path:
                image_path = "images/basic_test.jpg"
            if os.path.exists(image_path):
                run_preprocessing_pipeline(image_path)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        
        elif choice == '2':
            image_path = input("请输入图像路径 (默认: images/color_test.jpg): ").strip()
            if not image_path:
                image_path = "images/color_test.jpg"
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                detect_color_targets(image)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        
        elif choice == '3':
            image_path = input("请输入图像路径 (默认: images/shape_number_test.jpg): ").strip()
            if not image_path:
                image_path = "images/shape_number_test.jpg"
            if os.path.exists(image_path):
                run_shape_recognition(image_path)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        
        elif choice == '4':
            image_path = input("请输入图像路径 (默认: images/armor_test.jpg): ").strip()
            if not image_path:
                image_path = "images/armor_test.jpg"
            color = input("请输入目标颜色 (blue/red, 默认: blue): ").strip()
            if not color:
                color = "blue"
            if os.path.exists(image_path):
                run_armor_detection(image_path, target_color=color)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        
        elif choice == '5':
            image_path = input("请输入图像路径: ").strip()
            mode = input("请输入调参模式 (color/edge/morphology/threshold, 默认: color): ").strip()
            if not mode:
                mode = "color"
            if os.path.exists(image_path):
                tuner = ParameterTuner()
                tuner.run(image_path, mode)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        
        elif choice == '6':
            generate_all_test_images()
        
        elif choice == '7':
            run_all_tests()
        
        elif choice == '8':
            run_robustness_test()
        
        else:
            print("[提示] 无效选项，请重新输入")


def command_line_mode():
    """
    命令行模式
    """
    parser = argparse.ArgumentParser(
        description='OpenCV图像识别考核 - 综合测试平台',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 交互模式
  python main.py -t all             # 运行全部测试
  python main.py -t preprocess -i test.jpg    # 预处理测试
  python main.py -t color -i test.jpg         # 颜色识别测试
  python main.py -t shape -i test.jpg         # 形状识别测试
  python main.py -t armor -i test.jpg         # 装甲板检测
  python main.py -t robustness                # 鲁棒性测试
  python main.py --generate                   # 生成测试图像
  python main.py --tuner test.jpg color       # 交互调参
        """
    )
    
    parser.add_argument('-t', '--test', 
                       choices=['preprocess', 'color', 'shape', 'armor', 'robustness', 'all'],
                       help='选择测试类型')
    parser.add_argument('-i', '--image', 
                       help='输入图像路径')
    parser.add_argument('-c', '--color', default='blue',
                       choices=['blue', 'red'],
                       help='装甲板目标颜色 (默认: blue)')
    parser.add_argument('-o', '--output', default='output',
                       help='输出目录 (默认: output)')
    parser.add_argument('--generate', action='store_true',
                       help='生成测试图像')
    parser.add_argument('--tuner', nargs=2, metavar=('IMAGE', 'MODE'),
                       help='交互调参工具 (图像路径 模式)')
    
    args = parser.parse_args()
    
    # 生成测试图像
    if args.generate:
        generate_all_test_images()
        return
    
    # 交互调参
    if args.tuner:
        image_path, mode = args.tuner
        if os.path.exists(image_path):
            tuner = ParameterTuner()
            tuner.run(image_path, mode)
        else:
            print(f"[错误] 图像不存在: {image_path}")
        return
    
    # 运行测试
    if args.test:
        if args.test == 'all':
            run_all_tests(output_dir=args.output)
        elif args.test == 'robustness':
            run_robustness_test(output_dir=args.output)
        elif args.test == 'preprocess':
            image_path = args.image or "images/basic_test.jpg"
            if os.path.exists(image_path):
                run_preprocessing_pipeline(image_path, args.output)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        elif args.test == 'color':
            image_path = args.image or "images/color_test.jpg"
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                detect_color_targets(image, output_dir=args.output)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        elif args.test == 'shape':
            image_path = args.image or "images/shape_number_test.jpg"
            if os.path.exists(image_path):
                run_shape_recognition(image_path, args.output)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        elif args.test == 'armor':
            image_path = args.image or "images/armor_test.jpg"
            if os.path.exists(image_path):
                run_armor_detection(image_path, args.output, args.color)
            else:
                print(f"[错误] 图像不存在: {image_path}")
        return
    
    # 默认进入交互模式
    interactive_mode()


if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        command_line_mode()
    else:
        interactive_mode()
