# OpenCV图像识别测试项目

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 视觉组 - OpenCV图像识别综合实现

## 项目简介

本项目是视觉组OpenCV图像识别的完整实现，涵盖图像预处理、颜色识别、几何图形识别、数字识别、装甲板检测等多个计算机视觉基础任务。项目采用模块化设计，代码结构清晰，注释详尽，便于学习和二次开发。

## 考核任务清单

### 一、基础必做任务（全部完成）

| 任务 | 描述 | 实现文件 |
|------|------|----------|
| 图像基础预处理 | 读取、灰度转换、高斯模糊、直方图均衡化 | `basic_preprocessing.py` |
| 颜色阈值色块识别 | HSV空间红/蓝色分割、形态学去噪、轮廓标注 | `color_detection.py` |
| 简单特征识别 | 矩形/圆形识别、0-9印刷体数字识别（无OCR库） | `shape_number_recognition.py` |

### 二、进阶选做任务（全部完成）

| 任务 | 描述 | 实现文件 |
|------|------|----------|
| 多条件目标精定位 | 融合颜色/轮廓/角点检测，装甲板精确定位 | `armor_detection.py` |
| 鲁棒性测试 | 不同光照/遮挡场景测试，参数调优报告 | `main.py` |
| 交互调参工具 | OpenCV Trackbar可视化调参界面 | `parameter_tuner.py` |

## 项目结构

```
OpenCV_Test/
├── src/                          # 源代码目录
│   ├── main.py                   # 主程序入口
│   ├── basic_preprocessing.py    # 图像基础预处理
│   ├── color_detection.py        # 颜色阈值色块识别
│   ├── shape_number_recognition.py # 几何图形与数字识别
│   ├── armor_detection.py        # 装甲板精定位
│   ├── parameter_tuner.py        # 交互调参工具
│   └── generate_test_images.py   # 测试图像生成
├── images/                       # 测试图像目录
│   ├── basic_test.jpg            # 基础预处理测试图
│   ├── color_test.jpg            # 颜色识别测试图
│   ├── shape_number_test.jpg     # 形状数字识别测试图
│   ├── armor_test.jpg            # 装甲板检测测试图
│   ├── lighting/                 # 光照测试图像组
│   └── occlusion/                # 遮挡测试图像组
├── output/                       # 输出结果目录
├── docs/                         # 文档目录
└── README.md                     # 项目说明文档
```

## 快速开始

### 环境配置

#### 系统要求
- Python 3.7+
- OpenCV 4.5+
- NumPy 1.19+

#### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖包
pip install opencv-python numpy
```

### 运行项目

#### 方式一：交互模式
```bash
cd src
python main.py
```

#### 方式二：命令行模式
```bash
# 运行全部测试
cd src
python main.py -t all

# 运行指定测试
cd src
python main.py -t preprocess -i ../images/basic_test.jpg
python main.py -t color -i ../images/color_test.jpg
python main.py -t shape -i ../images/shape_number_test.jpg
python main.py -t armor -i ../images/armor_test.jpg -c blue

# 生成测试图像
cd src
python main.py --generate

# 交互调参
cd src
python main.py --tuner ../images/color_test.jpg color
```

#### 方式三：单独运行模块
```bash
cd src

# 图像预处理
python basic_preprocessing.py ../images/basic_test.jpg

# 颜色识别
python color_detection.py ../images/color_test.jpg

# 形状数字识别
python shape_number_recognition.py ../images/shape_number_test.jpg

# 装甲板检测
python armor_detection.py ../images/armor_test.jpg blue

# 生成测试图像
python generate_test_images.py
```

## 实现思路

### 1. 图像基础预处理

```
输入图像 → 灰度转换 → 高斯模糊去噪 → 直方图均衡化 → 输出对比图
```

**关键算法：**
- `cv2.cvtColor()`: BGR转灰度/HSV
- `cv2.GaussianBlur()`: 高斯滤波去噪
- `cv2.equalizeHist()`: 直方图均衡化增强对比度

### 2. 颜色阈值色块识别

```
输入图像 → BGR转HSV → 颜色阈值分割 → 形态学去噪 → 轮廓查找 → 标注信息
```

**关键算法：**
- HSV颜色空间阈值分割（红色需处理0度跨越）
- 形态学开运算去噪、闭运算填补孔洞
- 轮廓面积筛选，计算中心点坐标

### 3. 几何图形与数字识别

#### 几何图形识别
```
预处理 → 边缘检测 → 轮廓逼近 → 顶点数分类 → 形状判定
```

**形状分类规则：**
| 顶点数 | 形状 |
|--------|------|
| 3 | 三角形 |
| 4 | 矩形/正方形（根据宽高比） |
| 5 | 五边形 |
| 6 | 六边形 |
| >6 | 圆形/椭圆（根据圆形度） |

#### 数字识别（模板匹配法）
```
预处理二值化 → 轮廓分割 → 尺寸归一化 → 模板匹配 → 置信度筛选
```

### 4. 装甲板精定位

```
输入图像 → 颜色检测 → 灯条查找 → 灯条配对 → 装甲板匹配 → 角点精定位
```

**匹配约束条件：**
- 角度差 < 15°
- 高度差比例 < 30%
- 中心距与高度比在合理范围
- 非极大值抑制去重

## 测试结果

### 基础任务测试结果

| 测试项目 | 测试图像 | 输出结果 | 状态 |
|----------|----------|----------|------|
| 图像预处理 | basic_test.jpg | 灰度/模糊/均衡化对比图 | 通过 |
| 颜色识别 | color_test.jpg | 红/蓝色掩码+标注图 | 通过 |
| 形状识别 | shape_number_test.jpg | 几何图形分类标注 | 通过 |
| 数字识别 | shape_number_test.jpg | 0-9数字识别结果 | 通过 |

### 进阶任务测试结果

| 测试项目 | 测试图像 | 输出结果 | 状态 |
|----------|----------|----------|------|
| 装甲板检测 | armor_test.jpg | 装甲板定位+角度+坐标 | 通过 |
| 光照鲁棒性 | lighting/*.jpg | 不同光照下识别率统计 | 通过 |
| 遮挡鲁棒性 | occlusion/*.jpg | 不同遮挡程度识别率 | 通过 |
| 交互调参 | 任意图像 | 实时参数调节界面 | 通过 |

## 参数调优报告

### HSV颜色阈值参考值

| 颜色 | H下限 | H上限 | S下限 | S上限 | V下限 | V上限 |
|------|-------|-------|-------|-------|-------|-------|
| 红色 | 0-10, 160-179 | - | 100 | 255 | 100 | 255 |
| 蓝色 | 100 | 130 | 100 | 255 | 100 | 255 |
| 绿色 | 40 | 80 | 50 | 255 | 50 | 255 |

### 形态学操作参数

| 操作 | 核大小 | 迭代次数 | 用途 |
|------|--------|----------|------|
| 开运算 | 5×5 | 1 | 去除小噪点 |
| 闭运算 | 5×5 | 1 | 填补小孔洞 |

### Canny边缘检测参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 低阈值 | 50 | 边缘连接阈值 |
| 高阈值 | 150 | 强边缘阈值 |
| 高斯核 | 5×5 | 去噪预处理 |

## 参考资料

### 官方文档
- [OpenCV官方文档](https://docs.opencv.org/4.x/)
- [OpenCV-Python教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### 核心算法参考
1. **颜色空间转换**
   - HSV颜色模型原理
   - 红色在HSV中的特殊性处理

2. **形态学操作**
   - 开运算/闭运算定义与应用
   - 结构元素选择与优化

3. **轮廓分析**
   - 轮廓逼近算法（Douglas-Peucker）
   - 形状描述符计算

4. **模板匹配**
   - 归一化相关系数匹配
   - 多尺度模板匹配策略

5. **角点检测**
   - Shi-Tomasi角点检测算法
   - 亚像素级角点精定位

### 开源项目参考
- OpenCV官方示例代码
- RoboMaster视觉开源方案

## 代码规范

### 注释规范
- 所有函数必须包含docstring说明
- 关键算法步骤添加行内注释
- 复杂逻辑添加段落说明

### 命名规范
- 函数名：小写下划线（snake_case）
- 类名：大驼峰（CamelCase）
- 常量：全大写下划线（UPPER_CASE）

### 模块化设计
- 单一职责原则
- 高内聚低耦合
- 可复用的工具函数

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---

> 💡 **提示**: 本项目为考核练习用途，代码仅供学习参考。实际应用中请根据具体场景调整参数。
