# 工业零件 X 光图像阈值分割与缺陷标注

本项目对应数字图像处理课程设计选题 2，目标是对工业零件或焊缝 X 光图像进行预处理、缺陷区域分割、缺陷标注与特征统计，并输出课程设计报告所需的结果图和统计表。

## 项目目标

- 对 X 光图像进行去噪和对比度增强
- 自动定位焊缝主体区域，减少文字、编号等非缺陷内容的干扰
- 提取疑似缺陷区域并进行框选标注
- 统计缺陷面积、周长、外接矩形等特征
- 生成实验结果图、统计表和课程设计材料

## 目录结构

```text
data/
  raw/                      原始 X 光图像
  processed/                预处理后的灰度图
docs/
  experiment_analysis.md    实验分析草稿
  issue3_checklist.md       当前任务清单
  references.md             参考文献整理
  report_outline.md         报告提纲
final/
  report.docx               最终报告
  slides.pptx               最终答辩材料
results/
  images/
    preprocess/             预处理对比图
    segmentation/           缺陷分割与标注结果图
  tables/
    defect_statistics.csv   缺陷统计结果
src/
  preprocess.py             预处理脚本
  segment_defects.py        分割与标注脚本
  contour_analysis.py       缺陷统计脚本
```

## 成员分工

- `lx`：图像收集、灰度化、滤波去噪、增强处理
- `lsj`：阈值分割、轮廓提取、缺陷标注
- `zjy`：缺陷特征统计、实验结果分析、README 与最终材料整理

## 运行说明

当前项目主流程基于 OpenCV 实现，推荐使用 Python 3.10 及以上版本。

### 1. 安装依赖

```powershell
pip install opencv-python numpy pillow
```

如果后续需要绘制更完整的统计图表，可额外安装：

```powershell
pip install matplotlib pandas
```

### 2. 准备数据

将原始 X 光图像放入 `data/raw/`，支持以下格式：

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

### 3. 执行流程

在项目根目录依次运行：

```powershell
python src/preprocess.py --input-dir data/raw
python src/segment_defects.py --input-dir data/processed
python src/contour_analysis.py --input-dir results/images/segmentation --output results/tables/defect_statistics.csv --filename-keyword _weld_mask
```

### 4. 各脚本功能

- `src/preprocess.py`
  - 使用中值滤波、非局部均值去噪和 CLAHE 完成预处理
  - 输出增强后的灰度图到 `data/processed/`
  - 输出原图与预处理图对比图到 `results/images/preprocess/`
- `src/segment_defects.py`
  - 自动检测焊缝亮带
  - 仅在焊缝中心检测区域内查找暗缺陷候选
  - 输出掩码图、标注图和对比图到 `results/images/segmentation/`
- `src/contour_analysis.py`
  - 读取二值掩码图
  - 统计面积、周长、外接框等特征
  - 输出 CSV 统计表到 `results/tables/defect_statistics.csv`

### 5. 输出结果

运行完成后，主要结果如下：

- `results/images/preprocess/`：预处理对比图
- `results/images/segmentation/`：缺陷分割与标注结果图
- `results/tables/defect_statistics.csv`：缺陷统计表

## 当前仓库状态

- 已完成项目目录和课程设计文档骨架
- 已完成基于 OpenCV 的预处理、分割、标注和统计主流程
- 已能对测试图像输出结果图和缺陷统计表
- 后续可继续进行参数调优、结果筛选和课程设计报告整理
