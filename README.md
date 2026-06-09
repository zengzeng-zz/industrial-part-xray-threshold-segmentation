# 工业零件 X 光图像阈值分割与缺陷标注

本项目对应数字图像处理课程设计课题 2，目标是对工业零件 X 光图像进行预处理、阈值分割、缺陷区域提取和统计分析，并形成可复现的实验结果。

## 项目目标

- 对工业零件 X 光图像进行基础预处理，减弱噪声并提升对比度
- 实现全局阈值、Otsu、自适应阈值等分割方法
- 自动提取缺陷轮廓并完成缺陷框选标注
- 统计缺陷区域的面积、周长、外接矩形等特征
- 输出结果图、统计表和课程设计报告素材

## 目录结构

```text
data/
  raw/                原始 X 光图像
  processed/          预处理后的图像
docs/
  references.md       参考文献整理
  report_outline.md   报告写作提纲
  experiment_analysis.md  实验分析草稿
final/
  report.docx         最终报告
  slides.pptx         最终答辩材料
results/
  images/
    preprocess/       预处理结果图
    segmentation/     分割与标注结果图
  tables/
    defect_statistics.csv  缺陷统计结果
src/
  contour_analysis.py 缺陷特征统计模块
```

## 成员分工

- 组员 A：图像收集、灰度化、滤波去噪、增强处理
- 组员 B：阈值分割、轮廓提取、缺陷标注
- 组长：缺陷特征统计、实验结果分析、README 与最终材料整理

## 运行说明

当前仓库优先保证 Issue 3 对应内容完整，因此先提供缺陷统计模块。等组员 A、B 提交预处理图和分割结果后，可直接运行以下命令生成统计表：

```powershell
python src/contour_analysis.py --input-dir results/images/segmentation --output results/tables/defect_statistics.csv
```

脚本支持的输入包括：

- 二值缺陷掩码图
- 已经分割好的缺陷区域图
- 黑白两色明显可分的标注结果图

## 建议协作流程

1. 每名成员在自己的分支上开发并提交 commit。
2. 每个阶段完成后发起 Pull Request 到 `main`。
3. 合并前至少由另一名成员进行一次 review。
4. 组长负责汇总结果图、统计表、报告和最终答辩材料。

## 当前仓库状态

- 已整理课题说明和项目骨架
- 已补充缺陷统计模块与文档模板
- 待补充预处理代码、阈值分割代码和真实实验数据
