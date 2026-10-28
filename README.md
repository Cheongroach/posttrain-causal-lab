# PostTrain Causal Lab: manuscript branch

本分支按“入门 → 研究设计 → 执行协议 → 实测结果”的顺序组织：

1. [一步步认识大模型后训练因果追踪项目-friendly.md](一步步认识大模型后训练因果追踪项目-friendly.md)
2. [RESEARCH.md](RESEARCH.md)
3. [RUN.md](RUN.md)
4. [A0 REPORT.md](experiments/a0-20261027-110309/REPORT.md)
5. [24-H20冻结协议](experiments/scale24-20261027/PROTOCOL.md) 与 [分析口径](experiments/scale24-20261027/ANALYSIS.md)
6. [24-H20实测报告](experiments/scale24-20261027/REPORT.md)
7. [已有checkpoint的长度诊断协议](experiments/length-probe-20261028/PROTOCOL.md)

24-H20矩阵已完成并独立复核，12个cell各256步，实际32.82 GPU·h。OPD-256降低训练截断率（41.28%→27.60%），但验证正确率的seed差为−25.00／+23.44／+15.63个百分点，均值+4.69，收益仍不确定。固定标准答案前缀的donor干预平均效果在12个cell中均超过匹配随机，不能据此识别训练历史原因。A0的四步结果保持原工程适用范围。

为诊断seed差异，6-H20纯推理任务已启动：重评原OPD checkpoint的64／128／256步，在原dev比较256／512-token上限，并验证逐题CUDA RNG契约。当前没有该轮终态结果；优化更新为0，不能解释为RNG重训收益。

配套图：

- [干预设计 SVG](figures/intervention-design.svg)
- [可复现绘图入口](figures/render_figure.py)

精简机读证据位于 `experiments/a0-20261027-110309/evidence/`。内部任务链接、调度配置、平台日志和大体积运行归档没有公开。

核心实现位于同一仓库的 [`code` 分支](https://github.com/Cheongroach/posttrain-causal-lab/tree/code)。同时签出三个分支的方法见 [`main` 分支阅读指引](https://github.com/Cheongroach/posttrain-causal-lab/blob/main/READING_GUIDE.md)。
