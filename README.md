# PostTrain Causal Lab: manuscript branch

本分支按“入门 → 研究设计 → 执行协议 → 实测结果”的顺序组织：

1. [一步步认识大模型后训练因果追踪项目-friendly.md](一步步认识大模型后训练因果追踪项目-friendly.md)
2. [RESEARCH.md](RESEARCH.md)
3. [RUN.md](RUN.md)
4. [A0 REPORT.md](experiments/a0-20261027-110309/REPORT.md)
5. [24-H20冻结协议](experiments/scale24-20261027/PROTOCOL.md) 与 [分析口径](experiments/scale24-20261027/ANALYSIS.md)

24-H20矩阵已提交排队：三配对seed × SFT／KD／OPD-32／OPD-256，每cell两卡、256步。沿用4B／9B，主要检验输出预算改变后的完整数字正确率和固定前缀干预效果；当前没有该轮实测结果。A0的四步结果仍只作为工程门禁证据。

配套图：

- [干预设计 SVG](figures/intervention-design.svg)
- [可复现绘图入口](figures/render_figure.py)

精简机读证据位于 `experiments/a0-20261027-110309/evidence/`。内部任务链接、调度配置、平台日志和大体积运行归档没有公开。

核心实现位于同一仓库的 [`code` 分支](https://github.com/Cheongroach/posttrain-causal-lab/tree/code)。同时签出三个分支的方法见 [`main` 分支阅读指引](https://github.com/Cheongroach/posttrain-causal-lab/blob/main/READING_GUIDE.md)。
