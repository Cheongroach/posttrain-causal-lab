# PostTrain Causal Lab

这个项目研究一个具体问题：后训练信号怎样经过 LoRA 参数更新和模型内部计算，最终改变输出行为；哪些观察属于执行证据，哪些还需要质量评测或训练干预。

早期 **A0 执行一致性实验** 已完成：

- Qwen3.5-4B student 与 Qwen3.5-9B teacher 的真实权重路径通过。
- 第一层 GDN identity 与原 donor restore 的 logits 最大绝对误差均为 0。
- adapter、optimizer、RNG 和数据游标恢复后，同一下一步的 adapter 最大绝对误差为 0。
- SFT、固定前缀 KD、学生前缀 OPD 各完成四步。
- 四个 OPD rollout 全部触及 32-token 上限；本轮没有模型质量或机制定位结论。

后续24-H20矩阵已完成并独立复核：3个seed × SFT／KD／OPD-32／OPD-256，每cell两卡、256步，沿用4B／9B，实际32.82 GPU·h。OPD-256降低训练截断率（41.28%→27.60%），验证正确率均值差+4.69个百分点，但seed差为−25.00／+23.44／+15.63，收益仍不确定。固定标准答案前缀下的donor干预超过匹配随机效果，尚未识别训练信号的历史原因。详见 [实测报告](https://github.com/Cheongroach/posttrain-causal-lab/blob/manuscript/experiments/scale24-20261027/REPORT.md)、[冻结协议](https://github.com/Cheongroach/posttrain-causal-lab/blob/manuscript/experiments/scale24-20261027/PROTOCOL.md) 和 [复核口径](https://github.com/Cheongroach/posttrain-causal-lab/blob/manuscript/experiments/scale24-20261027/ANALYSIS.md)。

## 三个分支

| 分支 | 内容 |
|---|---|
| [`main`](https://github.com/Cheongroach/posttrain-causal-lab/tree/main) | 项目入口和多 worktree 阅读指引 |
| [`code`](https://github.com/Cheongroach/posttrain-causal-lab/tree/code) | A0／24卡训练、干预、恢复、结果复核代码与CPU测试 |
| [`manuscript`](https://github.com/Cheongroach/posttrain-causal-lab/tree/manuscript) | 逐步入门稿、研究手稿、A0实测报告、24卡冻结协议与分析方法 |

推荐不要反复切换分支。按照 [READING_GUIDE.md](READING_GUIDE.md) 用 `git worktree` 把三个分支同时签出到同一项目目录。

## 从哪里开始

第一次阅读：

1. 先完成 `manuscript` 分支的《一步步认识大模型后训练因果追踪项目》。
2. 再看 `code` 分支的 `contracts.py` 和测试。
3. 最后沿 `pilot.py` 阅读真实 A0 执行顺序。

研究结论与工程验收严格分开。A0 证明执行、干预、恢复和产物链可用，不证明 OPD 优于 SFT／KD，也不证明第一层 GDN 是数学能力机制。

## 公开边界

仓库公开核心源码、模型／数据 pin、精简机读证据和科学手稿。模型权重、内部调度配置、任务 URL、平台日志和 469 MB 原始运行归档没有上传。

License: Apache-2.0.
