# PostTrain Causal Lab

这个项目研究一个具体问题：后训练信号怎样经过 LoRA 参数更新和模型内部计算，最终改变输出行为；哪些观察属于执行证据，哪些还需要质量评测或训练干预。

当前完成的是 **A0 执行一致性实验**：

- Qwen3.5-4B student 与 Qwen3.5-9B teacher 的真实权重路径通过。
- 第一层 GDN identity 与原 donor restore 的 logits 最大绝对误差均为 0。
- adapter、optimizer、RNG 和数据游标恢复后，同一下一步的 adapter 最大绝对误差为 0。
- SFT、固定前缀 KD、学生前缀 OPD 各完成四步。
- 四个 OPD rollout 全部触及 32-token 上限；本轮没有模型质量或机制定位结论。

## 三个分支

| 分支 | 内容 |
|---|---|
| [`main`](https://github.com/Cheongroach/posttrain-causal-lab/tree/main) | 项目入口和多 worktree 阅读指引 |
| [`code`](https://github.com/Cheongroach/posttrain-causal-lab/tree/code) | 核心训练、干预、恢复代码与 CPU 测试 |
| [`manuscript`](https://github.com/Cheongroach/posttrain-causal-lab/tree/manuscript) | 逐步入门稿、研究手稿、A0 协议、实测报告与图 |

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
