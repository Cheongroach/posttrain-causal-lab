# 24-H20：OPD 输出预算与干预的定向验证

本轮从 A0 的直接观察出发：四个 OPD rollout 全部被 32-token 上限截断。问题是，扩到 256 token 是否减少截断、改变独立题目上的最终数字正确率；训练后的变化能否在底座 donor 替换下消失，且超过等 L2 强度随机扰动。

这是一次从 2 卡 pilot 到 24 卡配对矩阵的扩大。沿用已验证的 Qwen3.5-4B／9B 与 FP32 student；没有更新百 B 模型，也没有把 A0 当成已有机制结论。用户本轮明确要求一次扩大规模，替代了此前逐组提交小实验的执行安排。

## 冻结矩阵

| 条件 | 监督前缀 | 目标 | 训练 rollout 上限 |
|---|---|---|---:|
| SFT | 给定标准答案 | response CE | 不采样 |
| KD | 给定标准答案 | full-vocabulary KL | 不采样 |
| OPD-32 | 当前学生采样 | 同一 full-vocabulary KL | 32 |
| OPD-256 | 当前学生采样 | 同一 full-vocabulary KL | 256 |

每条件使用 seed 20261027、20261028、20261029，共 12 个 cell。每个 cell 独占 2 张 H20：GPU 0 为 FP32 student，GPU 1 为冻结 BF16 teacher。3 Worker × 8 卡在同一任务中提交，各 Worker 包含同一 seed 的四个条件；它们是独立配对训练，不是 24-rank DDP 或模型分片训练。

各 cell 使用完整 256 道 train 题，单样本一步，共 256 步；与 A0 的四步相比增加 64 倍。三个 seed 的样本顺序分别固定，同一 seed 的四条件共享初始 adapter、顺序、optimizer 和 RNG 契约。LoRA r16、alpha32、dropout0，attachment 为 `o_proj/out_proj/down_proj`；AdamW lr=1e-4，weight decay0，grad norm clip1。

KD／OPD 的目标为：

$$
\mathrm{KL}(p_T\Vert p_S)=\sum_v p_T(v)\log\frac{p_T(v)}{p_S(v)}
$$

温度 1，按有效 response token 平均；首个 EOS 参与监督，之后 padding 被排除。采样序列和 teacher 停止梯度。OPD-32／256 的唯一预设配方差别是生成上限；重新采样后路径和 token 总量也会变化，因此主对照衡量预算改变的完整配方效果，不声称隔离了等 token 的优化收益。

## 质量与数据边界

沿用 A0 固定数据：256 train、64 dev、64 verification，均来自官方 GSM8K train。官方 test 不访问。总共有 64 个独立 verification 题目，跨 seed 复用不能按 192 道独立题解释。

所有生成评测使用同一 non-thinking prompt、256-token 上限和贪心生成。预先指定 OPD-256 vs OPD-32 为主要配对比较。

| 指标 | 定义 |
|---|---|
| 主要指标 `complete_numeric_correct` | 归一化后的最终数字等于官方答案，且生成以 EOS 结束 |
| `numeric_correct` | 最终数字等于答案，单独保留截断状态 |
| `strict_correct` | 数字正确、有 `####` 标记、以 EOS 结束 |
| `format_valid` | 出现可解析的 `#### <number>` |
| `truncated` | 达到上限时仍未生成 EOS |
| gold-prefix NLL／teacher KL | 同一给定答案前缀上的诊断，不能代替生成质量 |

数字解析优先取最后一个 `####` 后数字，否则取文本中最后一个数字；去除千位逗号后用 Decimal 比较。它是固定 parser 的数值正确代理，不包含人类对推理内容的复核。报告将格式和数字正确分开。

## 计算干预与恢复

训练完成后，在 dev 的前 8 题上扫描 64 个模块输出：32 个 GDN／Gated Attention 输出和32个 FFN 输出。对相同 token 前缀，分别捕获关闭 adapter 的底座激活和训练后激活。逐位置替换为底座 donor，或加入与 donor 差值等 L2 norm 的固定 seed 随机扰动；每题附 identity 对照。

指标为标准答案 token 的平均 log-prob 下降，单位 nats/token。按底座 patch 的 dev 平均下降选择 top3，字典顺序解决同分；随后固定位置，在未参与搜索的 64 道 verification 题重算 donor、随机和 identity。该结果是固定前缀上的计算介导证据，不自动证明自由生成质量被同一模块介导，也不证明训练信号的历史原因。

checkpoint 在第 64、128、256 步保存。第 128 步从同一状态对一个固定下一前缀比较连续执行和恢复重放，检查 adapter atol=1e-7、rtol=1e-6，loss 绝对差≤1e-6。诊断更新完成后回到第 128 步，再按正常策略继续训练，避免多计算的一步进入正式轨迹。恢复仍是同进程重放。

## 执行状态

24-H20矩阵已提交排队，尚无本轮训练、质量、干预或运行时间结果。原提交在分配前停止，现沿用冻结配方恢复排队。任务与基础设施信息不公开。

本地13项CPU契约门禁和6项合成数据分析检查通过。终态结果必须收齐三份Worker归档并通过12个cell的复核；分析口径见 [ANALYSIS.md](ANALYSIS.md)。
