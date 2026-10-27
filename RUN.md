# A0 执行协议

本轮回答一个工程问题：Qwen3.5-4B 的 LoRA 更新、GDN 输出干预和训练状态恢复，能否在真实权重上通过固定门禁。Qwen3.5-9B 提供蒸馏分布。四步诊断不能回答数学能力或后训练机制是否改善。

内部运行编号为 `a0-20261027-110309`。公开仓库保留协议、核心代码、精简证据与结果，不公开调度平台信息。

## 固定配置

| 项目 | 本轮设置 |
|---|---|
| 分配 | 1 Worker × 2 NVIDIA H20 |
| GPU 0 | Qwen3.5-4B，FP32 student，LoRA r16／alpha32／dropout0 |
| GPU 1 | Qwen3.5-9B，BF16 teacher，只推理 |
| LoRA 模块 | `o_proj`、`out_proj`、`down_proj`；真实路径由模型枚举后保存 |
| 后端 | PyTorch 2.9.1、Transformers 5.2.0；Torch GDN，SDPA full attention |
| 优化器 | AdamW，lr 1e-4，weight decay 0，梯度 norm clip 1 |
| 数据 | GSM8K train 中独立冻结 256／64／64；A0 只访问 train |
| 输出协议 | non-thinking，`#### <number>`；最多 256 输入＋响应 token，OPD 最多生成 32 token |
| 随机性 | seed 20261027；所有分支恢复同一 adapter／optimizer／RNG 状态 |
| 执行窗口 | 容器入口开始计时，最多 3600 秒；排队时间另计 |

模型文件按公开 manifest 的 revision、size 和 SHA256 校验后，只加载语言参数。视觉和 MTP 不参与文本 A0。学生和教师核对完整 token→ID 映射、特殊 token、输出词表及实际渲染的 prompt。

固定前缀 KD 与学生前缀 OPD 均优化完整词表的

$$
\mathrm{KL}(p_{\mathrm{teacher}}\Vert p_{\mathrm{student}})
$$

温度为 1，按有效响应 token 平均；教师及当轮生成序列停止梯度。SFT 使用相同响应边界的 CE。首个 EOS 被监督，后续 padding 被排除。生成触及预算时记录 `truncated`，不按正常完成计入质量。

## 执行与验收

1. 校验代码包、依赖和模型的 SHA256。
2. 重跑 CPU 契约检查，再加载 student 与 teacher；语言权重的 missing／unexpected／mismatched keys 必须为空。
3. 在第一层 GDN 输出上做 identity、置零和原 donor 恢复。Identity／恢复 logits 的容差为 atol=1e-6、rtol=1e-6；置零必须产生大于 1e-7 的最大 logits 差。
4. 做一次完整 SFT 更新，保存 adapter、optimizer、CPU／CUDA／Python RNG、配置来源与游标。对同一下一批次比较连续执行和恢复重放：adapter atol=1e-7、rtol=1e-6，loss 绝对差不超过 1e-6。
5. 分别恢复共同初始状态，运行 SFT／KD／OPD 各四步。每步 loss／梯度必须有限、梯度 norm 非零；更新后 adapter 必须改变，关闭 adapter 必须恢复初始 logits。
6. 成功、失败或超时都回收代码与结果；产物回收失败不能标记为完整成功。

恢复门禁测试的是同进程中的训练状态重放。它没有故障注入，也没有验证跨 Worker、跨机器的接管。第一层 GDN 置零是干预执行检查，没有做层级搜索、匹配随机对照或独立行为验证。

## 预启动发现

Transformers 5.2 的 `apply_chat_template(..., tokenize=True)` 默认返回 `BatchEncoding`。如果把其 `len()` 当成 token 数，会得到字段数而不是序列长度。

实现因此显式设置 `return_dict=False`，并要求结果是平坦整数 token 列表。修复后首批 prompt 长度为 74～117 token，256 道训练题均满足 A0 预算。

## 公开证据

| 问题 | 文件 |
|---|---|
| 干预门禁 | `experiments/a0-20261027-110309/evidence/intervention.json` |
| checkpoint 重放 | `experiments/a0-20261027-110309/evidence/recovery.json` |
| 三个四步分支 | `experiments/a0-20261027-110309/evidence/short-arms.json` |
| 完成标记与显存 | `experiments/a0-20261027-110309/evidence/completed.json` |
| 独立复核摘要 | `experiments/a0-20261027-110309/evidence/verification.json` |

只有收齐终态产物并核对上述门禁后，才能给出 A0 通过结论。
