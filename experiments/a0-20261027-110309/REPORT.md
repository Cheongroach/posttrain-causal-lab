# A0：2-H20 执行一致性通过

Qwen3.5-4B student 与 Qwen3.5-9B teacher 已完成首轮工程 pilot。真实权重加载、LoRA 更新、第一层 GDN 输出干预和同进程训练状态重放均通过固定门禁。SFT、固定前缀 KD、学生前缀 OPD 各完成四步；这轮没有测量质量提升。

内部平台终态为 `done / finished`，Worker 退出码为 0。公开仓库不包含内部任务 URL、调度配置或原始平台日志。

## 本轮回答的问题

| 工程问题 | 直接观察 | 允许的结论 |
|---|---|---|
| GDN hook 会不会自行改变输出？ | Identity 与原 donor 恢复的 logits 最大绝对误差均为 0 | 此 prompt、位置和后端上的无效干预保持输出 |
| 置零是否实际改变计算？ | 第一层 GDN 输出置零，logits 最大绝对变化 20.3036 | 该干预生效；没有定位行为机制 |
| optimizer／RNG／游标恢复是否正确？ | 同一下一批次 loss 均为 2.928878784；adapter 最大绝对误差 0 | 同进程状态重放与连续执行一致 |
| 三种目标是否完成更新？ | 各四步 loss／梯度均有限、梯度 norm 非零、adapter 确实改变 | 三条执行链可运行 |
| adapter 关闭是否恢复底座？ | 三个分支关闭 adapter 后，初始 probe logits 最大绝对误差均为 0 | 本轮底座保持冻结，变化来自 adapter |
| 产物能否复核？ | 代码文件及五个 checkpoint 哈希匹配；adapter／optimizer 张量有限 | 本轮产物完整并通过独立核对 |

干预覆盖第一层 GDN 的全部 prompt／response 位置，只使用一个 probe。恢复测试在同一进程内重新加载 adapter、optimizer、CPU／CUDA／Python RNG 和数据游标；没有验证进程崩溃或跨机器接管。

## 设置与短程诊断

任务使用 1 Worker × 2 NVIDIA H20。GPU 0 运行 FP32 student，GPU 1 运行冻结的 BF16 teacher。LoRA r16／alpha32／dropout0，64 个输出投影 attachment sites，共 9,437,184 个可训练参数。软件为 PyTorch 2.9.1、Transformers 5.2.0、PEFT 0.18.1，使用原生 Torch GDN 和 SDPA。

三个分支都从相同 adapter、optimizer 和 RNG 状态出发，处理相同四个 prompt。KD 和 OPD 均使用完整词表 $\mathrm{KL}(p_{\mathrm{teacher}}\Vert p_{\mathrm{student}})$、温度 1、有效响应 token 平均；区别是固定标准答案前缀和学生当轮生成前缀。SFT 使用标准答案 CE。

| 分支 | 完整更新 | 有效响应 token 总数 | 最大 adapter 元素更新绝对值 |
|---|---:|---:|---:|
| SFT | 4 | 22 | 0.000401159 |
| KD | 4 | 22 | 0.000401159 |
| OPD | 4 | 128 | 0.000401147 |

四个 OPD rollout 全部触及 32-token 上限并被标记为截断。这里验证的是截断序列上的生成、教师反馈和更新链，不是四个完整数学回答。不同前缀及不同 token 预算下的 loss 不构成算法优劣比较。正式 A1 需要先冻结更合适的输出预算和质量评价，再做配对 seed。

本轮没有访问 GSM8K 官方 test、开发题或干预验证题。数据包只包含独立冻结 train 子集；没有质量评分、遗忘或跨 seed 机制结论。

## 资源与可追溯性

PyTorch peak allocated memory：student 19.41 GiB，teacher 16.76 GiB。这是框架分配统计，不包含所有驱动和外部库开销。

按最终 Pod 从 schedule 到 stop 的 392 秒计算，2 卡共约：

$$
2\times\frac{392}{3600}=0.218\ \mathrm{GPU\cdot h}
$$

此口径包含容器准备、模型复制和结果归档，排队与实验前资产准备另计。模型加载至 pilot 完成为 44.41 秒；单 seed 短程运行不作为吞吐 benchmark。

## 精简证据入口

| 证据 | 文件 |
|---|---|
| GDN identity／restore／zero | [intervention.json](evidence/intervention.json) |
| 连续执行与恢复重放 | [recovery.json](evidence/recovery.json) |
| SFT／KD／OPD 更新记录 | [short-arms.json](evidence/short-arms.json) |
| 完成标记、时长、显存 | [completed.json](evidence/completed.json) |
| 独立复核摘要 | [verification.json](evidence/verification.json) |

代码包 SHA256：`f3a48857a90f6b0a9c8ff4cea5d92ac8f06ff04bcbcb574602bee0ef1dbe8999`。

原始回收归档 SHA256：`4b0849d41b1cf5533c0fed633751372996fc17c9cf80397b17a52d744690e7b7`，大小 469,222,645 bytes。原始归档未公开；上述精简 JSON 由该归档复核得到。
