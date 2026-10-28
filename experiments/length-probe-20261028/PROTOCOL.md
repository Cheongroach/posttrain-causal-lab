# 已有 OPD checkpoint 的长度敏感性诊断

24-H20实验已完成，但输出预算收益在三个seed间方向不一致。后续逐题检查发现：前两个seed的已结束回答多数数字正确，差异主要伴随结束率变化；第三个seed的OPD-32还存在较多已结束但错误的短数字回答。这里先直接改变评测预算，检验256-token上限与训练阶段对观察的影响。

## 固定范围

| 项目 | 设置 |
|---|---|
| checkpoint | 原实验OPD-32／OPD-256，seed20261027／28／29，各第64／128／256步 |
| 权重 | 原Qwen3.5-4B、FP32、相同LoRA模块；从已验证checkpoint逐值提取adapter |
| 数据 | 原dev 64题；不访问原verification和官方test |
| 生成 | 相同non-thinking prompt，独立运行256和512-token贪心生成 |
| EOS | 保持原模型配置248044；tokenizer chat end248046另行标记，不修改主要评分 |
| 总上下文 | 最多1024 token，显式检查prompt＋512 |
| 资源 | 3 Worker × 2 H20，每Worker负责一个seed的两个条件，共6卡 |
| 时限 | 每Worker入口7200秒，入口窗口合计最多12 GPU·h；实际按Pod分配时间另算 |
| 优化更新 | 0；没有重新训练或修改旧实验权重 |

评分继续使用冻结的最终数字parser，并分别报告数字正确、EOS完成正确、格式、截断和token数。每个checkpoint的512-token输出必须包含其256-token输出的完整前缀；第256步的256-token输出必须与原dev记录逐token一致，否则整组不进入结果汇总。总计18个checkpoint、36个评测设置、2304条生成记录。

## 诊断判据

先逐seed比较同checkpoint的预算变化，再比较同预算下的64／128／256步曲线。统计预算扩展后“原先截断→结束且正确”“原先截断→结束但错误”“仍截断”的题数。若延长后正确答案增加，说明预算影响了既定checkpoint的可观测表现；不能据此断言训练算法获得了新能力。

这些是已观察原结果后的诊断，不是新验收。原dev只用于定位和后续配方选择；三个seed共享64题，不能作为192道独立新题。任何训练修改需要新的冻结协议和未触碰的确认集。

## 逐题 RNG 修复

新采样入口使用稳定SHA256：

$$
s_i=H(\text{rollout seed},\text{example ID},\text{epoch},\text{sample index})\bmod 2^{63}.
$$

哈希还包含固定版本域；不包含条件名、生成预算或处理顺序。采样上下文临时设置CPU、指定CUDA设备和Python RNG，退出或异常时恢复外层状态。这样前一题消耗更多随机数不会使下一题错位；模型参数变化仍可使两条件产生不同token，不能把RNG配对理解为强制轨迹相同。

CPU检查覆盖题序、长短前缀、额外采样、嵌套和异常恢复。本次在各cell的真实第256步模型上，用同一dev题分别采样8／16 token，要求短序列为长序列前缀；中间插入无关随机采样后，仍需满足该门禁及外层状态恢复。该检查验证真实CUDA采样契约，不构成“逐题RNG改善质量”的训练证据。

实施入口：[case_rng.py](https://github.com/Cheongroach/posttrain-causal-lab/blob/code/posttrain_causal_lab/case_rng.py)、[length_probe.py](https://github.com/Cheongroach/posttrain-causal-lab/blob/code/posttrain_causal_lab/length_probe.py)。本轮结果返回后，再决定逐题RNG重训及学习率、batch的独立消融，避免同时修改多个变量。

## 执行状态

6-H20纯推理诊断已提交并启动，尚无该轮终态结论。18项Worker契约与4项分析检查通过，后者包含六cell完整合成归档复核。真实CUDA RNG探针与长度结论须等待终态产物核对。
