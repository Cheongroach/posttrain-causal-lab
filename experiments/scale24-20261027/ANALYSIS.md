# 24-H20 结果复核与统计口径

本方法在任务排队、尚无训练结果时补齐。Worker使用的16文件代码包与 `protocol.json` 保持冻结；本地分析入口 [analyze_scale24.py](https://github.com/Cheongroach/posttrain-causal-lab/blob/code/posttrain_causal_lab/analyze_scale24.py) 不改变训练配方。

先复核执行与测量，再计算结果。三个Worker必须全部归档、入口及四个子进程均退出0；缺失一个Worker或cell时，不汇总为成功实验。失败或超时保留已回收文件和平台终态，不自动重跑。

## 复核内容

| 对象 | 独立复核 |
|---|---|
| 代码与运行身份 | 原始代码包SHA256、16个成员哈希、Worker源文件、归档SHA256、trial、3×8卡布局、12个cell身份 |
| 数据与配对 | 256／64／64独立ID、冻结文件哈希；按seed重新生成train顺序；同seed四条件的初始adapter与CPU／CUDA／Python RNG一致 |
| 训练状态 | 256步顺序、loss／grad有限且grad norm非零；0／64／128／256游标及optimizer step、配置来源、checkpoint字节数与SHA256；张量有限；导出adapter等于step256 |
| 生成与评分 | 从固定tokenizer重新解码raw token；按模型EOS判定截断；重新提取最终数字、计算四个正确／格式指标及token长度，再与聚合文件核对 |
| 干预与搜索 | dev前8题×64模块×2处理＋8 identity＝1032条；verification64题×3位置×2处理＋64 identity＝448条；无缺失／重复；按dev记录重新选择top3 |
| 随机与identity | donor／random激活L2匹配，离线核对atol=1e-5、rtol=1e-5；log-prob下降等于trained−patched；identity激活L2为0、gold分数保持 |
| 恢复 | step128固定前缀重放，保存的loss绝对差≤1e-6；最大adapter误差有限；冻结运行时已逐元素检查atol=1e-7、rtol=1e-6 |

模型config的EOS为248044，tokenizer的`<|im_end|>`为248046。分析使用模型实际generation contract的EOS，与冻结运行路径一致；不能只根据文本为空或tokenizer EOS另改截断口径。

完整logits与两条重放adapter差张量没有归档，离线不能独立重算逐元素相对容差。报告会保留运行时断言的来源与保存的标量，而不把后者称为再次完成tensorwise门禁。主要评分是固定parser代理，未有人类推理复核。

## 主比较

对同seed的两个OPD条件，在相同64道verification题上计算：

$$
\Delta_s=\frac{1}{64}\sum_{i=1}^{64}
\left[c_{s,i}^{256}-c_{s,i}^{32}\right],
\qquad
\bar{\Delta}=\frac{1}{3}\sum_{s=1}^{3}\Delta_s.
$$

$c$ 为 `complete_numeric_correct`：最终数字正确且以EOS结束。图显示三个seed原始点和均值，表格保留每个cell。各条件共有192次评测，独立verification题目仍为64道。

均值的不确定性使用三个配对seed的样本标准差：

$$
\bar{\Delta}\pm t_{0.975,2}\frac{s_\Delta}{\sqrt{3}},
\qquad t_{0.975,2}=4.30265273.
$$

该区间依赖seed独立与均值分布近似，条件是固定64题；三个seed很少，区间不代表全体GSM8K题目不确定性。保留逐题跨seed配对差值，不把192条记录送入假设独立的检验。次要指标和四个条件的干预结果为描述性分析，不据多项比较挑出最优条件后宣称显著。

OPD-256−32是生成上限改变后的整体配方效果；路径、截断与监督token总量一起变化。固定前缀干预使用dev选定位置，报告verification上的donor log-prob drop及其与等L2随机扰动的差，单位nats/token；它不是自由生成正确率差，也不识别训练信号的历史原因。

## 成本与产物

成本按所有trial、所有Pod的schedule→stop记录累加，每Pod8卡。准备、训练、额外恢复诊断和归档均计入；排队不计入。缺失stop时间则标记成本不完整，不用cell耗时或72 GPU·h上限替代。无started_at、无Pod且终态的原trial支持“无已记录分配”的判断；不能仅凭stopped字符串认定零成本。

终态已完成独立复核，见 [REPORT.md](REPORT.md)、[analysis.json](analysis.json) 和 [audit.json](audit.json)。两张实测图分别记录预算比较与固定前缀干预，已实际渲染、打开检查，修正图例遮挡和点重叠；报告按科学证据、理解顺序、语言冗余完成三轮编辑。

手动复核需要原始归档、源包及来源索引。本仓库保存定稿报告与精简证据，原始大体积归档没有公开；入口为：

```bash
python -m posttrain_causal_lab.analyze_scale24 \
  --experiment /path/to/raw/scale24-20261027
```

本地7项分析检查使用临时合成数据，覆盖完整矩阵的已知配对效果，以及缺失Worker、raw token／评分不一致、重复干预、L2失配、checksum合法但含NaN的checkpoint、不完整成本记录和词表大小缓存。它们检验分析器的拒绝条件，不能代替实际训练结果。

回收后的两处修订属于本地分析：安全地逐文件解包，兼容未提供tar过滤参数的Python；把固定词表大小缓存一次，避免逐token重复构造。原来的Worker包、采样记录、评分规则、随机强度容差和配对统计保持不变，分析计划原文件仍保留。来源哈希与修订记录见 [reproducibility.json](reproducibility.json)。
