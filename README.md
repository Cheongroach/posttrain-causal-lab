# PostTrain Causal Lab: code branch

本分支保存A0和24-H20配对实验的可移植核心实现。两轮均已完成执行验证；24卡独立复核显示训练截断减少，完整数字正确率的预算收益仍不确定。详见 [`manuscript`实测报告](https://github.com/Cheongroach/posttrain-causal-lab/blob/manuscript/experiments/scale24-20261027/REPORT.md)。

```text
posttrain_causal_lab/
├── contracts.py             # response mask、CE/KL、输出干预
├── pilot.py                 # 真实 Qwen3.5 A0 主路径
├── scale24.py               # 单cell：256步、生成评测、dev选址与verification干预
├── scale24_worker.py        # 一个8卡Worker上的四个独立2卡cell
├── scale_contracts.py       # 3配对seed布局、数字评分、匹配随机扰动
├── analyze_scale24.py       # 回收后复核、配对统计与科研图
├── assets.py                # 模型文件下载／复制与哈希校验
├── test_contracts.py        # 数学和状态契约
├── test_model_contract.py   # 微型原生 Qwen3.5 集成检查
├── test_scale24.py          # 布局、评分与干预契约
├── test_analysis.py         # 临时合成矩阵与分析失败注入
├── assets/models.json       # Qwen3.5-4B／9B revision、size、SHA256
├── data/manifest.json       # 冻结数据划分的来源和哈希
└── ops/freeze_assets.py     # 从官方 train split 重建冻结划分
```

内部调度器、队列、HDFS 路径和任务平台接口不属于公开可移植实现，因此未放入本分支。

## CPU 契约测试

建议使用 Python 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest \
  posttrain_causal_lab.test_contracts \
  posttrain_causal_lab.test_model_contract \
  posttrain_causal_lab.test_scale24 \
  posttrain_causal_lab.test_analysis \
  posttrain_causal_lab.test_case_rng \
  posttrain_causal_lab.test_length_analysis -v
```

共20项检查：前13项覆盖微型Qwen3.5与训练／干预契约，后7项使用临时合成归档验证分析器，包含固定词表大小缓存的回归检查。合成矩阵包含完整12个cell布局及已知配对差值，所有夹具位于系统临时目录；不下载4B／9B权重或GSM8K，也不证明真实GPU路径。

长度诊断另加9项检查：5项逐题RNG／前缀契约、4项长度分析（包含六cell完整归档的已知救回数），上述命令合计29项。GPU任务中只运行其冻结包的18项契约。

## 重建冻结数据

下载 `opencompass/gsm8k` 的固定 parquet 后：

```bash
python -m posttrain_causal_lab.ops.freeze_assets \
  --train-parquet /path/to/train-00000-of-00001.parquet \
  --model-manifest posttrain_causal_lab/assets/models.json \
  --wheels /path/to/pinned-wheels
```

脚本要求源文件 SHA256 与 `data/manifest.json` 中记录的固定版本一致，再按 seed `20261027` 生成互斥的 256／64／64 划分。公开仓库不重复分发 GSM8K 内容或 Python wheels。

## 24-H20矩阵

3个配对seed × SFT／KD／OPD-32／OPD-256 × 每cell两卡；沿用Qwen3.5-4B FP32学生与9B BF16教师，每cell256步。它扩大了训练与验证矩阵，没有采用24-rank DDP，也没有扩大模型参数量。

冻结配置在[`manuscript`分支协议](https://github.com/Cheongroach/posttrain-causal-lab/blob/manuscript/experiments/scale24-20261027/PROTOCOL.md)。准备符合模型manifest的本地模型文件、重建的三个数据split及该协议的`protocol.json`后，一个8卡Worker可运行：

```bash
python -m posttrain_causal_lab.scale24_worker \
  --worker-index 0 --models /path/to/models --data /path/to/data \
  --output /path/to/results/cells --protocol /path/to/protocol.json
```

Worker索引分别为0、1、2，各自使用相应seed。每个子进程独占两张H20，执行生成评测、训练状态重放、dev全层搜索和verification固定选址干预。主比较是OPD-256−32，统一256-token贪心评测；192次评测复用同一64题，不能按192道独立题统计。

`analyze_scale24.py`复核该冻结运行的源包、归档、checkpoint、raw-token评分、选址与配对状态，通过后生成`analysis.json`、`REPORT.md`和两组PNG／SVG。本轮已完成这些门禁及图表视觉和科学复核。脚本把固定词表大小缓存一次，避免逐token重复构造。本仓库发布精简机读结果与图表，没有分发原始大体积归档。

## 长度与随机性诊断

`length_probe.py`重评已有64／128／256步adapter，在原dev上独立运行256／512-token贪心生成。共同前缀与旧第256步输出必须逐token一致。`analyze_length_probe.py`重新解码、评分、检查归档和预算配对，分别统计原先截断后正确完成、错误完成和仍截断的题数。

`case_rng.py`提供`case_sampling(rollout_seed, example_id, epoch, sample_index, cuda_devices)`上下文。版本化SHA256不含条件名、长度预算或处理顺序，临时设置采样流后恢复CPU／CUDA／Python外层状态。它消除前题长度引起的后题随机数错位，不强制不同模型产生相同回答。CPU契约已通过；本次6卡诊断还会验证真实模型的CUDA短／长采样前缀与外层状态恢复。

该诊断已启动，优化更新0，不能作为逐题RNG提升训练稳定性或质量的证据。配置见[长度诊断协议](https://github.com/Cheongroach/posttrain-causal-lab/blob/manuscript/experiments/length-probe-20261028/PROTOCOL.md)。

## 阅读顺序

1. `contracts.py`
2. `test_contracts.py`
3. `test_model_contract.py`
4. `pilot.py`
5. `assets.py`

从`pilot.py`扩展到`scale24.py`，再看`test_analysis.py`和`analyze_scale24.py`。科学问题、逐步入门稿、A0结果和24卡协议在`manuscript`分支。
