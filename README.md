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
  posttrain_causal_lab.test_analysis -v
```

共20项检查：前13项覆盖微型Qwen3.5与训练／干预契约，后7项使用临时合成归档验证分析器，包含固定词表大小缓存的回归检查。合成矩阵包含完整12个cell布局及已知配对差值，所有夹具位于系统临时目录；不下载4B／9B权重或GSM8K，也不证明真实GPU路径。

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

## 阅读顺序

1. `contracts.py`
2. `test_contracts.py`
3. `test_model_contract.py`
4. `pilot.py`
5. `assets.py`

从`pilot.py`扩展到`scale24.py`，再看`test_analysis.py`和`analyze_scale24.py`。科学问题、逐步入门稿、A0结果和24卡协议在`manuscript`分支。
