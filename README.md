# PostTrain Causal Lab: code branch

本分支保存 A0 的可移植核心实现：

```text
posttrain_causal_lab/
├── contracts.py             # response mask、CE/KL、输出干预
├── pilot.py                 # 真实 Qwen3.5 A0 主路径
├── assets.py                # 模型文件下载／复制与哈希校验
├── test_contracts.py        # 数学和状态契约
├── test_model_contract.py   # 微型原生 Qwen3.5 集成检查
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
  posttrain_causal_lab.test_model_contract -v
```

测试使用随机初始化的微型 Qwen3.5，不下载 4B／9B 权重，也不证明真实 GPU 路径。

## 重建冻结数据

下载 `opencompass/gsm8k` 的固定 parquet 后：

```bash
python -m posttrain_causal_lab.ops.freeze_assets \
  --train-parquet /path/to/train-00000-of-00001.parquet \
  --model-manifest posttrain_causal_lab/assets/models.json \
  --wheels /path/to/pinned-wheels
```

脚本要求源文件 SHA256 与 `data/manifest.json` 中记录的固定版本一致，再按 seed `20261027` 生成互斥的 256／64／64 划分。公开仓库不重复分发 GSM8K 内容或 Python wheels。

## 阅读顺序

1. `contracts.py`
2. `test_contracts.py`
3. `test_model_contract.py`
4. `pilot.py`
5. `assets.py`

科学问题、逐步入门稿和 A0 结果在 `manuscript` 分支。
