# 同一项目目录内阅读三个分支

仓库把入口、代码和手稿放在三个分支：

| 分支 | 内容 | 推荐用途 |
|---|---|---|
| `main` | 项目地图、分支说明、签出命令 | 第一次进入项目 |
| `code` | 核心训练、干预、恢复代码与测试 | 逐函数阅读或运行 CPU tests |
| `manuscript` | 入门稿、研究方案、执行协议、实验报告与图 | 连续阅读证据链 |

不要在同一个工作目录中反复 `git switch`。使用 `git worktree`，让三个分支同时出现在一个项目目录下。

## 第一次签出

```bash
git clone https://github.com/Cheongroach/posttrain-causal-lab.git
cd posttrain-causal-lab

git fetch origin main code manuscript
git branch code origin/code
git branch manuscript origin/manuscript

mkdir -p .branches
git worktree add .branches/code code
git worktree add .branches/manuscript manuscript
```

得到：

```text
posttrain-causal-lab/
├── README.md                    # main 分支
├── READING_GUIDE.md             # main 分支
└── .branches/
    ├── code/                    # code 分支
    │   └── posttrain_causal_lab/
    └── manuscript/              # manuscript 分支
        ├── 一步步认识大模型后训练因果追踪项目-friendly.md
        ├── RESEARCH.md
        ├── RUN.md
        └── experiments/a0-20261027-110309/REPORT.md
```

三个目录共享同一个 `.git` 对象库，不会重复保存完整提交历史；每个 worktree 可以独立打开、搜索和修改。

## 推荐阅读顺序

1. 当前目录的 `README.md`。
2. `.branches/manuscript/一步步认识大模型后训练因果追踪项目-friendly.md`。
3. `.branches/code/posttrain_causal_lab/contracts.py`。
4. `.branches/code/posttrain_causal_lab/test_contracts.py`。
5. `.branches/code/posttrain_causal_lab/pilot.py`。
6. `.branches/manuscript/RESEARCH.md`。
7. `.branches/manuscript/experiments/a0-20261027-110309/REPORT.md`。

## 日常更新

```bash
git pull --ff-only
git -C .branches/code pull --ff-only
git -C .branches/manuscript pull --ff-only
```

查看三个目录状态：

```bash
git worktree list
git status --short
git -C .branches/code status --short
git -C .branches/manuscript status --short
```

## 删除附加 worktree

确认没有未提交修改后：

```bash
git worktree remove .branches/code
git worktree remove .branches/manuscript
git branch -d code manuscript
```

这只删除本地签出，不删除 GitHub 上的远端分支。
