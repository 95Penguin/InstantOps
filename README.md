# InstantOps reproduction

在本地 MicroSS/GAIA 7 月数据上，用过去 5 分钟预测下一分钟业务故障状态，
包含持续故障，并导出根因服务排名。采用两层 GCN＋GRU；这是框架级复现，
标签阈值和实现细节不声称与论文完全一致。

## 环境与入口

```bash
uv sync
.venv/bin/python scripts/run_micross.py --help
```

统一入口支持 `prepare`、`graph`、`label`、`train`、`run`；
不写子命令等同于 `run`。每个子命令支持 `--help`。
原始数据默认放在 `data/MicroSS`，原始分卷 ZIP 需齐全，解压读取依赖 7-Zip。

## 本地复用已有数据训练

```bash
.venv/bin/python scripts/run_micross.py train \
  --data-dir data/processed/micross_mps_smoke_01 \
  --graph data/processed/micross_mps_smoke_01/service_adjacency_1m.npz \
  --labels data/processed/micross_mps_smoke_01/entry_failure_v2.csv \
  --output-dir artifacts/entry_failure_mps_v2 \
  --epochs 5 --device mps
```

新输出目录应与旧实验区分。业务标签生成方法见 [标签与评估](docs/labels.md)。
必须显式传入 `--labels`：不传时仍使用历史注入开始代理标签。

首次处理原始数据可逐步运行：

```bash
.venv/bin/python scripts/run_micross.py --check-inputs-only
.venv/bin/python scripts/run_micross.py prepare
.venv/bin/python scripts/run_micross.py graph --training-data-dir data/processed/micross_v2
```

全流程 `run` 支持 `--skip-prepare --skip-graph` 复用中间产物。
原始数据、中间产物、训练权重均不纳入 Git。

## 代码组织

```text
scripts/run_micross.py    统一命令与流程编排
src/data/config.py       数据路径、字段配置
src/data/prepare.py      指标、日志、调用链预处理
src/data/preflight.py    原始分卷检查，供预处理和建图共同使用
src/data/graph.py        分钟级调用图
src/data/labels.py       业务故障事件与根因代理标签
src/data/dataset.py      面板读取、滑动窗口、MicroSS 适配与时间划分
src/models/             GCN＋GRU
src/training/           训练、标准化和评估
src/rca/                遮蔽归因与排名
tests/                  单元与合成数据端到端测试
```

## 文档与验证

- [标签、评估和实验边界](docs/labels.md)
- [数据预处理约定](docs/preprocessing.md)
- [已完成的 v1 实验结果](docs/entry-failure-v1-results.md)

```bash
.venv/bin/python -m unittest discover -s tests -v
```

产物包括 `best_model.pt`、`metrics.json`、`predictions.jsonl`。
故障预测报告 Precision/Recall/F1/AP；RCA 报告 Hit@K/MRR/MFR/MAR，
区分窗口与事后事件聚合结果，并报告可用根因标签数量。
