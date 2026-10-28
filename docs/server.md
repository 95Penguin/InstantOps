# MicroSS 服务器运行

服务器连接信息保存在本机 `config/server.local.env`（Git 忽略），公开文档使用
`<user>@<host>` 占位。已连接服务器核对资源；实际运行以日志与退出状态为准。

原始数据统一放在项目 `data/MicroSS/`，包含 run、metric、trace、business 子目录。
该目录被 Git 忽略，需要单独同步到服务器。代码与数据传到服务器后，在项目根目录执行。
需要 uv 和可在 PATH 中找到的
7zz 或 7z；trace/business 的所有分卷必须放在各自 zip 旁边。
metric_full.zip 必须是可被 Python zipfile 完整读取的合并包。

```bash
uv sync --locked
uv run python scripts/run_micross.py --epochs 20 --device cuda --source-timezone UTC
```

默认输入是项目 data/MicroSS，临时索引在 data/tmp/micross。
如果数据位于其他磁盘，仍可用位置参数指定数据目录，并通过 --work-dir 指定临时目录。
没有确认服务器有可用 CUDA，
可将 `--device cuda` 改为 `--device auto`。服务器的 Python 环境由 uv 重建，
不上传本地 .venv。数据应包含 run、metric、trace 和 business。

默认输出在 data/processed/micross_v2 与 artifacts/micross_v2，避免复用旧特征。
预处理、建图成功后重复训练可以使用 `--skip-prepare --skip-graph`。
只有相同原始数据、history 和 v2 输出时才可复用，改变 history 时重新建图。

运行后检查：quality_report.json（指标常量/无效值/服务覆盖）、graph_quality.json
（跨服务匹配和有边分钟数）、metrics.json（各分区正样本数）、predictions.jsonl
（逐分钟故障概率及服务排名）、best_model.pt（权重与变换参数）。
没有正例的验证/测试段不具备完整的预测或 RCA 评价意义。

当前仍使用注入开始时间作为下一分钟标签。业务日志只提供输入特征，run 日志仅提供
监督标签。没有将全部 span 的错误率当作应用 SLA 失败标签。
动态图在每个输入分钟使用已完成子 span 的调用次数，双方调用次数相加后构成
对称权重图；未来分钟的边不会加入过去的图。该构图方式是明确的实现选择。

现阶段已完成合成数据检查，并在服务器完成真实数据全量预处理、建图、
CPU 单 epoch 训练及预测/RCA 导出。

原始文本时间未标时区；UTC 当前仅是默认假设。正式训练前需核对指标 epoch 与
日志事件的时区。如果证据表明文本是北京时间，使用 --source-timezone Asia/Shanghai
重做预处理和图；不要仅因服务器位于中国就改时区。该配置会写入 feature_schema.json。

## 首轮完整数据 CPU 验证

服务器检查到约 31 GiB 总内存、29 GiB 可用内存、133 GiB 可用磁盘及 RTX 3090。
当前已有 PyTorch 是 CUDA 13 构建，与现有驱动不兼容，首轮明确使用 CPU。
GPU 长训练前应在独立环境安装兼容驱动的官方 PyTorch 构建，不直接升级系统驱动。

`scripts/run_micross_server.sh` 为原始分卷数据补齐 7-Zip、解压 July run CSV、
合并 metric ZIP，然后执行启动检查、全量预处理、建图和一个 epoch 训练。
原始分卷保留。源时区 UTC 仍是待核实假设，标签仍为注入开始代理。
使用已有 Python 包创建 uv system-site-packages 环境，命令带 --no-sync，
避免首轮下载大体积依赖或重装不兼容 CUDA 构建；这不是锁文件完全重建环境。

```bash
RUN_ID=micross_full_smoke_01
mkdir -p "artifacts/$RUN_ID"
nohup bash scripts/run_micross_server.sh "$RUN_ID" \
  > "artifacts/$RUN_ID/run.log" 2>&1 < /dev/null &
echo $! > "artifacts/$RUN_ID/pid"
tail -n 30 "artifacts/$RUN_ID/run.log"
```

每次使用新的 RUN_ID，避免覆盖已有产物。exit_status 出现且为 0 才表示整条流程成功。
run.log 含环境版本和 /usr/bin/time -v 的内存峰值；尚未结束时不能报告已完成训练。
