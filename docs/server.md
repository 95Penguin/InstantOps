# MicroSS 服务器运行

服务器连接信息保存在本机 `config/server.local.env`（Git 忽略），公开文档使用
`<user>@<host>` 占位。尚未连接、检查系统或执行任务。

代码与数据传到服务器后，在项目根目录执行。需要 uv 和可在 PATH 中找到的
7zz 或 7z；trace/business 的所有分卷必须放在各自 zip 旁边。
metric_full.zip 必须是可被 Python zipfile 完整读取的合并包。

```bash
uv sync --locked
uv run python scripts/run_micross.py /实际路径/GAIA-DataSet/MicroSS \
  --work-dir /实际路径/临时索引目录 \
  --epochs 20 --device cuda --source-timezone UTC
```

路径是占位示例，需按服务器实际存放位置修改。没有确认服务器有可用 CUDA，
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

现阶段已完成合成数据检查及本地真实前缀小样本的全流程运行，记录见
micross-local-check.md；真实数据全量运行及资源需求以服务器实际结果为准。

原始文本时间未标时区；UTC 当前仅是默认假设。正式训练前需核对指标 epoch 与
日志事件的时区。如果证据表明文本是北京时间，使用 --source-timezone Asia/Shanghai
重做预处理和图；不要仅因服务器位于中国就改时区。该配置会写入 feature_schema.json。
