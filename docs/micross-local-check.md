# MicroSS 本地真实小样本检查（2026-10-24）

按用户要求，本机仅做有读取上限的真实数据检查，完整训练仍放服务器。
源数据：`/Users/95penguin/project/GAIA-DataSet/MicroSS`。

## 数据核对

- trace 7 月表包含 10 个服务；31 个 z 分卷加 zip，与目录声明的 32 卷相符。
- business 7 月只有 webservice1 表；45 个 z 分卷加 zip，与目录声明的 46 卷相符。
  其余服务没有业务日志输入，补零不能解释为服务健康或没有错误。
- 每个 trace 表先检查 2,000 行，共 20,000 行，必需字段无缺失。
- 业务日志前 10,000 行均可按 message 中的时间及级别解析。
- 目录检查和局部解压不等于全包 CRC/解压完整性校验。
- 源文本时区仍按 UTC 假设处理；服务器正式实验前仍需核实。

## 实际执行

从原始文件制作独立的小型 ZIP，未修改原始数据。限制每个 trace 表读取前
10,000 行，业务日志前 10,000 行；选定指标成员最多读取前 3,000 行。
再保留假定 UTC 下 2021-07-01 09:00 到 2021-07-02 00:00 的记录。
各表前缀不是同步抽样，因此同一时间段也可能缺少 trace/log，不能用于正式性能比较。

实际写入：151,200 条指标、100,000 条 trace、10,000 条 webservice1 日志、16 条注入记录。
调用 scripts/run_micross.py，history=5、batch_size=32、epochs=1、device=cpu。
预处理 → 动态图 → 训练 → 测试 → RCA 排名与 checkpoint 导出全部返回成功。

输出面板 843 分钟，838 个可用窗口，训练/验证/测试为 586/126/126；
正样本数为 11/3/2。测试集的两个正例均未预测出来，Precision/Recall/F1 为 0，
AP 约 0.0239。Accuracy 约 98.4% 只是类别不平衡下全负预测的结果，不能作为有效性证据。

截取样本的父 span 匹配率约 87.4%，843 个面板分钟中 276 分钟有边；
这是前缀截取后的结果，不代表全量数据的匹配率与图覆盖。
RCA 有 2 个正例窗口，Hit@1=0.5；样本太少且覆盖不完整，只确认导出可用。

## 结果文件

- `/private/tmp/micross-preflight.json`：原始文件目录和字段抽样检查。
- `/private/tmp/instantops-real-micross-smoke/sample_manifest.json`：抽样范围及各服务计数。
- `/private/tmp/instantops-real-micross-smoke/pipeline.log`：完整运行日志。
- `/private/tmp/instantops-real-micross-smoke/processed/`：特征、标签、图与质量报告。
- `/private/tmp/instantops-real-micross-smoke/results/`：checkpoint、指标、逐窗口排名。

临时目录可能被系统清理。全量运行与有效性验证仍待服务器完成。
