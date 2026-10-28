# 框架边界与未完成工作

## 代码组织

模型、批次计算、指标函数与 MicroSS 数据适配器保持分层，便于后续替换标签、
特征和切分策略。当前仓库只保留已经跑通的 MicroSS 路径。

共享输入张量：node_features [B,T,N,F]、adjacency [B,T,N,N]。
服务数 N 与特征数 F 由适配器提供，模型里没有 MicroSS 服务名。
当前 GraphWindowDataset 是完整 CSV 面板的读取实现：标签列保留旧名称
fault_next_minute，二值节点目标取并集作为应用目标。它不是所有数据集的通用标签规则。
已有独立应用标签、缺失根因标签或事件分组的数据应实现独立 Dataset，
并明确提供 failure_target 和 node_targets 的含义，不能直接套用这个 CSV 读取器。

## 论文与当前实现核对（2026-10-24）

来源：[InstantOps ICPE 2024 原文](https://research.spec.org/icpe_proceedings/2024/proceedings/p119.pdf)。

- 算法 1 指定两层 GCN 加 GRU；默认模型已修正为两层。
- 论文构造逐分钟交互图；当前默认路径已改为按完成时间的逐分钟调用次数图。
- 当前节点监督头是本项目的选择；置零遮蔽函数已改名 masking_importance，
  不能把它称为已经复现论文的完整 permutation testing。
- 已补 MFR/MAR，与 Hit@K/MRR 一起计算；并列分数采用稳定服务顺序，排名从 1 开始。

## MicroSS 本轮进度

- 已按指标独立输出并添加观测标记，网络累计字节转换为速率，保留训练集拟合标准化。
- 已实现业务日志计数和严重级别特征，以及资源质量/服务覆盖报告。
- 已实现逐分钟、按子 span 完成时间聚合的权重图与 Dataset 时间对齐检查。
- 已接入 masking RCA，并导出逐窗口排名，保留 node-head 可选路径。
- 已增加 scripts/run_micross.py 全流程入口；服务器信息和命令见 server.md。

## 尚未完成

1. 源时间对齐：原始 trace/business/run 时间没有时区，默认 UTC 是待验证的假设。
   --source-timezone 可以指定源时区，epoch 指标时间保持原样，配置随数据保存；
   必须在真实数据中验证，不能从主机时区或文件名直接断定。
2. 应用失败标签：仍为注入开始代理标签。顶层请求识别尚未验证，不能据 span 的 5xx
   比例实现论文的应用失败规则。CPU 突发注入能否被提前预测也需要实验证据。
3. 当前保留辅助监督节点头以及对称调用次数图，masking 使用归一化空间置零（训练均值），
   按失败类概率下降排名。这些是实现选择，不声称完全复现论文的 permutation testing。
4. 独立加载推理、断点续训、事件级评价、时间外推稳定性验证。
5. 其他数据集与对比算法按当前计划暂缓。

## 验证范围

已通过合成数据回归检查，并在服务器完成全量 MicroSS 预处理、全量建图、
CPU 单 epoch 训练以及预测和 RCA 排名导出。
