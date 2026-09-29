# V8：六卡跨引擎 PD 已验收基线

正式名称：**ds4.1flash6卡（4spark+2张6000Dpro）V8版本**。命名日期：2026-09-20，Asia/Singapore。对应当天最后完成 DSpark 与计算参数恢复的状态，来源为 [最终复测报告](../hetero-v2/xyvllm-dspark-20260920/REPORT.md)。本次完成本地版本标记、补丁和证据冻结；没有新增线上性能测量。

## 与 V7 的显著差异

| 项目 | V7 | V8 | 实际意义 |
|---|---|---|---|
| P：输入预填充 | 双 RTX 6000Dpro，vLLM，21 层，TP2 | 双 RTX 6000Dpro，vLLM，21 层，TP2 | 6000D 继续负责读入长输入 |
| D：生成输出 | 四台 Spark，vLLM，40 层，TP4 | 四台 Spark，SGLang，40 层，TP4/EP4 | 核心变化是解码引擎及其计算、投机和调度实现 |
| KV 交接 | vLLM → vLLM，NixlConnector 兼容补丁 | vLLM → SGLang，跨引擎 NIXL 接收、页布局和状态适配 | 对齐不同缓存页，分离数据与 scale，检查地址及完成通知；Indexer 由 FP8 转为 MXFP4，主 KV 保持字节传输 |
| D 尾部补算参数 | XY_PD_TAIL=1280 | XY_PD_TAIL=256 | 搬运窗口状态并适配状态恢复后缩短补算；实际长度还受输入长度和边界对齐影响 |
| D 分块参数 | 每步 token 总预算 1536 | chunked_prefill_size=2048 | 参数语义不同，不能把数值比例当性能比例 |
| 投机与图执行 | DSpark K=5；CUDA Graph FULL_AND_PIECEWISE | DSpark block=5；目标验证和草稿解码 Graph 开启；prefill Graph 关闭 | 两版都有 DSpark 和 CUDA Graph，变化在实现及跨引擎适配 |
| 最终 D 计算设置 | vLLM 既有实现 | FP8 后端 flashinfer_cutlass；关闭共享专家融合 | 固定本次已验收的 SGLang 计算配置 |
| D 并发与调度 | 最大请求数 12；记录过 P 排队和补算干扰 | 最大运行请求 12；重叠调度开启，实测达到 12 路 | 上限没有从 8 升到 12；P 排队和接入补算仍存在，本次未另行改造调度策略 |
| 前缀缓存处理 | 已记录跨配置旧前缀空输出、乱码风险 | 跨引擎请求使用独立 cache salt | 避免复用缺少历史窗口状态的前缀，同时失去重复前缀命中的收益 |

两版都是 **P21 预填充 → D40 完整解码**。每个新输出 token 仍由四台 Spark 执行完整 40 层；不能理解成解码阶段分担 21+19 层，也不是此前六卡单实例 PP 实验。

来源：[V7 固定参数](../ds4.1flash6gpu-v7/README.md)、[V7 性能边界](../ds4.1flash6gpu-v7/watchdog/PERFORMANCE.md)、[V8 缓存适配](../hetero-v2/xyvllm-patch/README.md)、[最终生效参数](evidence/effective.json)、[最终健康与参数证据](evidence/final-health-rank0.json)。尾部 256 来自恢复 DSpark 前的四节点环境审计；最终增量只改变投机和计算设置，保留该尾部路径，未重新采集全量环境。

## 已验收性能与比较边界

| V8 负载 | 单请求 decode | 整批端到端聚合输出 |
|---|---:|---:|
| 黄金英文代码原题，200-token 预算，C1，三轮均值 | 72.833 tokens/s | 64.800 tokens/s |
| 同一原题，C8，三轮均值 | 每流均值 33.223 tokens/s | 231.833 tokens/s |
| 精确 8192-token 输入，256-token 输出，C1 | 47.844 tokens/s | 40.861 tokens/s |
| 同一 8192-token 负载，C12 | 每流中位数 13.281 tokens/s | 118.937 tokens/s |

单请求 decode 按首段内容到末段内容计时；端到端聚合为总输出 token / 整批墙钟时间，包含首字等待和排队。投机流式包可能包含多个 token，按 usage 计数。

**不能据此给出 V8 相对 V7 的加速倍数。** V7 的 8192/C8/512 聚合 decode 160.27 tokens/s，与上表负载、输出预算、计时窗口不同。黄金单流 81.433 来自独立 SGLang TP4 黄金记录，也不是 V7。16.748 → 47.844 是当前跨引擎方案恢复配置前后的同负载对照。

V8 尚未达到单流 80、聚合 300+ 目标。P 长输入排队、D 接入补算影响吞吐；D 满 8 路剖析的主要时间在计算及跨卡通信，剩余差距未完全分解。没有独立纯 prefill 速度数据，不能用输入 token / 首字等待代替。

## 分阶段验收

- 跨引擎搬运修复、DSpark 关闭阶段：7 项正确性检查、24 个场景、172 条请求，覆盖 8192–100000 输入和并发 1–16。见 [搬运阶段报告](evidence/transport-report.md)。
- **最终 V8**：4 项正确性检查、10 个矩阵场景共 55 条请求、黄金原题 27 条请求全部通过；最终矩阵最长输入 32768 tokens。另有 1+8 条剖析请求通过，不计入吞吐。见 [最终报告](evidence/final-report.md)。
- 最终健康证据记录 P/D/代理健康、P KV 占用 0、队列清空、四台补丁哈希一致。早期 100000-token 全矩阵不能冒充最终 DSpark 配置下的完整重测。

## 冻结内容与使用

[baseline.json](baseline.json) 保存正式版本、配置、来源与逐文件 SHA256。patch/ 为当前补丁副本，runtime.env 为最终增量环境；packages/ 同时保存跨引擎基础补丁包及最终计算增量。应用顺序为基础包 → 最终增量，并合并 manifest、应用 runtime.env。**单独用早期基础包会遗漏最终 DSpark/计算恢复。**

本目录是版本定义、补丁和证据快照；运行依赖既有 P21 vLLM、D40 SGLang 镜像、模型、Engram 字典及节点部署配置。没有重新制作角色镜像，也没有新机器冷恢复演练。V7 的完整镜像交付仍在 [V7 交付清单](../ds4.1flash6gpu-v7/DELIVERY.md)，不能直接充当 V8 的 SGLang 镜像。

项目根目录执行校验：

```powershell
python ds4.1flash6gpu-v8/verify_baseline.py
```

SHA256SUMS 包含本目录与根 V8 入口的校验值。复制的历史报告保留原文，其相对路径按原来源目录解释；本 README 的链接可直接使用。旧 ds4.1flash6gpu-v8-candidate 保留为历史候选，正式 V8 以本目录为准。
