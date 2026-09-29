# vLLM TP2 → SGLang TP4 跨引擎 PD

适用于本项目既有 DSV4.1 P21/D40 镜像与尾部重算补丁。P21 计算共享压缩 KV 的来源层；D40 对齐接收缓存，再执行尾部重算并生成答案。

## 本次修复

- 2026-09-28：代理的本地 D 哨兵请求在 `DecodePreallocQueue.add` 进入独立的完整预填调度分支，不再经过 fake KV 接收、虚构 handoff 或尾部截断。复用引擎原生分块预填、KV 分配和多模态处理；中间块不进入解码，全部输入计算完成后才生成回答。真实 P→D 请求继续使用原有尾部重算与 digest 校验。
- 本地预填清除哨兵请求的 `skip_radix_cache_insert` 标记，使已计算的中间块正常提交；本地块和已就绪的 P→D 请求交替调度，并为正在预填的请求保留槽位，避免长图文阻塞文本准入。
- `func_timer.py` 让监控 Histogram 初始化保持幂等，修复工作节点 HTTP 生命周期再次初始化时的 `DuplicateTimeseries` 退出；映射已加入 `manifest.add`。真实依赖验收：同一进程调用三次初始化，保持同一个 Histogram 实例。
- 本地回归：`python .diagnostics/pd-full-prefill-fix-20260928/test_local_prefill.py`。内容验收脚本 `probe_full_prefill.py` 使用随机校验码与 512×256 图片，核查长文前中后信息、颜色、多轮和 SSE 完整性。HTTP 200 不作为内容正确的充分证据。
- 按完整的缓存组分别取块号；`kv_item_lens` 是每页字节数，地址不再重复乘页大小。
- 区分 P 端每块 64 个缓存条目与 D 端每页 128/256 个条目，分别搬运数据和缩放因子，跳过页末填充。
- P 端 FP8 indexer 经反量化后调用 D 端现有 MXFP4/RNE 存储内核；主 KV 保持逐字节传输。
- 处理 P 端 SWA 列表的尾部裁剪；传输前 21 层窗口状态。偶数边界重启 ratio-2 pending-pair 状态，并由既有尾部重算恢复其余 SWA。
- 校验每个地址边界、请求 token 数、缓存组与 engine ID。缺页、空搬运、布局不符均报错。
- 完成后按 vLLM `request_id:consumer_count` 协议通知两个 P rank；握手消息字段顺序为 `get_meta_msg, pp_rank, tp_rank`。
- 禁用会向 ZMQ 端口发送 HTTP 的旧 bootstrap 心跳。前门检查真实 HTTP 健康接口，每次请求检查 NIXL 状态。
- P、D 每条请求使用独立 cache salt，避免复用缺少历史 SWA 页的前缀缓存；健康探针不进入跨引擎尾部逻辑。

## 文件与部署

代理使用独立 HTTP 连接，避免复用已被 D 端关闭的空闲连接；不会自动重放可能已被引擎接受的推理 POST。验证脚本 `test_proxy_http.py` 通过真实 HTTP 服务检查该行为。

将 `manifest.add` 中的文件复制至四台 Spark 既有 overlay，并合并 manifest。`decode.py`、`prefill.py` 必须保留本项目既有尾部重算版本；本目录的 `utils.py` 与其配套。新增的 `layout.py`、`transfer.py` 也必须挂载。修改已导入的 Python 模块后需重启服务。

前门运行于 P 容器的 `/srv/xy_pd_proxy.py`，监听 5701。P 端只能使用带 `kv_transfer_params` 的 producer 请求。极短文本请用聊天接口；裸 completions 至少需要 8 个输入 token。`XY_VLLM_CONSUMERS` 默认 4，与当前 D TP4 一致。

本轮各节点的原 overlay 备份位于 `/var/tmp/dsv41-d40/xy178-before.tgz`。还原时先停止 D 容器，再恢复该备份与原 manifest，最后按原启动脚本启动。

## 验证

完整压测结果见 [REPORT.md](../xyvllm-results-20260920/REPORT.md)，逐请求记录与 CSV 在同一目录。可复用的压测和本地验证脚本位于 [xyvllm-validation](../xyvllm-validation/)。

本地验证与实机证据在项目 `.subagent-work/xy178-main/`：

- `test_layout.py`：实际搬运模拟字节，验证非零页、分散块、数据/scale 分区、裁剪窗口和越界拒绝。
- `test_tail.py`：健康探针、短输入拒绝、长短输入对齐及幂等。
- `test_benchmark.py`：真实 HTTP/SSE 桩验证 TTFT、token usage 和失败统计。
- `receiver-probe3.log`：实际接收器通过 648、8192、100000 token 请求的显存搬运、格式转换与完成通知。
- `benchmark.py`：先做中文问答与长文检索门禁，之后按实际 token 数运行流式压测，保存逐请求 JSONL 与聚合 JSON。
