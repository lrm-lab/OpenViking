# P1 第一段：A1+B1 确定性验证（2026-10-26）

完成时间：2026-10-26T09:32:41.118220+00:00。本报告仅覆盖纯解析和身份映射；没有模型调用、提案重建、AO 重建或 D 账号写入。卡产量与质量没有重新测量，不能由本结果推断。

## 版本与隔离

- coforge 基线 `0fad029f5`；06:03 diagnose 写门控独立提交 `9f3c38ed3`，仅指定四文件；A1 提交 `a798a7088`。
- OpenViking-main 基线 `a7bcb1781`；B1 与本报告同一逻辑提交，最终 hash 见 `git log -1` 与进度日志。
- `COFORGE_EVAL_TASK_TRACE_FROM_FILE`、`OPENVIKING_CARD_READ_REF_ZERO_BASED` 均默认关闭。A1 的 runner 挂点额外要求 `domain=software_engineering`，KW/IR 不进入文件导出路径。
- 没有 push，没有启动、重启、重载任何服务，没有执行容器变更，没有读取 `~/.pi/agent/models.json`，没有 OpenRouter/grok 请求。
- 无任何请求发往在线 OV、Web、验证器；测试的 HTTP 写入使用内存 fake runtime，HTTP proxy 测试仅用临时随机端口。
- 源码修改未执行热加载。只读检查时八个保护端口均仍监听；8792/8745/8746/8747 的宿主进程启动时间分别为 10-24 12:59:06、10-24 04:51:59、10-24 13:26:50、10-25 02:54:15 UTC，早于本次任务，启动参数均无 `--watch/--hot/--reload`。没有取得四个 OV 监听器的进程身份；不声称完成了容器镜像前后比对。
- 原有 `agent-message-attention-index.ts` 和其测试的两处工作区修改未加入任何提交，其余原有未跟踪产物保留。

## A1：87 个训练会话

输入为 `results/train/*/*attempts.jsonl` 的全部 87 个 `work_dir`；逐个从所属 agent 的 `.builtin-sessions` 按 header session ID 选择，核对 `docker-task-runtime.json` 的 `hostWorkspace/sessionDirectory/cwd` 与文件头。未做全局搜索或“第一个 jsonl”回退。会话 SHA256 在解析前后完全一致；87 个文件的来源 hash 记录于 coforge 的分片 shape fixture。

| 指标 | 修复前基线 | A1 开关开启 |
|---|---:|---:|
| cwd 严格比较失败 / 导出错误 | 87/87 | **0/87** |
| 有 Task 工具证据的训练会话 | 0/87 | **87/87** |
| 原账本 Task 工具行 / 可导出工具调用 | 0（1085 行账本） | **4613** |
| 成功频道发送边界 | 原路径未导出 | **133** |
| 原始 assistant toolCall 核对 | 4746 | **4613 + 133 = 4746** |
| 未发送尾部工具 | 原逻辑丢弃 | **1575，涉及 78 会话** |

4613 次工具中，3038 次位于成功发送之前，1575 次位于未发布尾部；133 次成功发送是分组边界，不重复作为工具观测。以上为可入库的解析覆盖，**没有向真实账本写入这些行**。

本轮另修正文件导出的压缩遗漏：8 个真实会话有 compaction，使用模型可见上下文会漏掉 360 次调用；现在读取完整活动 parent 分支，保留压缩前消息。旧路径保持原上下文语义。

尾部以独立 runtime observation 写入，来源 ID 为 `runtime-observation:<session>:<exchanges-hash>`，正文明确 `not published to channel`，不伪造 channel messageId 或成功发送收据。稳定来源 ID 使正常 append 和重试去重均能保留观测。无 toolResult 的调用仅记录“invoked / completion unknown”的 ID、工具和输入，不写成 completed 工具行。身份错配、歧义、非法会话 ID、损坏 JSON、旧会话格式均显式失败；拒绝旧格式以避免 SDK 自动迁移写文件。

真实样本有 75 个 `final_settled`、12 个 `quiet_no_offer`；11 个无 TASK_COMPLETE。真实 87 会话脱敏 shape fixture 保留全部调用顺序及成功发送边界；另有无 TASK_COMPLETE、held、quiet 空 stop、deadline 未返回工具、压缩前工具、正常发送后残余工具的回归。历史训练中没有 deadline 终止样本，deadline 使用合成边界 fixture，不冒充历史样本。

## B1：读索引 → snapshot AO 身份 → block 归属

实现先逐个验证 raw index 为严格整数且 `0 <= index < len(snapshot.records)`，再取得该位置的 `ao_id`，最后与 revision block 的 AO 身份求交。拒绝负数、bool、float、字符串、None 与越界；不通过 `sequence + 1` 推断位置。非法输入在注册 revision 或调用 proposer 之前失败。

| 指标 | SWE 87 份裁决 | IR 6 份反例 |
|---|---:|---:|
| 可核验的 raw-read 下界索引 | **105** | **6** |
| 索引合法率 | **105/105 = 100%** | **6/6 = 100%** |
| legacy 映射为空（同一下界输入） | 79/87 | 6/6 |
| 修复后映射为空 | **0/87** | **0/6** |
| 消除映射错位的裁决数 | **87/87** | **6/6** |
| 已确认下界映射的 AO 数 | **105**（legacy 18） | **6** |
| 穷举单个 raw index 的身份检查 | **1085** | **157** |
| 修复后未读 AO 被登记为已读 | **0** | **0** |

“未读登记为已读=0”还由穷举位置检查和显式前一行反例保护：raw index 1 只登记实际位置 1 的 AO，不能登记 sequence=1 的前一行；另测 sequence 稀疏、重排、重复索引及不属于 block 的实际读取行。未属于 block 的已读行不登记为卡证据。

SWE 原裁决为 54 empty / 32 rejected / 1 proposed；其中原 `card_trace.read_aos=0` 为 76/87，32 个 rejected 均含 `ao_not_read`。IR 六份拒绝裁决均 `read_aos=0`，**其中五份错误码为 `ao_not_read`，另一份 `adj-165ad84cc4701f1dbc062a79` 为 `ao_not_in_block`**。fixture 保留真实错误码，不将其改写为六份 ao_not_read。

重要边界：磁盘 `reader-report.json` 保存的是 `decisive_ao_refs`，完整 `ReaderNavigationResult.reads/read_ao_refs` 未持久化。guarded Reader 校验 decisive refs 属于实际 raw reads，因此本报告用其作**可证明已读下界**。没有把下界声称为完整读历史；真实原裁决 76 个读数为零与下界 legacy 79 个为空不矛盾（有 3 份包含额外实际读取）。历史完整读取覆盖率和真实旧版“未读误登记”发生次数无法仅靠这些文件恢复；本次没有虚构数字。穷举及回归证明的是新映射函数对所有合法位置均不引入未读身份。

## 默认关闭与回归

- A1：从基线 `0fad029f5` 提取原 `channelTraceFromSteps`，对 87 份真实 shape fixture 比较默认关闭时的 JSON 字节：**87/87 一致，差异 0**。未改 KW/IR prompt 或协议；runner 的 SWE domain 条件保证 KW/IR 不走新路径。
- B1：从基线 `a7bcb1781` 提取原 `read_ao_ids`，扫描 IR D 的全部 **229** 会话，比较 **479** 个持久 revision 的 flag-off 输出字节：**差异 0**。真实 SWE/IR fixture 还逐项测试 unset 和 `0` 两种关闭形态。
- coforge：27 项相关隔离测试通过；benchmark TypeScript 检查通过；独立 A1 审查的 append 过滤、compaction 漏失、旧格式迁移写入与 pending 完成状态问题均已修正复核。
- OpenViking：读身份、proposer、Reader 共 **133 项测试通过**，12 条既有 Pydantic 弃用告警。
- 较宽的 agent package typecheck 在未修改的 `test/openviking-memory-tools.test.ts:569` 报 `MemoryAgentTurnBudget` 参数类型不匹配；没有为此改动其他文件或放宽检查。未运行全仓构建或线上 E2E。

## 复算入口与产物

从 coforge 目录运行（全部本地、无模型）：

```sh
mise exec -- bun scripts/test/isolated-test-run.ts bun benchmark/evolbench-collab/scripts/p1-phase1.ts
mise exec -- bun scripts/test/isolated-test-run.ts bun benchmark/evolbench-collab/scripts/p1-off-baseline.ts
```

从 OpenViking-main 目录运行：

```sh
python scripts/p1_card_read_refs.py
python -m pytest -o addopts='' -q tests/test_card_read_ref_zero_based.py tests/test_card_proposer.py tests/test_aig_toolloop_reader.py
```

脚本只写分析输出与仓库中的脱敏 fixture；原会话、裁决与 D 存储只读。逐会话明细位于本报告同目录，B1 fixture 含原裁决路径与 SHA256，A1 fixture 含会话 SHA256。本报告同文副本随 B1 提交保存于 `OpenViking-main/docs/validation/`。

| 分析产物 | SHA256 |
|---|---|
| `P1-A1-parsed.json` | `d03218b6c01253319981744e8918c24a86de490b03fc952ad8b11b3009e9092e` |
| `P1-A1-off-baseline.json` | `f18f28bf876a915d5823ac2dcdbd45fa1d030e7bc450cc53f877ac7d4afa726f` |
| `P1-B1-parsed.json` | `a6428f0d670066a66e67cfb33dbd0c62065d11dbf1acd79ff3ca3aada5abcae9` |
| `P1-IR-off-baseline.json` | `a8283350fa1062aa12736f5a0a6d883f0e14a89e66dedc5a7ffe361fab4284b2` |

P1 第一段确定性门槛通过；模型提案重建属于另行安排的第二段，本次未执行。
