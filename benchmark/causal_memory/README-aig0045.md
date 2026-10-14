# ADR-0045 Wave E2：四域 verdict 冒烟

评测直连：把 sidecar/attempt 的 `{reward, graded, error, detail}` 归一成
`OutcomeSignal`，写入 `AoSnapshot.outcome`，再按
`select_attribution_mode` 决定是否跑四角色。无信号不触发。

对接的 E1 符号（本 worktree 实名）：

| 模块 | 符号 |
| --- | --- |
| `openviking/session/outcome_signal.py` | `normalize_verdict`, `select_attribution_mode`, `OutcomeSignal` |
| `openviking/session/aig_tool_surface.py` | `AoSnapshot.outcome`, `TOOL_CATALOG` |
| `openviking/session/aig_toolloop_builder.py` | `run_toolloop`, `ToolloopProvider`, `FakeToolloopProvider` |
| `openviking/session/aig_toolloop_orchestrator.py` | `run_critic_refiner` |
| `openviking/session/aig_toolloop_reader.py` | `run_reader`, `DeterministicReader` |

## 密钥（人工导出，不要写入仓库）

真模型只在人工冒烟时调用。适配器从环境变量 `LENOVO_GATEWAY_API_KEY` 读取，
不硬编码、不打印。

密钥在本机 `~/.pi/agent/models.json` 的 `providers.lenovo_gateway.apiKey`。
不要把该文件贴进报告。自行导出示例：

```bash
python3 -c 'import json,pathlib; p=pathlib.Path.home()/".pi/agent/models.json"; print("set the env from providers.lenovo_gateway.apiKey; do not print it")'
export LENOVO_GATEWAY_API_KEY='<paste locally, never commit>'
```

`lenovo_gateway.baseUrl` 是 `https://modelfactory.lenovo.com/llm/v1`。
本适配器打的是 ADR 指定的
`https://modelfactory.lenovo.com/llm/v1/chat/completions`，模型
`gemma-4-31B-it`（该条目的 `models` 列表里目前只有 DeepSeek / GLM-5.3-Flash，
gemma 名以 ADR 为准，不从 json 猜）。

## 运行

单测必须零 LLM（仓库约定）：

```bash
uv run --no-sync pytest -o addopts= tests/test_lenovo_gemma_provider.py tests/test_aig0045_verdict_smoke.py tests/test_outcome_signal.py tests/test_aig_outcome_injection.py
```

合成素材自检（Fake builder，不碰网）：

```bash
uv run --no-sync python benchmark/causal_memory/aig0045_verdict_smoke.py --material synthetic --builder fake
```

无 key 干跑（只做 normalize / 模式选择 / 快照构造）：

```bash
uv run --no-sync python benchmark/causal_memory/aig0045_verdict_smoke.py --material synthetic --dry-run
```

真模型冒烟（人工；builder=gemma，critic/refiner 空违规直冻，reader=DeterministicReader）：

```bash
export LENOVO_GATEWAY_API_KEY='<local>'
uv run --no-sync python benchmark/causal_memory/aig0045_verdict_smoke.py \
  --material synthetic --builder gemma \
  --max-calls 64 --max-seconds 1800
```

真实素材：

```bash
uv run --no-sync python benchmark/causal_memory/aig0045_verdict_smoke.py \
  --material real \
  --verdicts path/to/verdicts.jsonl \
  --sessions benchmark/causal_memory/out \
  --dry-run
```

`--sessions` 可以是含 `*.jsonl` 的目录，或单个 `ao_records.jsonl`。
verdict 行：`{run_id, domain_kind, reward, graded, error, detail}`，
可选 `session_id`。与 AO 行的 `session_id` 或文件名 stem 对齐。

报告默认写到：

```
benchmark/causal_memory/out/aig0045_smoke_report.json
```

模型侧 `.stats` 随 `ToolloopResult.audit` 一并进入报告的 `provider_audit`，
并在 work root 落 `provider_audit.jsonl`（进程退出钩子也会 flush）。

## 日志读取长度

`read_full_segment` 的整段返回上限为 20,000 字符（包含截断提示）。
单条 AO 的 action / observation 仍分别保留最多 2,000 字符并附截断标记。
整段只在 AO 之间截断，保留完整 AO 前缀及原始快照索引，不切断最后一条 AO。
截断提示给出已返回 / 总 AO 数和下一条 AO 索引；通过
`read_step_range` 或 `read_step` 读取剩余部分。
`read_step` 不应用读取层截断，但无法恢复素材导出时已经截掉的内容。

## 四角色优化与审计

- AO 在动作发生时分配索引，结果按 tool-call ID 回填；未知显式 ID 不猜测匹配。
- 导出 verdict 保留 manifest 原题。旧素材可加 `--manifest <manifest.jsonl>`；
  无 manifest 时从 AO 中的人类消息恢复，已被导出截掉的正文无法还原。
- Builder 每轮携带当前图，仅保留最近六次构建操作；已归图的旧证据省略，
  未归图证据和刚重新读取的证据保留。完整历史仍保存在审计中。
- Reader / Refiner 的工具历史只传一份，非工具结果单独保留。
- Refiner 拒绝无变化编辑；连续两次编辑失败或重复读取停止本轮。
  修改后的图在退出前重新审计，保留实际修复成果。
- Builder 连续六次未提供新信息且未改变图时以 `no_progress` 退出。
- 每次完整 Reader 运行保存 `reader-report.json`，含报告、hash 和读取证据。
- 结果区分 `builder_submitted` / `graph_frozen` / `reader_verified`，
  `no-signal` 保留 `skipped_no_signal`，不伪装成提交失败。
- 正式评测默认 `--attempt-policy fixed --attempts 3`；日常调试可用
  `--attempt-policy until-verified --attempts 3`，首个完整校验通过后停止。

## 素材准备

合成：驱动内置 6 条（code×3 + 文档×3），verdict 覆盖四域且覆盖
`{pass, fail, 无信号}`。完整 4×3 笛卡尔在
`tests/test_outcome_signal.py`。无信号行必须出现 `fallback_reason=not_triggered`。

真实：

1. 把 1933 容器里的会话树拷到宿主机（只读 `docker cp`，不要重启容器/redis）。
2. 跑导出脚本。**导出路径写死**：`benchmark/causal_memory/out/ao_records.jsonl`。

```bash
docker cp coforge-openviking-prototype-1:/data/viking/evol-389e96deae16/user/eval-admin/sessions /tmp/ov-sessions
uv run --no-sync python scripts/export_ao_records.py \
  --sessions-root /tmp/ov-sessions \
  --out benchmark/causal_memory/out/ao_records.jsonl
```

3. 准备 verdicts JSONL（评测 sidecar `/verify` 行）。
4. `--material real --verdicts … --sessions benchmark/causal_memory/out`。

## AO 布局探测结论（1933 只读，2026-10-12）

- 健康检查：`GET http://127.0.0.1:1933/health` →
  `openviking 0.4.9`，容器 `coforge-openviking-prototype-1`（`127.0.0.1:1933`）。
- 会话树存在：容器内
  `/data/viking/evol-389e96deae16/user/eval-admin/sessions`，
  **136** 个 `coforge-quiet-*` 目录（与任务书一致）。
- **`ao-ledger.jsonl` 在整个 `/data/viking` 下为 0。** AO 台账不在该层。
- 每个会话实际布局：`.meta.json` / `.overview.md` / `.abstract.md` /
  清空后的 live `messages.jsonl` / `history/archive_NNN/messages.jsonl`。
  `.meta.json` 见 `commit_count=1`、`message_count=0`、`retention` 已 drain。
- 配置 `/run/secrets/openviking_prototype_ov_conf` 仅有 `server/storage/vlm`，
  **无 causal / skill_trajectory**。因此
  `maybe_record_session_tool_exchanges` 不会写 ledger。
- 1933 OpenAPI **没有** AO/ledger 端点；`/api/v1/sessions` 需 API key（401）。
- 最近证据：`history/archive_*/messages.jsonl`（`role` + `parts`，多数 `type=text`）。
- 推导路径：导出脚本优先读 `ao-ledger.jsonl`；否则从 archive/live messages
  用 `AOLedgerRecorder.record_messages` 回收 tool 交换，再退回把文本 turn
  收成 `captured_state=derived_from_archive` 的 AO 行。这是派生，不是原生台账。
- 这些 quiet-window 会话不是完整 code toolloop；真实四域素材需要另备
  verdicts JSONL + 真正有 tool 交换（或已有 ledger）的会话。

## 冒烟通过标准（判定线）

报告末尾 `gates` 五条，①–④ 有明确 pass/fail，⑤ 只记录：

1. **submit 率 ≥ 80%**：分母是触发了四角色的素材（不含无信号）。
2. **Φ(G) paper_exact 结构违规 = 0**：对已 submit 的图跑 `evaluate_phi`。
3. **reader schema + 引用合规率**：`parse_reader_report` 通过且
   `decisive_ao_refs` 全落快照下标内。阈值同 80%。
4. **双模式可区分**：success 素材冻结图的 edge `effect` 不含「出错/错误」。
   抽检全部 success 行（100% ≥ 10%），`gates.dual_mode_distinguishable` 注明。
5. **预算耗尽率**：只记录，不作为失败线。默认预算 `calls=64` /
   `seconds=1800`（覆盖脚本回放默认 120s）。

无信号素材必须有「不触发」记录：`mode=null`、`submit=false`、
`fallback_reason=not_triggered`。

合成 + Fake 预填：①–④ pass，⑤ rate=0。真 gemma 冒烟以人工报告为准。
