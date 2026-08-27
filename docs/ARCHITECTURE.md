# 架构

A Share Lab 是本地优先的确定性研究程序。UI、MCP 和通知都只是入口；它们不能绕过数据
质量、策略和风险服务。

```mermaid
flowchart TB
    C["CSMAR\n只读历史基线"] --> H["Hybrid Loader\n显式拼接"]
    I["Infoway EOD\n沪深每日增量"] --> V["Quarantine / Verified\n覆盖·单位·指数共同截止"]
    V --> H
    P["Ports\n行情·财务·公告·通知"] --> A["Adapters\nCSMAR·授权行情·SQLite·Keychain"]
    A --> Q["Quality Gate\n共同截止·单位·PIT·权利范围"]
    H --> Q
    Q --> S["Research Services\n候选硬门·组合优化·行动单"]
    S --> R["Immutable Research Archive"]
    S --> U["Streamlit UI"]
    S --> M["Read-only MCP"]
    S --> N["Server酱 / Bark 摘要"]
```

## 分层规则

- `domain/`：不可变模型、状态和市场规则；
- `ports/`：供应商无关的输入/输出契约；
- `adapters/`：读取本地库或经授权 API，标准化后立即校验；
- `analytics/`：纯函数指标、趋势、介入点和组合风险；
- `services/`：编排数据门、研究漏斗、归档和通知；
- `ui/`：只展示服务返回的结构化结果；
- `mcp_server.py`：只读最小工具，不返回原始授权数据；
- `migrations/`：研究运行、证据、组合和真实结果的不可变档案。

语言模型不得计算价格、指标、权重、收益或概率。它可以在结构化结果形成后解释证据、
提出反例，并对最终少量候选做公开信息复核。

## 失败策略

系统 fail closed：来源失败、截止日不一致、字段漂移、单位不明、证据时点不安全或风险
预算失败时返回明确状态，不静默换源，也不放宽门槛凑股票。

## 每日收盘增量

- `csmar.duckdb` 永久作为只读历史基线；自动更新不会向它写入任何记录；
- Infoway 日线按供应商、未复权口径和交易日写入独立 `market_overlay`；
- 每个交易日先进入 staging。沪深股票覆盖率不低于 98%、六个核心指数齐全、交易日连续、
  日期/单位/OHLC/前收均合法后，股票与指数才一起写入 verified manifest；
- 不完整批次进入 quarantine，不能推进共同截止日；后续交易日不能越过失败日；
- 股票成交量按已校准合同由“手”转换为“股”，成交额为人民币元，并逐行验证隐含成交价
  落在当日高低价内；指数使用成分证券合计量额，不与指数点位相除；
- Infoway 当前自动母集明确为沪深 A 股，不包含北交所，也排除其清单中混入的指数和 B 股；
- 混合读取只追加历史基线之后的 verified 交易日。重叠数据不覆盖 CSMAR，新代码若不在
  CSMAR 证券主表中则隔离，所有行继续保留来源与取得时间。

## 入口边界

- 本地网页：面向非技术用户的主入口；
- MCP：供 ChatGPT/Codex 在本机只读调用；
- 通知：只发行动摘要，任何通道失败都不改变研究结论；
- 本项目没有券商下单端口。
