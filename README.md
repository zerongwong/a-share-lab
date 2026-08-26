# A股研究室（A Share Lab）

A Share Lab 是一个本地优先、证据优先的 A 股中期研究项目。主流程面向 **1 周到 1 年**
的持有期，默认使用 **3 个月（13 周）**，通常用于 1–3 个月的组合研究。它从用户合法
取得并保存在本机的数据中筛选四只股票，按 `3:2:2:1` 排列，同时保留现金并明确显示
数据截止日、排除原因、历史风险统计和失效条件。

项目不连接券商、不自动下单、不承诺收益，也不把历史重叠窗口统计表述成未来概率。
尾盘打板、日内高抛低吸和实时交易监控不属于当前主产品；相关实验代码即使仍在源码中，
也不会在主导航中启用。

## 设计原则

- 数据和确定性规则负责计算，模型只负责解释证据；
- 财务和公告必须按实际披露时间使用，不能用会计期末日代替可知时间；
- 缺失因子保持缺失并降低覆盖度，不能填成“中性 0.5”；
- ST、退市、停牌、形成日不可买、数据不足和走势过热可以让结果为空；
- 每次研究记录数据截止、规则版本、输入哈希、风险提示和失效条件；
- 原始预测不可事后改写，真实结果写入独立记录；
- 融资默认关闭，不自动加杠杆，也不向亏损仓摊低成本。

## 当前主流程

1. 从本机标准化 CSMAR 日线库读取共同截止日的 A 股截面；
2. 过滤历史不足、流动性不足、异常原始价格跳变、ST/退市和不可执行标的；
3. 以共同截止日全市场宽度和核心指数 MA20/60/120、波动、回撤构成双重大盘风险门；
   任一明确 `risk_off` 都暂停建立新组合；
4. 资产负债表当前快照只有在决策日晚于取得日且个股价格也更新到取得日后，才作为较低
   权重的“资产负债表稳健度”窄因子；历史回放永远禁用；
5. 只有在完整、点时覆盖时才启用完整财务、新闻和板块因子；大盘环境不作为单只股票加分项；
6. 在合格集合中生成四股组合，并按 `3:2:2:1` 分配股票仓；
7. 展示历史 CAGR、波动、Sharpe、Sortino、Calmar、回撤分布和收益情景；
8. 可选地把研究结论写入本机不可篡改的 SQLite 档案。

可选持有期为：

| 选项 | 交易周 | 典型用途 |
|---|---:|---|
| 1 周 | 1 | 最短观察周期，噪声最高 |
| 1 个月 | 4 | 短中期波段 |
| 3 个月 | 13 | 默认中期周期 |
| 6 个月 | 26 | 中长期趋势 |
| 1 年 | 52 | 长期研究 |

持有期是统计和研究窗口，不代表必须持有到期。结构失效、基本面恶化或数据质量门触发时，
应重新研究；程序不会自动卖出。

“当前研究”不会把陈旧截面伪装成今日结果：个股日线至少要覆盖上一完整工作日，否则返回
数据未就绪；中国法定休市日历尚未接入时采用保守工作日判断，遇节假日可切换“历史回放”
复核明确截止日，但历史模式不会使用今天取得的财务快照。

## 安装

推荐 Python 3.12：

```bash
cd a_share_lab
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

AKShare 适配器是个人研究用的可选依赖：

```bash
.venv/bin/python -m pip install -e ".[akshare,dev]"
```

在 VS Code 中打开本目录后，可从“终端 → 运行任务”启动界面或运行测试。手动启动：

```bash
.venv/bin/python -m streamlit run src/ashare_lab/ui/A股研究室.py
```

界面默认只监听 `127.0.0.1:8501`。

## 本地数据

源码仓库不包含任何真实市场数据。默认运行数据目录位于 macOS：

```text
~/Library/Application Support/A股研究助手/
├── research.db
├── cache/
│   ├── csmar/
│   │   └── csmar.duckdb
│   └── csmar_reference/
│       └── csmar_reference.duckdb
└── reports/
```

其中：

- `csmar.duckdb` 是用户在本机生成的标准化研究库；
- `csmar_reference.duckdb` 保存独立的核心指数PIT日线和资产负债表当前快照；
- `research.db` 保存不可篡改研究档案与后续结果；
- 原始 ZIP、Excel、Parquet、DuckDB、SQLite、新闻正文和生成报告均被 `.gitignore`
  排除；
- CSMAR、iFinD、Choice、Infoway、LongBridge 等数据仍受各自许可约束，开源许可证只
  覆盖本项目代码，不授予第三方数据权利。

当合法导出中另有 `FS_Combas` 资产负债表和 `IDX_Idxtrd` 指数文件时，可导入独立参考库：

```bash
.venv/bin/ashare-import-csmar-reference "/path/to/CSMAR-export" \
  --common-cutoff 2026-08-24 --retrieved-at 2026-08-25
```

资产负债表若缺少普通财报实际公告日，只标记为取得日的当前快照，禁止回填历史回测；
指数记录若原始OHLC关系不可能，会被隔离并在导入报告中计数，不会被程序擅自修复。

不要把学校、公司或个人账户取得的数据发到 GitHub。另一台电脑使用时，应由使用者自己
取得合法数据权限、在本机导入，并配置自己的数据接口凭据。

## 在 ChatGPT/Codex 中只读调用

项目提供一个最小、无界面的 MCP 入口。它使用官方 Python MCP SDK，因此是独立的可选
依赖，不影响本地网页：

```bash
.venv/bin/python -m pip install -e ".[mcp,dev]"
cp .env.example .env
```

按本机情况设置环境变量。`ASHARE_MCP_ALLOW_LICENSED_DERIVED_RESULTS` 默认是 `false`；
只有在确认数据许可允许连接的模型服务处理衍生研究结果后，才可以在本机环境中改为
`true`。不要把真实路径中的授权文件、API 密钥或隧道令牌提交到仓库。

启动 Streamable HTTP MCP：

```bash
set -a
source .env
set +a
.venv/bin/ashare-mcp
```

默认地址是 `http://127.0.0.1:8765/mcp`。提供三个只读工具：

- `get_data_status`：检查本地目录、研究档案和许可开关，不返回路径或密钥；
- `generate_portfolio`：以 `live` 或 `historical` 模式临时生成四股组合，不写档案、
  不返回原始数据；历史模式不会带入今天取得的财务快照；
- `get_latest_research`：只读查询最近一次组合或个股研究，不读取持仓和新闻正文。

所有工具都声明 `readOnlyHint=true`、`destructiveHint=false` 和
`openWorldHint=false`，服务不提供下单、修改持仓或删除档案的能力。

ChatGPT 开发者模式连接本地 MCP 时需要一个可访问的 HTTPS 地址。开发测试可使用临时、
受访问控制的 HTTPS 隧道，把它转发到本机 `/mcp`；当前版本没有多人身份认证，因此不要
把它作为长期公开网址。要给家人使用，更安全的方式是让对方克隆项目、安装自己的本地
服务并使用自己的授权数据。若未来部署共享服务，必须先增加 OAuth、用户隔离、速率限制、
日志脱敏和数据许可审查。

当前 MCP 保持 data-only，不带 ChatGPT 内嵌组件。先把三个工具的输入、输出和只读边界
验证稳定，再决定是否增加组件，符合官方的“先完成一个窄目标”的构建顺序：
[Bring your app to ChatGPT](https://learn.chatgpt.com/use-cases/chatgpt-apps)。

## 配置和凭据

- 公共、无密钥配置放在 `config/`；
- 示例环境变量见 `.env.example`；
- 真实密钥只放操作系统钥匙串、部署平台 Secret Store 或进程环境；
- 不在源码、TOML、测试、截图、Issue、聊天或日志中粘贴密钥；
- 密钥一旦暴露，立即在数据提供方控制台吊销并重新生成。

## 测试

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

网络集成测试应显式标记。普通测试必须能在无网络、无真实数据、无密钥的环境中运行。
MCP 测试使用合成 SQLite 夹具，验证默认关闭、只读查询和不返回原始数据。

## 开源和贡献

本项目采用 [Apache License 2.0](LICENSE)，与相邻的 TradingAgents 项目许可证保持一致，
并保留专利授权与 NOTICE 机制。详细的数据排除和第三方归属说明见 [NOTICE](NOTICE)，
安全报告方式见 [SECURITY.md](SECURITY.md)，贡献规则见
[CONTRIBUTING.md](CONTRIBUTING.md)。

如果未来复制或修改 TradingAgents 或其他第三方源码，必须先确认许可证兼容，并保留原始
版权、归属与修改说明。普通网页“公开可见”不等于允许批量抓取、缓存、模型处理或再发布。

## 研究免责声明

本项目只用于研究与软件验证，不构成投资顾问、证券推荐、收益保证或交易执行服务。
历史收益、Sharpe、回撤、情景分位和排名分都可能因幸存者偏差、复权、披露时点、滑点、
税费、成交限制和市场结构变化而失效。任何真实交易决定与损失由使用者自行承担。
