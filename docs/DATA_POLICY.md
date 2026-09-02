# 数据与许可政策

Apache-2.0 只授权本项目源码，不授权任何第三方市场数据、新闻、研报或账户内容。

## 可以进入仓库

- 源码、字段契约和迁移；
- 完全合成、不可还原真实供应商数据的测试夹具；
- 不包含账户、路径、密钥和持仓的文档；
- 数据文件的不可逆校验逻辑，但不是校验对象本身。

## 不得进入仓库

- CSMAR、Tushare、BaoStock、AKShare、iFinD、Choice、Infoway、LongBridge 等原始或转换数据；
- Excel、ZIP、Parquet、DuckDB、SQLite、缓存和研究报告；
- 新闻或研报正文；
- API Key、OAuth Token、Server酱 SendKey、Bark 设备 Key、券商凭据；
- 真实持仓、账户金额、未脱敏截图和日志。

## 数据使用原则

1. 每位使用者自行取得合法权限并在本机导入；
2. `event_at`、`published_at`、`provider_available_at`、`first_seen_at` 和
   `retrieved_at` 分开保存；
3. 训练/回放只使用形成时点已经可知的数据；
4. 未获授权的正文不落盘、不发给模型、不在通知或 UI 中再分发；
5. 更换供应商必须形成独立数据快照，不能无提示拼接序列；
6. 凭据存操作系统钥匙串或 Secret Store，暴露后立即在提供方轮换。

## 本机混合数据边界

- CSMAR 历史基线、当前 `zero_budget_eod` 收盘链和旧 Infoway 链分目录保存，不把外源
  数据冒充 CSMAR，也不跨链拼接；
- Tushare/BaoStock/AKShare 与 Infoway 的标准化 Parquet、manifest、receipt、quarantine 都属于个人研究数据，必须
  留在本机并由 `.gitignore` 排除；
- 自动更新目前只声明沪深 A 股覆盖。北交所、新闻、财务与公告不会因价格源可用而被推定
  为已授权或已覆盖；
- 只有完整通过质量门的交易日才能进入研究。失败批次可以保留作本地审计，但不得作为
  “部分可用”数据静默拼入；
- 任一供应商字段或单位发生变化时停止更新并提示升级适配器，不猜测倍数；AKShare 只核验，
  不会被当作 Tushare 的修补或静默替代源。

公开 App、多人服务和市场行情再展示需要独立的数据授权、隐私与证券业务合规审查；个人
本地研究许可不能自动扩大为商业再分发许可。
