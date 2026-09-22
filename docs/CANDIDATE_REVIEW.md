# 候选财务与公告正文审阅

这是“周线突破候选 → 财务与公告双重确认 → 组合比较”中的人工/研究者审阅工具，
**不是自动尽调或自动生成通过结论的工具**。现有数值财务、交易状态核验仍独立运行，
公告审阅通过也不代表最终允许买入。

默认读取本机应用数据目录下的 `cache/candidate_evidence`，审阅文件保存在
`candidate_reviews`。以下命令中的代码必须带已核验交易所，如 `601298.SH`；
仅作使用示例，不是对该股票的推荐或已经完成审阅的声明。

## 1. 查看待审阅候选与创建草稿

先运行正常研究流程取得候选的当天取证回执，再执行：

```bash
.venv/bin/python -m ashare_lab.cli.candidate_review status
.venv/bin/python -m ashare_lab.cli.candidate_review template 601298.SH
```

模板仅在回执和完整公告清单有效时创建，输出本机草稿路径。模板的状态为 `pending`，
全部检查项为 `false`，逐条公告分类为 `pending`，不填审阅人、不填结论、不伪造已经阅读的文件。
重复创建同一草稿不会覆盖文件；草稿本身不会成为生产通过证据。

## 2. 下载并研读官方原文

草稿 `available_documents` 列出本轮公告标题、时间、编号和官方链接。
下载其中指定编号：

```bash
.venv/bin/python -m ashare_lab.cli.candidate_review download 601298.SH 公告编号
```

只允许清单中的 `https://static.cninfo.com.cn/finalpage/...PDF`，禁止跳转至其他网站，
每份文件有超时和 50MB 限制。返回相对文件路径、原链接和 SHA256；**下载不等于研读**，
不会自动填写文件角色、发现或通过状态。

研究者应实际阅读最新完整财报、最近年度审计材料及潜在重大风险公告，逐条审阅完整公告清单，
把财务数值与原文核对。编辑草稿时逐项记录：

- 真实审阅人、含时区的完成时间、具体理由；未完成时保持 `pending`。
- `documents`：公告编号、原链接、相对文件路径、SHA256、具体原文发现及角色。
  最新财报角色为 `latest_financial_report`，年度审计为 `annual_audit`，其他风险文档可用 `material_risk`。
- `triage`：每条公告真实判断为 `document_reviewed` 或 `routine_nonmaterial`，并填写理由；
  潜在重大风险公告不能仅凭标题归为常规事项。
- `checklist`：仅实际完成对应工作后逐项确认，包括 `financial_metrics_cross_checked`。
- `status`：证据与研读均完成才可填 `pass`；发现否决事项填 `veto` 并说明。不能批量机械勾选来凑推荐。

报告期不等于首次公告日；最新财报与年度审计角色会核对公告标题和适用期间。
CSMAR 资产负债表不能代替利润、现金流或官方正文。

## 3. 验证，再显式登记

```bash
.venv/bin/python -m ashare_lab.cli.candidate_review validate 601298.SH --file /本机草稿绝对路径.json
.venv/bin/python -m ashare_lab.cli.candidate_review register 601298.SH --file /本机草稿绝对路径.json --confirm-reviewed
```

验证不修改正在生效的审阅文件。登记必须显式确认真实研读，且当前清单、财务哈希、文件哈希、
时间与审阅内容全部通过验证；`pending` 或无效草稿不可登记。已有生效文件不会默认覆盖，
确需更新时添加 `--replace-existing`；旧文件和登记审计保存在 `history` 中，可恢复。

回执过期、清单或财务内容变化、缺少原文、文件被修改、未来时间及跨目录路径都会失败关闭。
次日先刷新回执：若出现新公告或财务更正，应重新研读变化，不能沿用旧“通过”。
退出码：成功操作 `0`，校验结果仍未知 `3`，操作被拒绝或失败 `2`。

可用命令前的 `--cache-dir` 和 `--review-dir` 指定测试目录，默认不需要设置。
本工具不连接券商、不修改持仓、不发送微信、不自动下单；私有回执、原文与审阅记录不提交开源仓库。
