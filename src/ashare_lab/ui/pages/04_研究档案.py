from __future__ import annotations

import pandas as pd
import streamlit as st

from ashare_lab.bootstrap import application_data_dir, build_repository

st.set_page_config(page_title="研究档案", page_icon="🗂️", layout="wide")
st.title("研究档案")
st.caption("原预测不可修改；未来真实表现写入另一张结果表，用来检验而不是美化历史。")

try:
    rows = build_repository().list_runs(limit=200)
except Exception as exc:
    st.error(f"档案暂时无法读取：{exc}")
    rows = []

if not rows:
    st.info("还没有研究记录。先到“个股四周期”完成一次分析。")
else:
    display = pd.DataFrame(rows)
    keep = [
        "created_at",
        "run_type",
        "as_of",
        "data_cutoff",
        "strategy_version",
        "status",
        "id",
    ]
    st.dataframe(
        display[[column for column in keep if column in display]], hide_index=True, width="stretch"
    )

with st.expander("数据库位置"):
    st.code(str(application_data_dir() / "research.db"))
