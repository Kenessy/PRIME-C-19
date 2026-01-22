"""
Simple live dashboard for VRAXION logs.

Usage:
  streamlit run tools/live_dashboard.py -- --log logs/current/tournament_phase6.log

Dependencies:
  pip install streamlit plotly pandas
"""

import argparse
import os
import re
from typing import List, Dict

import pandas as pd
import streamlit as st
import plotly.express as px


LOG_PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[\d\.]+)\s+\|"
    r".*?shard=(?P<shard_count>[\d\-]+)/(?P<shard_size>[\d\-]+)"
    r"(?:,\s*traction=(?P<traction>[\d\.\-]+))?"
)


def parse_log(path: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            try:
                rows.append(
                    {
                        "step": int(m.group("step")),
                        "loss": float(m.group("loss")),
                        "shard_count": float(m.group("shard_count")),
                        "shard_size": float(m.group("shard_size")),
                        "traction": float(m.group("traction")) if m.group("traction") else None,
                    }
                )
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["step"], keep="last")
        .sort_values("step")
        .reset_index(drop=True)
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/current/tournament_phase6.log", help="Path to log file")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh seconds")
    args = parser.parse_args()

    st.set_page_config(page_title="VRAXION Live Dashboard", layout="wide")
    st.title("VRAXION Live Dashboard")
    st.caption(f"Log: {os.path.abspath(args.log)} (refresh {args.refresh}s)")

    df = parse_log(args.log)
    if df.empty:
        st.warning("No parsed data yet. Waiting for log lines with shard info...")
        st.experimental_rerun()

    toggles = st.multiselect(
        "Series to show",
        ["loss", "shard_count", "shard_size", "traction"],
        default=["loss", "shard_count", "traction"],
    )

    if toggles:
        melted = df.melt(id_vars="step", value_vars=toggles, var_name="metric", value_name="value")
        fig = px.line(melted, x="step", y="value", color="metric", markers=True)
        fig.update_layout(legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one series.")

    st.experimental_rerun()


if __name__ == "__main__":
    main()
