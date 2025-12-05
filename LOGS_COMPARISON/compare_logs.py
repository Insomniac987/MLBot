#!/usr/bin/env python3
"""
compare_logs.py

Lee y compara:
 - /mnt/data/logs_backtesting.txt
 - /mnt/data/prod_log_params.txt

Produce:
 - /mnt/data/compare_rows_backtest.csv
 - /mnt/data/compare_rows_prod.csv
 - /mnt/data/comparison_summary.json
 - /mnt/data/comparison_plots.png

Diseñado para ser tolerante a variaciones en el formato de logs.
"""
import re
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BACKTEST_LOG = Path("logs_backtesting.txt")
PROD_LOG = Path("prod_log_params.txt")

def parse_log(path: Path):
    """
    Parse a log file into a list of dicts (rows). The log format is inconsistent;
    this function builds rows by reading lines and collecting fields until a new Step appears.
    """
    rows = []
    cur = {}
    # common regexes
    step_re = re.compile(r"\b[Ss]tep[:\s]*([0-9]+)")
    step_alt_re = re.compile(r"\bstep[:\s]*([0-9]+)")
    action_re = re.compile(r"[Aa]ction[:\s]*\[*([^\],]+)\]*")  # captures e.g. [0] or 0
    reward_re = re.compile(r"[Rr]eward[:\s]*([-\d\.]+)")
    equity_re = re.compile(r"[Ee]quity[:\s]*([-\d\.]+)")
    equity_norm_re = re.compile(r"[Ee]quity_norm[:\s]*([-\d\.]+)")
    cur_pct_re = re.compile(r"\bcur_pct[:\s]*([-\d\.e]+)")
    drawdown_re = re.compile(r"\bdrawdown[:\s]*([-\d\.]+)")
    pos_re = re.compile(r"\bpos[:\s]*\[(.*?)\]")
    tipo_re = re.compile(r"Tipo_trade[:\s]*([\w_]+)")
    profit_re = re.compile(r"\bProfit[:\s]*([-\d\.]+)")
    # lines that can include key:value of form "key: value"
    generic_kv_re = re.compile(r"^\s*([a-zA-Z_]+)\s*:\s*(.+)$")

    def flush_current():
        nonlocal cur, rows
        if cur:
            # coerce numeric types
            for k in list(cur.keys()):
                v = cur[k]
                # try numeric
                if isinstance(v, str):
                    v_strip = v.strip()
                    try:
                        if v_strip == "":
                            continue
                        if re.match(r"^-?\d+(\.\d+)?$", v_strip):
                            cur[k] = float(v_strip)
                        else:
                            # keep as string
                            cur[k] = v
                    except Exception:
                        pass
            rows.append(cur)
            cur = {}

    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # detect start of new step
        m = step_re.search(line) or step_alt_re.search(line)
        if m:
            # start new row
            # but if current has content, flush first
            if cur:
                flush_current()
            cur = {}
            cur["step"] = int(m.group(1))
            # still parse the remainder of line for other fields
            # continue to try regex extraction on same line
        # try extract action/reward/equity etc from the line and add to cur
        m_action = action_re.search(line)
        if m_action:
            a = m_action.group(1).strip()
            # normalize action like "[0]" or "0"
            a = re.sub(r"[^\-0-9]", "", a)
            try:
                cur["action"] = int(a)
            except:
                cur["action"] = a
        m_reward = reward_re.search(line)
        if m_reward:
            try:
                cur["reward"] = float(m_reward.group(1))
            except:
                pass
        m_equity = equity_re.search(line)
        if m_equity:
            try:
                cur["equity"] = float(m_equity.group(1))
            except:
                pass
        m_equity_norm = equity_norm_re.search(line)
        if m_equity_norm:
            try:
                cur["equity_norm"] = float(m_equity_norm.group(1))
            except:
                pass
        m_curpct = cur_pct_re.search(line)
        if m_curpct:
            try:
                cur["cur_pct"] = float(m_curpct.group(1))
            except:
                pass
        m_draw = drawdown_re.search(line)
        if m_draw:
            try:
                cur["drawdown"] = float(m_draw.group(1))
            except:
                pass
        m_pos = pos_re.search(line)
        if m_pos:
            # store as text or list of ints
            s = m_pos.group(1)
            try:
                cur["pos"] = [int(x.strip()) for x in s.split(",") if x.strip()!='']
            except:
                cur["pos"] = s
        m_tipo = tipo_re.search(line)
        if m_tipo:
            cur["tipo_trade"] = m_tipo.group(1)
        m_profit = profit_re.search(line)
        if m_profit:
            try:
                cur["profit"] = float(m_profit.group(1))
            except:
                pass

        # generic key: value lines (like "drawdown: 0.016")
        m_kv = generic_kv_re.match(line)
        if m_kv:
            k, v = m_kv.group(1).strip(), m_kv.group(2).strip()
            # avoid overwriting fields already captured
            if k not in cur:
                # attempt numeric
                try:
                    if re.match(r"^-?\d+(\.\d+)?$", v):
                        cur[k] = float(v)
                    else:
                        cur[k] = v
                except:
                    cur[k] = v

    # flush final
    if cur:
        flush_current()

    df = pd.DataFrame(rows)
    # Normalize columns
    # ensure step sorted and unique
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    # fill missing numeric with NaN
    numeric_cols = ["reward","equity","equity_norm","cur_pct","drawdown","profit"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def align_and_compare(df_b, df_p):
    """
    Align both DataFrames by normalizing the step axis to [0,1] and then aligning
    by nearest normalized position. This lets us compare logs of different lengths/dates.
    """
    def add_norm_index(df):
        if "step" in df.columns:
            smin = df["step"].min()
            smax = df["step"].max()
            if smax == smin:
                df["_norm_pos"] = 0.0
            else:
                df["_norm_pos"] = (df["step"] - smin) / (smax - smin)
        else:
            df["_norm_pos"] = np.linspace(0,1,len(df))
        return df

    df_b = add_norm_index(df_b.copy())
    df_p = add_norm_index(df_p.copy())

    df_b = clean_for_merge(df_b, "BACKTEST")
    df_p = clean_for_merge(df_p, "PROD")

    # join by nearest _norm_pos using merge_asof
    df_b_sorted = df_b.sort_values("_norm_pos").reset_index(drop=True)
    df_p_sorted = df_p.sort_values("_norm_pos").reset_index(drop=True)

    merged = pd.merge_asof(df_b_sorted, df_p_sorted, on="_norm_pos",
                           suffixes=("_b", "_p"), direction="nearest", tolerance=0.05)
    # tolerance 0.05 -> align points roughly in same phase (±5% of episode). If NaN, no close match.
    # For unmatched, we'll keep them as separate later.

    # compute comparison metrics per row where both exist
    cmp = merged.copy()
    def safe_diff(a,b):
        if pd.isna(a) or pd.isna(b): return np.nan
        return a - b
    cmp["delta_equity_norm"] = cmp.apply(lambda r: safe_diff(r.get("equity_norm_b"), r.get("equity_norm_p")), axis=1)
    cmp["same_action"] = cmp.apply(lambda r: (not pd.isna(r.get("action_b")) and not pd.isna(r.get("action_p"))
                                              and int(r.get("action_b")) == int(r.get("action_p"))), axis=1)
    # other stats
    stats = {}
    # action distributions
    for which, df in [("backtest", df_b), ("prod", df_p)]:
        actions = df["action"].dropna().astype(str) if "action" in df.columns else pd.Series([], dtype=float)
        stats[f"{which}_n_rows"] = len(df)
        stats[f"{which}_unique_actions"] = actions.value_counts().to_dict()
        stats[f"{which}_pct_holds"] = (actions == "0").sum() / max(1, len(actions))
        # approximated winrate if reward or profit present
        if "reward" in df.columns:
            stats[f"{which}_mean_reward"] = float(df["reward"].dropna().mean())
            stats[f"{which}_median_reward"] = float(df["reward"].dropna().median())
        if "profit" in df.columns:
            wins = (df["profit"] > 0).sum()
            losses = (df["profit"] < 0).sum()
            stats[f"{which}_wins"] = int(wins)
            stats[f"{which}_losses"] = int(losses)
            stats[f"{which}_winrate_est"] = float(wins / max(1, wins+losses))
    # merged metrics
    valid_pairs = cmp.dropna(subset=["action_b","action_p"])
    stats["pairs_aligned"] = int(len(valid_pairs))
    stats["pairs_same_action_pct"] = float(valid_pairs["same_action"].mean()) if len(valid_pairs) else None
    # correlation between equity_norm columns if exist
    if "equity_norm_b" in cmp.columns and "equity_norm_p" in cmp.columns:
        tmp = cmp[["equity_norm_b","equity_norm_p"]].dropna()
        if len(tmp) > 2:
            stats["equity_norm_corr"] = float(tmp["equity_norm_b"].corr(tmp["equity_norm_p"]))
        else:
            stats["equity_norm_corr"] = None

    # divergence metrics
    stats["mean_delta_equity_norm"] = float(cmp["delta_equity_norm"].abs().dropna().mean()) if "delta_equity_norm" in cmp.columns else None
    stats["max_delta_equity_norm"] = float(cmp["delta_equity_norm"].abs().dropna().max()) if "delta_equity_norm" in cmp.columns else None

    # action distribution difference
    vc_b = pd.Series(stats.get("backtest_unique_actions",{}))
    vc_p = pd.Series(stats.get("prod_unique_actions",{}))
    # align indices
    all_actions = set(vc_b.index.astype(str)) | set(vc_p.index.astype(str))
    diff = {}
    for a in all_actions:
        bcnt = int(vc_b.get(a,0))
        pcnt = int(vc_p.get(a,0))
        diff[a] = {"backtest": bcnt, "prod": pcnt, "diff": bcnt-pcnt}
    stats["actions_count_comparison"] = diff

    return cmp, stats, df_b, df_p

def save_outputs(cmp_df, df_b, df_p, stats):
    out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_b.to_csv(out_dir / "compare_rows_backtest.csv", index=False)
    df_p.to_csv(out_dir / "compare_rows_prod.csv", index=False)
    cmp_df.to_csv(out_dir / "compare_rows_merged.csv", index=False)
    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved CSVs and summary to {out_dir}")

    # plots
    fig, axes = plt.subplots(3,1, figsize=(10,12))
    # 1) action counts
    if "action" in df_b.columns:
        axes[0].bar(df_b["action"].astype(str).value_counts().index.astype(str),
                    df_b["action"].astype(str).value_counts().values, alpha=0.6, label="backtest")
    if "action" in df_p.columns:
        axes[0].bar(df_p["action"].astype(str).value_counts().index.astype(str),
                    df_p["action"].astype(str).value_counts().values, alpha=0.6, label="prod")
    axes[0].set_title("Action distribution (count)")
    axes[0].legend()
    # 2) equity_norm over normalized step
    if "equity_norm" in df_b.columns:
        axes[1].plot(df_b["_norm_pos"], df_b["equity_norm"], label="backtest")
    if "equity_norm" in df_p.columns:
        axes[1].plot(df_p["_norm_pos"], df_p["equity_norm"], label="prod")
    axes[1].set_title("equity_norm over normalized episode")
    axes[1].legend()
    # 3) scatter same-action indicator from merged
    # attempt to load merged file
    try:
        merged = pd.read_csv("compare_rows_merged.csv")
        if "same_action" in merged.columns:
            sa = merged["same_action"].fillna(False).astype(int)
            axes[2].scatter(merged["_norm_pos"], sa, s=6)
            axes[2].set_ylim(-0.1,1.1)
            axes[2].set_title("Same action (1) / different action (0) per aligned row")
    except Exception:
        pass

    plt.tight_layout()
    fig_path = Path("comparison_plots.png")
    plt.savefig(fig_path)
    print(f"Saved plot to {fig_path}")

def clean_for_merge(df, side):
    before = len(df)
    df = df.dropna(subset=["_norm_pos"]).copy()
    after = len(df)
    print(f"[{side}] dropped {before - after} rows with NaN _norm_pos")
    return df

def main():
    print("Parsing backtest log:", BACKTEST_LOG)
    df_b = parse_log(BACKTEST_LOG)
    print("Parsed backtest rows:", len(df_b))

    print("Parsing prod log:", PROD_LOG)
    
    df_p = parse_log(PROD_LOG)
    print("Parsed prod rows:", len(df_p))

    if df_b.empty and df_p.empty:
        print("Ambos logs parecen vacíos, revisa las rutas.")
        return

    merged_cmp, stats, df_b, df_p = align_and_compare(df_b, df_p)
    print("Comparison stats summary snippet:")
    for k,v in list(stats.items())[:10]:
        print(" ", k, ":", v)

    save_outputs(merged_cmp, df_b, df_p, stats)
    print("Done. Outputs in current path")

if __name__ == "__main__":
    main()
