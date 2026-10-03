import glob, os, itertools, warnings, pickle
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FEE = 0.07
SYM = {"BTC":"btc","ETH":"eth","SOL":"sol","XRP":"xrp","DOGE":"doge","BNB":"bnb","HYPE":"hype"}

R = pd.read_parquet("res_cache_5m_all.parquet")
BK = pd.read_parquet("book_5m.parquet")
BK["ts"] = pd.to_datetime(BK.mkt_open, utc=True); BK["asset"] = BK.asset.astype(str)
A5 = BK[BK.elapsed_req == 5][["asset","ts","au","ad"]].rename(columns={"ad":"ask_up","au":"ask_dn"})

rows = []
for f in sorted(glob.glob("port_logs/*.csv")):
    d = pd.read_csv(f, on_bad_lines="skip")
    stream = os.path.basename(f).replace(".csv","")
    d["stream"] = stream
    d["cell"] = d["strategy"].astype(str)
    d["asset"] = d.symbol.str.split("/").str[0].str.upper().map(SYM)
    d["ts"] = pd.to_datetime(d.start_date, format="%m/%d/%y %H:%M", errors="coerce", utc=True)
    d["side"] = np.where(d.forecast.astype(str).str.contains("⬆"), "UP",
                np.where(d.forecast.astype(str).str.contains("⬇"), "DOWN", None))
    rows.append(d[["stream","cell","asset","ts","side"]])
S = pd.concat(rows, ignore_index=True).dropna(subset=["ts","side","asset"])
S = S.merge(R, on=["asset","ts"], how="left").merge(A5, on=["asset","ts"], how="left")
S["win"] = np.where(S.side=="UP", S.y, 1-S.y)
S["ask"] = np.where(S.side=="UP", S.ask_up, S.ask_dn)
S = S.dropna(subset=["y","ask"]).copy()
S["pps"] = np.where(S.win==1, 1-S.ask, -S.ask) - FEE*S.ask*(1-S.ask)
S["pnl5"] = S.pps * (5.0/S.ask)
print(f"raw resolved+priced rows (9 original streams): {len(S)}")

# candidates = (stream, cell, side) combos with n>=5 -- same grouping as the pasted portfolio table
cand = S.groupby(["stream","cell","side"])
cand_stats = cand.agg(n=("win","size"), w=("win","sum"), pnl=("pnl5","sum")).reset_index()
cand_stats = cand_stats[cand_stats.n >= 5].copy()
cand_stats["wr"] = cand_stats.w/cand_stats.n*100
cand_stats["roi"] = cand_stats.pnl/(5*cand_stats.n)*100
print(f"candidate (stream,cell,side) combos with n>=5: {len(cand_stats)}")

# event index-sets per candidate (for dedup-aware greedy union)
S = S.reset_index(drop=True)
groups = S.groupby(["stream","cell","side"]).indices  # dict: key -> array of row indices

with open("portfolio_candidates.pkl","wb") as f:
    pickle.dump({"S": S, "cand_stats": cand_stats, "groups": groups}, f)
print("saved portfolio_candidates.pkl")
print(cand_stats.sort_values("pnl", ascending=False).head(10).to_string(index=False))
