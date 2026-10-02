import pandas as pd, numpy as np, glob, os, warnings; warnings.filterwarnings('ignore')
FEE = 0.07
SYM = {"BTC":"btc","ETH":"eth","SOL":"sol","XRP":"xrp","DOGE":"doge","BNB":"bnb","HYPE":"hype"}

# ---- 1. rebuild the flat signal table from the raw port_logs CSVs ----
R  = pd.read_parquet("res_cache_5m_all.parquet")
BK = pd.read_parquet("book_5m.parquet")
BK["ts"]    = pd.to_datetime(BK.mkt_open, utc=True)
BK["asset"] = BK.asset.astype(str)
A5 = BK[BK.elapsed_req==5][["asset","ts","au","ad"]].rename(columns={"ad":"ask_up","au":"ask_dn"})

rows = []
for f in sorted(glob.glob("port_logs/*.csv")):
    d = pd.read_csv(f, on_bad_lines="skip")
    d["stream"] = os.path.basename(f)[:-4]
    d["cell"]   = d.get("strategy", "?").astype(str)          # <-- "cell" IS the raw strategy label
    d["asset"]  = d.symbol.str.split("/").str[0].str.upper().map(SYM)
    d["ts"]     = pd.to_datetime(d.start_date, format="%m/%d/%y %H:%M", errors="coerce", utc=True)
    d["prob"]   = pd.to_numeric(d.get("prob"), errors="coerce")
    d["side"]   = np.where(d.forecast.astype(str).str.contains("⬆"), "UP",
                  np.where(d.forecast.astype(str).str.contains("⬇"), "DOWN", None))
    rows.append(d[["stream","cell","asset","ts","side","prob"]])

S = pd.concat(rows, ignore_index=True).dropna(subset=["ts","side","asset"])
S = S.merge(R, on=["asset","ts"], how="left").merge(A5, on=["asset","ts"], how="left")
S["win"] = np.where(S.side=="UP", S.y, 1-S.y)
S["ask"] = np.where(S.side=="UP", S.ask_up, S.ask_dn)
res = S.dropna(subset=["y","ask"]).copy()
res["pps"]  = np.where(res.win==1, 1-res.ask, -res.ask) - FEE*res.ask*(1-res.ask)
res["pnl5"] = res.pps * (5.0/res.ask)

# ---- 2. dedupe to one row per real event, keep the set of cell-tags it carries ----
ded = (res.groupby(["asset","ts","side"])
          .agg(prob=("prob","first"), win=("win","first"), ask=("ask","first"), pnl5=("pnl5","first"),
               cells=("cell", lambda s: "|".join(sorted(set(s)))))
          .reset_index())

def report(name, mask):
    sub = ded[mask]
    n = len(sub)
    print(f"{name}: n={n}  wins={int(sub.win.sum())}  WR={sub.win.mean()*100:.2f}%  "
          f"pnl@5={sub.pnl5.sum():+.2f}  ROI={sub.pnl5.sum()/(5*n)*100:+.2f}%")

# ---- FILTER 1: single tag ----
f1 = ded.cells.str.contains("P2_hr55_71_K2")
report("FILTER 1  cell==P2_hr55_71_K2", f1)

# ---- FILTER 2 (triple/compound): all three clauses ANDed together ----
f3 = (ded.cells.str.contains("P2_hr55_71_K2")
      & ded.cells.str.contains("P4_pmpct20_K2")
      & (ded.prob >= 0.126))
report("FILTER 2  P2_hr55_71_K2 & P4_pmpct20_K2 & prob>=0.126", f3)
