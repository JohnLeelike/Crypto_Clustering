import pandas as pd, numpy as np, glob, os, warnings, itertools; warnings.filterwarnings('ignore')
FEE=0.07
SYM={"BTC":"btc","ETH":"eth","SOL":"sol","XRP":"xrp","DOGE":"doge","BNB":"bnb","HYPE":"hype"}

R=pd.read_parquet("res_cache_5m_all.parquet")
BK=pd.read_parquet("book_5m.parquet"); BK["ts"]=pd.to_datetime(BK.mkt_open,utc=True); BK["asset"]=BK.asset.astype(str)
A5=BK[BK.elapsed_req==5][["asset","ts","au","ad"]].rename(columns={"ad":"ask_up","au":"ask_dn"})

files = sorted(glob.glob("port_logs/*.csv")) + sorted(glob.glob("backtest_data/new_streams/*.csv"))
print(f"loading {len(files)} files")
rows=[]
for f in files:
    d = pd.read_csv(f, on_bad_lines="skip")
    stream = os.path.basename(f).replace(".csv","")
    strat_col = "strategy" if "strategy" in d.columns else "_15"
    d["stream"]=stream
    d["cell"]=d[strat_col].astype(str)
    d["asset"]=d.symbol.str.split("/").str[0].str.upper().map(SYM)
    d["ts"]=pd.to_datetime(d.start_date, format="%m/%d/%y %H:%M", errors="coerce", utc=True)
    d["prob"]=pd.to_numeric(d.get("prob"), errors="coerce")
    d["side"]=np.where(d.forecast.astype(str).str.contains("⬆"),"UP",
              np.where(d.forecast.astype(str).str.contains("⬇"),"DOWN",None))
    rows.append(d[["stream","cell","asset","ts","side","prob"]])
S = pd.concat(rows, ignore_index=True).dropna(subset=["ts","side","asset"])
print(f"raw rows (pre-dedup): {len(S)}")

S = S.merge(R, on=["asset","ts"], how="left").merge(A5, on=["asset","ts"], how="left")
S["win"]=np.where(S.side=="UP", S.y, 1-S.y)
S["ask"]=np.where(S.side=="UP", S.ask_up, S.ask_dn)
res = S.dropna(subset=["y","ask"]).copy()
res["pps"]=np.where(res.win==1, 1-res.ask, -res.ask) - FEE*res.ask*(1-res.ask)
res["pnl5"]=res.pps*(5.0/res.ask)
print(f"resolved+priced raw rows: {len(res)}")

ded = (res.groupby(["asset","ts","side"])
          .agg(prob=("prob","first"), win=("win","first"), ask=("ask","first"), pnl5=("pnl5","first"),
               cells=("cell", lambda s: frozenset(s)), streams=("stream", lambda s: frozenset(s)))
          .reset_index())
N=len(ded)
print(f"DEDUPED unique events: {N}")
print(f"\nper-asset counts:\n{ded.asset.value_counts()}")
print(f"\noverall WR: {ded.win.mean()*100:.2f}%  total pnl@5: {ded.pnl5.sum():+.2f}")

ded["cells"]=ded.cells.apply(lambda s:"|".join(sorted(s))); ded["streams"]=ded.streams.apply(lambda s:"|".join(sorted(s))); ded.to_parquet("ded_v2.parquet")
res.to_parquet("res_v2_flat.parquet")
print("\nsaved ded_v2.parquet, res_v2_flat.parquet")
