import pandas as pd, numpy as np, glob, os, warnings; warnings.filterwarnings('ignore')
FEE=0.07
R=pd.read_parquet("res_cache_5m_all.parquet")
BK=pd.read_parquet("book_5m.parquet"); BK["ts"]=pd.to_datetime(BK.mkt_open,utc=True); BK["asset"]=BK.asset.astype(str)
A5=BK[BK.elapsed_req==5][["asset","ts","au","ad"]].rename(columns={"ad":"ask_up","au":"ask_dn"})
SYM={"BTC":"btc","ETH":"eth","SOL":"sol","XRP":"xrp","DOGE":"doge","BNB":"bnb","HYPE":"hype"}
rows=[]
for f in sorted(glob.glob("port_logs/*.csv")):
    d=pd.read_csv(f,on_bad_lines="skip")
    d["stream"]=os.path.basename(f)[:-4]
    d["cell"]=d.get("strategy","?").astype(str)
    d["asset"]=d.symbol.str.split("/").str[0].str.upper().map(SYM)
    d["ts"]=pd.to_datetime(d.start_date,format="%m/%d/%y %H:%M",errors="coerce",utc=True)
    d["prob"]=pd.to_numeric(d.get("prob"),errors="coerce")
    d["conf"]=pd.to_numeric(d.get("confidence"),errors="coerce")
    d["side"]=np.where(d.forecast.astype(str).str.contains("⬆"),"UP",
              np.where(d.forecast.astype(str).str.contains("⬇"),"DOWN",None))
    rows.append(d[["stream","cell","asset","ts","side","prob","conf"]])
S=pd.concat(rows,ignore_index=True).dropna(subset=["ts","side","asset"])
S=S.merge(R,on=["asset","ts"],how="left").merge(A5,on=["asset","ts"],how="left")
S["win"]=np.where(S.side=="UP",S.y,1-S.y)
S["ask"]=np.where(S.side=="UP",S.ask_up,S.ask_dn)
res=S.dropna(subset=["y","ask"]).copy()
res["be"]=res.ask+FEE*res.ask*(1-res.ask)
res["pps"]=np.where(res.win==1,1-res.ask,-res.ask)-FEE*res.ask*(1-res.ask)
res["pnl5"]=res.pps*(5.0/res.ask)

# how much duplication is there on (asset, ts, side)?
dupe_counts = res.groupby(["asset","ts","side"]).size()
print("distinct (asset,ts,side) events:", dupe_counts.shape[0])
print("total rows (with duplicates across streams):", len(res))
print("rows appearing in >1 stream/cell:", (dupe_counts>1).sum(), "events covering", dupe_counts[dupe_counts>1].sum(), "rows")
print()
# check if prob is consistent across duplicate rows for same event
g = res.groupby(["asset","ts","side"])["prob"].nunique()
print("events where prob DIFFERS across duplicate rows:", (g>1).sum(), "/", len(g))

# inspect one dup example
ex = dupe_counts[dupe_counts>1].index[0]
print("\nexample duplicated event:", ex)
print(res[(res.asset==ex[0])&(res.ts==ex[1])&(res.side==ex[2])][["stream","cell","prob","conf","win","ask","pnl5"]])
res.to_parquet("res_flat.parquet")
