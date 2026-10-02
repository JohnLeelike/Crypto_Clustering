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

print("="*110)
print("PER-STREAM BREAKDOWN  (resolved+priced only)")
print("="*110)
g=res.groupby("stream").agg(n=("win","size"),W=("win","sum"),WR=("win","mean"),ask=("ask","mean"),
    be=("be","mean"),pnl=("pnl5","sum")).reset_index()
g["edge"]=(g.WR-g.be)*100; g["WR"]*=100; g["roi"]=g.pnl/(5*g.n)*100
g["se"]=np.sqrt(g.WR/100*(1-g.WR/100)/g.n)*100; g["t"]=g.edge/g.se
print(f"  {'stream':<16}{'n':>5}{'W':>4}{'WR':>8}{'ask':>7}{'break':>8}{'edge':>9}{'t':>7}{'pnl@5':>10}{'ROI%':>8}")
for _,r in g.sort_values("pnl",ascending=False).iterrows():
    print(f"  {r.stream:<16}{r.n:>5}{int(r.W):>4}{r.WR:>7.2f}%{r.ask:>7.3f}{r.be*100:>7.2f}%{r.edge:>+8.2f}p{r.t:>+7.2f}{r.pnl:>+10.2f}{r.roi:>+7.2f}%")
print(f"\n  TOTAL (raw pool, overlaps incl.): n={len(res)} pnl={res.pnl5.sum():+.2f} roi={res.pnl5.sum()/(5*len(res))*100:+.2f}%")

res.to_parquet("res_flat.parquet")
print("\nsaved res_flat.parquet for sweep")
