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
    d["side"]=np.where(d.forecast.astype(str).str.contains("⬆"),"UP",
              np.where(d.forecast.astype(str).str.contains("⬇"),"DOWN",None))
    rows.append(d[["stream","cell","asset","ts","side"]])
S=pd.concat(rows,ignore_index=True).dropna(subset=["ts","side","asset"])
print(f"portfolio signals: {len(S):,}   {S.ts.min()} -> {S.ts.max()}")
print("\nper-stream span:")
g=S.groupby("stream").agg(n=("ts","size"),first=("ts","min"),last=("ts","max"))
g["days"]=((g["last"]-g["first"]).dt.total_seconds()/86400).round(1); print(g.to_string())
S=S.merge(R,on=["asset","ts"],how="left").merge(A5,on=["asset","ts"],how="left")
S["win"]=np.where(S.side=="UP",S.y,1-S.y)
S["ask"]=np.where(S.side=="UP",S.ask_up,S.ask_dn)
res=S.dropna(subset=["y","ask"]).copy()
print(f"\nresolved AND priced at the real 5s ask: {len(res):,}")
res["be"]=res.ask+FEE*res.ask*(1-res.ask)
res["pps"]=np.where(res.win==1,1-res.ask,-res.ask)-FEE*res.ask*(1-res.ask)
res["pnl5"]=res.pps*(5.0/res.ask)
NOW=res.ts.max()
for days in (7,14):
    sub=res[res.ts>=NOW-pd.Timedelta(days=days)]
    print("\n"+"="*112); print(f"PORTFOLIOS — LAST {days}d, stream x cell x side (n>=5), by WR  [** >=75% *** >=90%]"); print("="*112)
    gg=sub.groupby(["stream","cell","side"]).agg(n=("win","size"),W=("win","sum"),WR=("win","mean"),
        ask=("ask","mean"),be=("be","mean"),pnl5=("pnl5","sum")).reset_index()
    gg=gg[gg.n>=5]
    gg["edge"]=(gg.WR-gg.be)*100; gg["se"]=np.sqrt(gg.WR*(1-gg.WR)/gg.n)*100
    gg["t"]=gg.edge/gg.se.replace(0,np.nan); gg["WR"]*=100
    print(f"  {'stream':<16}{'cell':<22}{'side':>5}{'n':>5}{'W':>4}{'WR':>8}{'ask':>7}{'break':>8}{'edge':>9}{'t':>7}{'$@5':>9}")
    for _,r in gg.sort_values("WR",ascending=False).head(22).iterrows():
        mk=" ***" if r.WR>=90 else (" **" if r.WR>=75 else "")
        print(f"  {r.stream:<16}{r.cell[:21]:<22}{r.side:>5}{r.n:>5}{int(r.W):>4}{r.WR:>7.2f}%{r.ask:>7.3f}"
              f"{r.be*100:>7.2f}%{r.edge:>+8.2f}p{r.t:>+7.2f}{r.pnl5:>+9.2f}{mk}")
    print(f"\n  cells n>=5: {len(gg)}   >=75% WR: {len(gg[gg.WR>=75])}   >=90%: {len(gg[gg.WR>=90])}"
          f"   >=75% AND positive edge: {len(gg[(gg.WR>=75)&(gg.edge>0)])}")
print("\n"+"="*112); print("POOLED BY PORTFOLIO, last 14d"); print("="*112)
sub=res[res.ts>=NOW-pd.Timedelta(days=14)]
g=sub.groupby("stream").agg(n=("win","size"),W=("win","sum"),WR=("win","mean"),ask=("ask","mean"),
    be=("be","mean"),pnl5=("pnl5","sum")).reset_index()
g["edge"]=(g.WR-g.be)*100; g["WR"]*=100
g["se"]=np.sqrt(g.WR/100*(1-g.WR/100)/g.n)*100; g["t"]=g.edge/g.se
print(f"  {'portfolio':<18}{'n':>5}{'W':>4}{'WR':>8}{'ask':>7}{'break':>8}{'edge':>9}{'t':>7}{'$@5':>9}")
for _,r in g.sort_values("WR",ascending=False).iterrows():
    print(f"  {r.stream:<18}{r.n:>5}{int(r.W):>4}{r.WR:>7.2f}%{r.ask:>7.3f}{r.be*100:>7.2f}%{r.edge:>+8.2f}p{r.t:>+7.2f}{r.pnl5:>+9.2f}")
