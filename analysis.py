# -*- coding: utf-8 -*-
"""
analysis.py — REAL analysis for
"Mobile Money, Financial Inclusion and Clean-Energy Access in Emerging
Economies"

Reads data/data_final.xlsx (real 38-country panel, 2014-2022). Reproduces every
table/figure. No synthetic data.

DV  : renew_energy (renewable energy consumption %, green-energy access proxy)
Key : mobile_subs (mobile-money enabler), account_own (financial inclusion),
      digital_incl (index)
X   : internet_users, electricity_access, urban_pop, ln_gdp_pc, inflation
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PAPER=Path(__file__).parent.parent; DATA=PAPER/"data"/"data_final.xlsx"
TAB=PAPER/"submission"/"tables"; FIG=PAPER/"submission"/"figures"
TAB.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
RJSON=PAPER/"data"/"results_econ.json"

CTRL=["internet_users","electricity_access","urban_pop","ln_gdp_pc","inflation"]
def star(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
def fmt(res,keep):
    o={k:dict(coef=round(float(res.params[k]),4),se=round(float(res.bse[k]),4),p=round(float(res.pvalues[k]),4)) for k in keep if k in res.params.index}
    o["_n"]=int(res.nobs); o["_r2"]=round(float(res.rsquared),3); return o

def main():
    full=pd.read_excel(DATA,sheet_name="main")
    df=full.dropna(subset=["renew_energy","mobile_subs"]+CTRL).copy()
    ctrl="+".join(CTRL); R={}

    m1=smf.ols(f"renew_energy ~ mobile_subs + {ctrl}", df).fit(cov_type="cluster",cov_kwds={"groups":df.iso3})
    m2=smf.ols(f"renew_energy ~ mobile_subs + {ctrl} + C(iso3)", df).fit(cov_type="cluster",cov_kwds={"groups":df.iso3})
    m3=smf.ols(f"renew_energy ~ mobile_subs + {ctrl} + C(iso3) + C(year)", df).fit(cov_type="cluster",cov_kwds={"groups":df.iso3})
    # account ownership specification (subset with Findex)
    da=full.dropna(subset=["renew_energy","account_own"]+CTRL).copy()
    m4=smf.ols(f"renew_energy ~ account_own + {ctrl} + C(iso3)", da).fit(cov_type="cluster",cov_kwds={"groups":da.iso3})
    R["M1_pooled"]=fmt(m1,["mobile_subs"]); R["M2_countryFE"]=fmt(m2,["mobile_subs"])
    R["M3_twowayFE"]=fmt(m3,["mobile_subs"]); R["M4_account_FE"]=fmt(m4,["account_own"])

    # clean cooking as alt DV
    dc=full.dropna(subset=["clean_cooking","mobile_subs"]+CTRL).copy()
    m5=smf.ols(f"clean_cooking ~ mobile_subs + {ctrl} + C(iso3)", dc).fit(cov_type="cluster",cov_kwds={"groups":dc.iso3})
    R["M5_cleancook_FE"]=fmt(m5,["mobile_subs"])

    # ---- Table 1 descriptives ----
    dv=["renew_energy","clean_cooking","mobile_subs","account_own","internet_users","electricity_access","urban_pop","gdp_pc","inflation"]
    t1=full[dv].describe().T[["count","mean","std","min","50%","max"]]; t1.columns=["N","Mean","SD","Min","Median","Max"]
    t1.round(3).to_excel(TAB/"table1_descriptive.xlsx")
    # ---- Table 2 main regressions ----
    rows=[]
    for lab,res,key in [("Pooled OLS",m1,"mobile_subs"),("Country FE",m2,"mobile_subs"),
                        ("Country+Year FE",m3,"mobile_subs"),("Account own + FE",m4,"account_own"),
                        ("Clean cooking DV + FE",m5,"mobile_subs")]:
        rows.append({"Model":lab,"regressor":key,"coef":round(res.params[key],4),
                     "SE":round(res.bse[key],4),"p":round(res.pvalues[key],4),
                     "sig":star(res.pvalues[key]),"N":int(res.nobs),"R2":round(res.rsquared,3)})
    pd.DataFrame(rows).to_excel(TAB/"table2_regressions.xlsx",index=False)
    with open(TAB/"table2_full_summary.txt","w",encoding="utf-8") as f:
        for lab,res in [("Pooled",m1),("CountryFE",m2),("TwoWayFE",m3),("AccountFE",m4),("CleanCookFE",m5)]:
            f.write(f"\n{'='*60}\n{lab}\n{'='*60}\n{res.summary().as_text()}\n")
    # ---- Table 3 heterogeneity: split by electricity access median ----
    med=df["electricity_access"].median()
    het=[]
    for lab,sub in [("Low electricity access",df[df.electricity_access<med]),
                    ("High electricity access",df[df.electricity_access>=med])]:
        if len(sub)>20:
            mm=smf.ols(f"renew_energy ~ mobile_subs + {ctrl} + C(iso3)", sub).fit(cov_type="cluster",cov_kwds={"groups":sub.iso3})
            het.append({"subgroup":lab,"mobile_subs coef":round(mm.params['mobile_subs'],4),
                        "p":round(mm.pvalues['mobile_subs'],4),"sig":star(mm.pvalues['mobile_subs']),"N":int(mm.nobs)})
    pd.DataFrame(het).to_excel(TAB/"table3_heterogeneity.xlsx",index=False)
    # ---- Table 4 robustness: drop COVID ----
    dx=df[~df.year.isin([2020,2021])]
    mr=smf.ols(f"renew_energy ~ mobile_subs + {ctrl} + C(iso3)", dx).fit(cov_type="cluster",cov_kwds={"groups":dx.iso3})
    pd.DataFrame([{"spec":"ex-COVID","mobile_subs coef":round(mr.params['mobile_subs'],4),
                   "p":round(mr.pvalues['mobile_subs'],4),"sig":star(mr.pvalues['mobile_subs']),"N":int(mr.nobs)}]).to_excel(TAB/"table4_robustness.xlsx",index=False)
    R["M6_exCOVID"]=fmt(mr,["mobile_subs"])

    # ---- Figures ----
    plt.figure(figsize=(7,5))
    sc=plt.scatter(df.mobile_subs,df.renew_energy,c=df.ln_gdp_pc,cmap="viridis",s=30,alpha=.7)
    plt.colorbar(sc,label="ln GDP per capita")
    b,a=np.polyfit(df.mobile_subs,df.renew_energy,1); xs=np.linspace(df.mobile_subs.min(),df.mobile_subs.max(),40)
    plt.plot(xs,a+b*xs,"r--",lw=2,label=f"slope={b:.3f}")
    plt.xlabel("Mobile subscriptions per 100"); plt.ylabel("Renewable energy consumption (%)")
    plt.title("Mobile penetration and clean-energy access — 38 EM (real)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG/"figure1_mobile_renew.pdf"); plt.savefig(FIG/"figure1_mobile_renew.tiff",dpi=300); plt.close()

    plt.figure(figsize=(7,5))
    da2=full.dropna(subset=["account_own","renew_energy"])
    plt.scatter(da2.account_own,da2.renew_energy,s=30,alpha=.6,color="#4C78A8")
    if len(da2)>5:
        b,a=np.polyfit(da2.account_own,da2.renew_energy,1); xs=np.linspace(da2.account_own.min(),da2.account_own.max(),40)
        plt.plot(xs,a+b*xs,"r--",lw=2,label=f"slope={b:.3f}"); plt.legend()
    plt.xlabel("Account ownership (% age 15+)"); plt.ylabel("Renewable energy consumption (%)")
    plt.title("Financial inclusion and clean-energy access (real)")
    plt.tight_layout(); plt.savefig(FIG/"figure2_account_renew.pdf"); plt.savefig(FIG/"figure2_account_renew.tiff",dpi=300); plt.close()

    plt.figure(figsize=(8,5))
    top=full.groupby("iso3")["renew_energy"].mean().sort_values(ascending=False).head(15).index
    sub=full[full.iso3.isin(top)]
    sub.boxplot(column="renew_energy",by="iso3",grid=False,ax=plt.gca())
    plt.suptitle(""); plt.title("Renewable energy consumption by country (top 15, real)")
    plt.xlabel("Country"); plt.ylabel("Renewable energy (%)"); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(FIG/"figure3_renew_by_country.pdf"); plt.savefig(FIG/"figure3_renew_by_country.tiff",dpi=300); plt.close()

    json.dump(R,open(RJSON,"w",encoding="utf-8"),indent=2)
    print("REAL RESULTS — Mobile Money & Clean-Energy Access (38 EM), N =",len(df))
    for m,l in [("M1_pooled","Pooled"),("M2_countryFE","CountryFE"),("M3_twowayFE","TwoWayFE"),("M6_exCOVID","ex-COVID")]:
        r=R[m]["mobile_subs"]; print(f"  {l:12s} mobile_subs={r['coef']:+.4f} (p={r['p']}) {star(r['p'])}")
    r=R["M4_account_FE"]["account_own"]; print(f"  account_own (FE) = {r['coef']:+.4f} (p={r['p']}) {star(r['p'])}")

if __name__=="__main__": main()
