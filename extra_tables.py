#!/usr/bin/env python3
"""extra_tables.py -- two paper tables built from the evaluation outputs:
  (1) tab_main_results_30min.tex : Table-1-format results at the 30-min horizon (per-user Wilcoxon vs
      Bayesian Loop, same functions as evaluate.py), and
  (2) tab_instance_percentiles_body.tex: for the four instances in the qualitative figure
      (main_vis_figure/selection_cache/a_selected.pkl, written by qualitative_figure.py), each
      model's 4-h RMSE percentile within that context's quality-filtered pool (same filter as
      qualitative_figure.select_fig_a), plus the instance's own selection-covariate percentile.
Usage: python extra_tables.py      (run evaluate.py and qualitative_figure.py first)
Writes into <OUT_DIR>/tables/."""
import os, sys, argparse, pickle, numpy as np, pandas as pd
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(v,'2')
os.environ['JAX_PLATFORMS']='cpu'
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import config, eval_config
ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--out-dir',default=eval_config.out_dir_for()); ap.add_argument('--data-dir',default=None)
ap.add_argument('--only',choices=['30min','pct','both'],default='both'); A=ap.parse_args()
if A.data_dir is None: A.data_dir=config.data_dir()
sys.argv=[sys.argv[0],'--model','ar1_rls']   # evaluate.py's platform check -> CPU
import evaluate as ev
OUT=A.out_dir; TAB=os.path.join(OUT,'tables'); os.makedirs(TAB,exist_ok=True)
df=pd.read_pickle(os.path.join(OUT,'metrics_df.pkl'))
NS='\\text{ns}'
PREF=ev.DISPLAY_ORDER  # G L E C T N F A R
NAMES=['Bayesian Loop','LSTM-MC','LSTM-Ensemble','LSTM-Conformal','Transformer-MC','Transformer-Ensemble','Transformer-Conformal','Pure AR1','Adaptive AR1-RLS']

def tab30():
    d=df[df.horizon=='30 min']
    rows=[('RMSE (mg/dL)','RMSE','min',False,1),('CRPS (mg/dL)','CRPS','min',False,2),('Coverage (95\\% PI)','Coverage','max',True,1),
    ('PI Width (mg/dL)','Width','min',False,1),('Interval Score','IS','min',False,1),None,
    ('Net Trend Accuracy (\\%)','P_Net_Trend','max',True,1),('Trajectory Correlation (r)','P_Traj_Corr','max',False,2),('Mean Directional Bias (mg/dL)','P_Mean_Dir_Bias','abs',False,1),None,
    ('Hypo Sensitivity (30-min, \\%)','P_Hypo_Sens','max',True,1),('Hypo Specificity (30-min, \\%)','P_Hypo_Spec','max',True,1),None,
    ('Hyper Sensitivity (30-min, \\%)','P_Hyper_Sens','max',True,1),('Hyper Specificity (30-min, \\%)','P_Hyper_Spec','max',True,1),None,
    ('CEG Zone A (\\%)','P_CEG_A','max',False,1),('CEG Zone D (\\%)','P_CEG_D','min',False,2),('CEG Zone E (\\%)','P_CEG_E','min',False,2)]
    def bold(v,rule,dec):
        r=np.round(v,dec)
        if rule=='min': return r==np.nanmin(r)
        if rule=='max': return r==np.nanmax(r)
        a=np.abs(r); return a==np.nanmin(a)
    out=[]
    for row in rows:
        if row is None: out.append('\\midrule'); continue
        lab,col,rule,pct,dec=row; sc=100 if pct else 1
        paired=d[[f'G_{col}',f'L_{col}']].dropna()
        m=np.array([paired[f'G_{col}'].mean(),paired[f'L_{col}'].mean()]+[d[f'{p}_{col}'].dropna().mean() for p in PREF[2:]])*sc
        pv=[ev.wilcoxon_pval_peruser(d,f'G_{col}',f'{p}_{col}') for p in PREF[1:]]
        cells=[]
        for i,(x,b) in enumerate(zip(m,bold(m,rule,dec))):
            s=f'{x:.{dec}f}'; s=f'\\textbf{{{s}}}' if b else s
            if i>0:
                ps=ev.pval_str(pv[i-1]); ps=NS if ps=='ns' else ps; s+='$^{'+ps+'}$'
            cells.append(s)
        out.append('  '+lab+' & '+' & '.join(cells)+' \\\\')
    out.append('\\midrule')
    for metric in ['Hypo','Hyper']:
        st={p:ev._operating_point_stats(d[f'{p}_P_{metric}_Prob'].values,d[f'{p}_P_{metric}_Sens'].notna().astype(float).values) for p in PREF}
        for key,lab,rule in [('far90',f'{metric} False-Alarm Rate (Sens=0.90, 30-min, \\%)','min'),('ppv90',f'{metric} PPV (Sens=0.90, 30-min, \\%)','max'),('sensAtSpec90',f'{metric} Sensitivity (Spec=0.90, 30-min, \\%)','max')]:
            v=np.array([st[p][key] for p in PREF])*100
            cells=['N/R' if np.isnan(x) else (f'\\textbf{{{x:.1f}}}' if b else f'{x:.1f}') for x,b in zip(v,bold(v,rule,1))]
            out.append('  '+lab+' & '+' & '.join(cells)+' \\\\')
        if metric=='Hypo': out.append('\\midrule')
    hdr=' &\n'.join(['Metric']+[f'\\rotatebox{{90}}{{{n}}}' for n in NAMES])
    n_users=d['uid'].nunique(); n_inst=len(d[['uid','eid']].drop_duplicates())
    n_inst_tex=f'{n_inst:,}'.replace(',','{,}')
    cap=(r"\textbf{Aggregate test-set performance across all nine models at the 30-minute horizon.} Counterpart of Table~\ref{tab:main_results}, computed identically over the same $N="+str(n_users)+r"$ users and "+n_inst_tex+r" held-out instances, except that every metric uses the 30-minute forecast window; these values match the 30-minute points in Figure~\ref{fig:horizons}. Bold = best (ties are all bolded). Superscripts denote a paired Wilcoxon test against Bayesian Loop on per-user means ($N="+str(n_users)+r"$ users; $*p<0.05$, $**p<0.01$, $***p<0.001$, ns~=~not significant; Section~\ref{sec:significance-testing}). The False-Alarm Rate, PPV, and Sensitivity (Spec=0.90) rows are pooled-cohort operating points defined as in Table~\ref{tab:main_results}. Metric definitions are in Section~\ref{app:eval-framework}.")
    tex=("% Generated by extra_tables.py from metrics_df.pkl\n\\begin{table}[!tp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
         f"\\caption{{{cap}}}\n\\label{{tab:main_results_30min}}\n\\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{l*{{9}}{{r}}}}\n\\toprule\n{hdr} \\\\\n\\midrule\n"
         +'\n'.join(out)+"\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    open(os.path.join(TAB,'tab_main_results_30min.tex'),'w').write(tex); print('wrote',os.path.join(TAB,'tab_main_results_30min.tex'))

def tabpct():
    cache=os.path.join(OUT,'main_vis_figure','selection_cache')
    sel=pickle.load(open(os.path.join(cache,'a_selected.pkl'),'rb'))
    pool=pd.read_pickle(os.path.join(cache,'master_pool.pkl')); pool['eid']=pool['eid'].astype(str)
    def evsum(csv,cands,val):
        x=pd.read_csv(csv); x['_eid']=x['ex'].astype(float).astype(int).astype(str); c=next(c for c in cands if c in x.columns)
        x=x[x[c]>0]; return x.groupby('_eid')[c].sum().to_dict()
    carbs=evsum(os.path.join(A.data_dir,'meals.csv'),['carbs','meal_dose','MLDOSE'],None)
    bolus=evsum(os.path.join(A.data_dir,'bolus.csv'),['bolus','dose'],None)
    pool['total_carbs']=pool.eid.map(lambda e:carbs.get(e,0.0)); pool['total_bolus']=pool.eid.map(lambda e:bolus.get(e,0.0))
    cgm_pq=os.path.join(A.data_dir,'cgm.parquet')  # written by t1dexi-preprocessing; cgm.csv in older builds
    cg=(pd.read_parquet(cgm_pq,columns=['ex']) if os.path.exists(cgm_pq)
        else pd.read_csv(os.path.join(A.data_dir,'cgm.csv'),usecols=lambda c: c=='ex'))
    key=cg['ex'].astype(float).astype(int).astype(str)
    cnt=key.value_counts(); valid=set(cnt[cnt>=48].index)
    pool=pool[pool.eid.isin(valid)]
    med_g=pool.g_rmse.median()
    r240=df[df.horizon=='240 min'].copy(); r240['eid']=r240.eid.astype(str).str.replace(r'\.0$','',regex=True)
    r240=r240.set_index('eid')
    CTX=['MEAL','CARB_INSTANCE','CORRECTION','STABLE']; COV={'MEAL':'total_carbs','CARB_INSTANCE':'total_carbs','CORRECTION':'total_bolus','STABLE':'obs_std'}
    res={}; sizes={}; covp={}
    for ctx in CTX:
        sub=pool[pool.itype==ctx]
        q=sub[(sub.g_rmse<2.5*sub.l_rmse)&(sub.g_rmse<3*med_g)&(sub.obs_min>=55)&(sub.obs_max<=380)]
        sizes[ctx]=len(q); e=str(sel[ctx]['eid'])
        if e not in set(q.eid): print(f'NOTE: pinned {ctx} instance {e} is outside the new quality-filtered pool; percentiles still computed against that pool')
        cv=q[COV[ctx]].values; x=pool.loc[pool.eid==e,COV[ctx]].values[0]
        covp[ctx]=100*(np.mean(cv<x)+0.5*np.mean(cv==x))
        for p in PREF:
            v=r240.loc[q.eid.values,f'{p}_RMSE'].values; xi=float(r240.loc[e,f'{p}_RMSE'])
            res[(p,ctx)]=(100*(np.mean(v<xi)+0.5*np.mean(v==xi)), xi)
    lines=[f"  {n} & "+' & '.join(f"{res[(p,c)][0]:.1f} ({res[(p,c)][1]:.1f})" for c in CTX)+r' \\' for p,n in zip(PREF,NAMES)]
    def n_tex(n): return f'{n:,}'.replace(',','{,}')
    cap=(r"Where each of the four qualitative example instances shown in Figure~\ref{fig:contexts} actually falls in each model's 4-hour RMSE distribution. Each cell gives the instance's percentile rank in RMSE (0--100; higher = worse, i.e.\ a larger forecast error than more of the pool; 50 = exact median error) within the quality-filtered pool for that clinical context (Section~\ref{app:instance_selection}), with the raw RMSE (mg/dL) in parentheses. "
         "Pool sizes: Meal $n="+n_tex(sizes['MEAL'])+"$; Carb Instance $n="+n_tex(sizes['CARB_INSTANCE'])+"$; Correction $n="+n_tex(sizes['CORRECTION'])+"$; Stable $n="+n_tex(sizes['STABLE'])+"$. "
         r"Percentiles are computed independently per model and context; an instance need not be near the 50th percentile for every model simply because it was selected as the anchor-median instance for one context-specific covariate (Section~\ref{app:instance_selection}). The final row (``Selected instance's covariate percentile'') reports, independent of any model, where the shown instance itself falls (0--100; 50 = exact median) on the context-specific covariate it was selected on (total carbohydrates for Meal/Carb Instance, total bolus for Correction, glucose SD for Stable) within the same quality-filtered pool.")
    tex=("% Generated by extra_tables.py (instances from figure_selection.json)\n"
         "\\begin{table}[!htbp]\n\\centering\n\\caption{"+cap+"}\n\\label{tab:instance_percentiles}\n\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n  Model & Meal & Carb Instance & Correction & Stable \\\\\n\\midrule\n"
         +'\n'.join(lines)+"\n\\midrule\n  Selected instance's covariate percentile & "+' & '.join(f'{covp[c]:.1f}' for c in CTX)+" \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TAB,'tab_instance_percentiles_body.tex'),'w').write(tex)
    print('pool sizes',sizes); print('\n'.join(lines)); print('covariate pct', {c:round(covp[c],1) for c in CTX})
if A.only in ('30min','both'): tab30()
if A.only in ('pct','both'): tabpct()
