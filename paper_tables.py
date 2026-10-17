#!/usr/bin/env python3
"""Paper-format main results table (tab_main_results.tex, 4-hour horizon, nine models)
and completeness-stratified table (tab_completeness_stratified.tex), built from
evaluate.py's metrics_df.pkl and completeness_analysis.py's stratified CSV.
The table preamble and caption (as in the manuscript) are below; only the numeric body is generated.

Usage: python paper_tables.py            (defaults to the main-analysis outputs)"""
import os, sys, re, argparse, numpy as np, pandas as pd
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(v,'2')
os.environ['JAX_PLATFORMS']='cpu'; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import config, eval_config
_OUT=eval_config.out_dir_for()
ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--metrics-pkl',default=os.path.join(_OUT,'metrics_df.pkl'))
ap.add_argument('--strat-csv',default=os.path.join(config.EVAL_RESULTS_DIR,'completeness_analysis','stratified_metrics_4h.csv'))
ap.add_argument('--out-dir',default=os.path.join(_OUT,'paper_tables'))
A=ap.parse_args(); sys.argv=[sys.argv[0],'--model','ar1_rls']  # evaluate.py's platform check -> CPU
import evaluate as ev
PREF=ev.DISPLAY_ORDER; NS='\\text{ns}'
NAMES=['Bayesian Loop','LSTM-MC','LSTM-Ensemble','LSTM-Conformal','Transformer-MC','Transformer-Ensemble','Transformer-Conformal','Pure AR1','Adaptive AR1-RLS']
os.makedirs(A.out_dir,exist_ok=True)
def fmt_cells(m,rule,dec,pv):
    r=np.round(m,dec)
    if rule=='abs': isb=np.abs(r)==np.nanmin(np.abs(r))
    elif rule=='min': isb=r==np.nanmin(r)
    else: isb=r==np.nanmax(r)
    out=[]
    for i,(x,b) in enumerate(zip(m,isb)):
        if np.isnan(x): out.append('N/R'); continue
        s=f'{x:.{dec}f}'; s=f'\\textbf{{{s}}}' if b else s
        if pv is not None and i>0:
            ps=ev.pval_str(pv[i-1]); s+='$^{'+(NS if ps=='ns' else ps)+'}$'
        out.append(s)
    return out
def table1(pkl):
    df=pd.read_pickle(pkl); d=df[df.horizon=='240 min']
    ROWS=[('RMSE (mg/dL)','RMSE','min',1,1),('CRPS (mg/dL)','CRPS','min',1,2),('Coverage (95\\% PI)','Coverage','max',100,1),('PI Width (mg/dL)','Width','min',1,1),('Interval Score','IS','min',1,1),None,
      ('Net Trend Accuracy (\\%)','P_Net_Trend','max',100,1),('Trajectory Correlation (r)','P_Traj_Corr','max',1,2),('Mean Directional Bias (mg/dL)','P_Mean_Dir_Bias','abs',1,1),None,
      ('Hypo Sensitivity (4-hr, \\%)','P_Hypo_Sens','max',100,1),('Hypo Specificity (4-hr, \\%)','P_Hypo_Spec','max',100,1),('Hypo Sensitivity (60-min, \\%)','P_Hypo_Sens_60min','max',100,1),('Hypo Specificity (60-min, \\%)','P_Hypo_Spec_60min','max',100,1),None,
      ('Hyper Sensitivity (4-hr, \\%)','P_Hyper_Sens','max',100,1),('Hyper Specificity (4-hr, \\%)','P_Hyper_Spec','max',100,1),('Hyper Sensitivity (60-min, \\%)','P_Hyper_Sens_60min','max',100,1),('Hyper Specificity (60-min, \\%)','P_Hyper_Spec_60min','max',100,1),None,
      ('CEG Zone A (\\%)','P_CEG_A','max',1,1),('CEG Zone D (\\%)','P_CEG_D','min',1,2),('CEG Zone E (\\%)','P_CEG_E','min',1,2)]
    lines=[]
    for r in ROWS:
        if r is None: lines.append('\\midrule'); continue
        lab,col,rule,sc,dec=r
        paired=d[[f'G_{col}',f'L_{col}']].dropna()
        m=np.array([paired[f'G_{col}'].mean(),paired[f'L_{col}'].mean()]+[d[f'{p}_{col}'].dropna().mean() for p in PREF[2:]])*sc
        pv=[ev.wilcoxon_pval_peruser(d,f'G_{col}',f'{p}_{col}') for p in PREF[1:]]
        lines.append('  '+lab+' & '+' & '.join(fmt_cells(m,rule,dec,pv))+' \\\\')
    for metric in ['Hypo','Hyper']:
        lines.append('\\midrule')
        st={p:ev._operating_point_stats(d[f'{p}_P_{metric}_Prob'].values,d[f'{p}_P_{metric}_Sens'].notna().astype(float).values) for p in PREF}
        for key,lab,rule in [('far90',f'{metric} False-Alarm Rate (Sens=0.90, 4-hr, \\%)','min'),('ppv90',f'{metric} PPV (Sens=0.90, 4-hr, \\%)','max'),('sensAtSpec90',f'{metric} Sensitivity (Spec=0.90, 4-hr, \\%)','max')]:
            v=np.array([st[p][key] for p in PREF])*100
            lines.append('  '+lab+' & '+' & '.join(fmt_cells(v,rule,1,None))+' \\\\')
    return lines
def latex_int(n):
    """Integer with LaTeX thousands separators, e.g. 59134 -> 59{,}134."""
    return f'{int(n):,}'.replace(',','{,}')
# Table preamble and caption; the generated rows replace the empty body between \midrule and \bottomrule.
TAB_MAIN_RESULTS=r'''\begin{table}[!tp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Aggregate test-set performance across all nine models (4-hour horizon).} Hypo/Hyper refer to hypoglycemia/hyperglycemia. Net Trend Accuracy, the continuous Hypo/Hyper Sensitivity/Specificity rows, CRPS (Continuous Ranked Probability Score), Coverage, PI (prediction interval) Width, and Interval Score are computed from the full posterior sample distribution; RMSE, Trajectory Correlation, Mean Directional Bias, and CEG (Clarke Error Grid) Zones use only the posterior mean trajectory. Bold = best (for Mean Directional Bias, a signed metric, best = smallest magnitude, i.e.\ closest to zero; for Coverage, best = highest). Superscripts denote a paired Wilcoxon test against Bayesian Loop on per-user means ($N=@N_USERS@$ users; $*p<0.05$, $**p<0.01$, $***p<0.001$, ns~=~not significant; Section~\ref{sec:significance-testing} explains why per-user aggregation is used); metric definitions are in Section~\ref{app:eval-framework}. Hypo/Hyper False-Alarm Rate (FAR: the fraction of negative (no-event) instances that would trigger a false alarm) and Positive Predictive Value (PPV: the fraction of raised alarms that are correct) are both evaluated at the most conservative threshold achieving Sensitivity~$\geq 0.90$; Sensitivity (Spec=0.90) is evaluated at the most liberal threshold achieving Specificity~$\geq 0.90$ instead; because the operating threshold is chosen on the pooled cohort, these rows are reported without significance tests. N/R = Specificity~$\geq 0.90$ is unreachable at any threshold for that model (Section~\ref{app:metrics}).}
\label{tab:main_results}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{l*{9}{r}}
\toprule
Metric &
\rotatebox{90}{Bayesian Loop} &
\rotatebox{90}{LSTM-MC} &
\rotatebox{90}{LSTM-Ensemble} &
\rotatebox{90}{LSTM-Conformal} &
\rotatebox{90}{Transformer-MC} &
\rotatebox{90}{Transformer-Ensemble} &
\rotatebox{90}{Transformer-Conformal} &
\rotatebox{90}{Pure AR1} &
\rotatebox{90}{Adaptive AR1-RLS} \\
\midrule
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
'''
TAB_COMPLETENESS_STRATIFIED=r'''\begin{table}[!htbp]
\centering
\small
\caption{4-hour RMSE and mean directional bias, stratified by the heuristic severity bin for likely missing or under-reported carbohydrates (defined in Section~\ref{app:completeness-methods}) within the retained test set (CLEAN $n=@N_CLEAN@$; MILD $n=@N_MILD@$; MODERATE $n=@N_MODERATE@$; SEVERE bin is empty by construction, $@N_SEVERE@/@N_TOTAL@$, and omitted). CLEAN and MILD are large, reliable bins; MODERATE is small and every model's estimate there is noisy (see Supplementary~\ref{app:completeness}).}
\label{tab:completeness_stratified}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{l*{6}{r}}
\toprule
   & \multicolumn{2}{c}{CLEAN} & \multicolumn{2}{c}{MILD} & \multicolumn{2}{c}{MODERATE} \\
  Model & RMSE & Bias & RMSE & Bias & RMSE & Bias \\
\midrule
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
'''
def splice(template,body,fill=None):
    t=template
    for k,v in (fill or {}).items(): t=t.replace(f'@{k}@',v)
    i=t.index('\\midrule',t.index('\\rotatebox{90}{Adaptive AR1-RLS}') if 'Adaptive AR1-RLS}' in t else t.index('Model & RMSE'))
    j=t.index('\\bottomrule')
    return t[:i]+'\\midrule\n'+'\n'.join(body)+'\n'+t[j:]
if A.metrics_pkl:
    _users=pd.read_pickle(A.metrics_pkl)['uid'].nunique()
    open(os.path.join(A.out_dir,'tab_main_results.tex'),'w').write(splice(TAB_MAIN_RESULTS,table1(A.metrics_pkl),{'N_USERS':str(_users)}))
    print('wrote tab_main_results.tex')
if A.strat_csv:
    s=pd.read_csv(A.strat_csv); lab=dict(zip(['Bayesian Loop','LSTM-MC','LSTM-Ensemble','LSTM-Conformal','Transformer-MC','Transformer-Ensemble','Transformer-Conformal','Pure AR1','Adaptive AR1-RLS'],NAMES))
    body=[]
    for n in NAMES:
        g=s[s.model==n].set_index('severity')
        cells=[]
        for sev in ['CLEAN','MILD','MODERATE']: cells+=[f"{g.loc[sev,'RMSE']:.1f}",f"{g.loc[sev,'P_Mean_Dir_Bias']:.1f}"]
        body.append(f'  {n} & '+' & '.join(cells)+' \\\\')
    t=TAB_COMPLETENESS_STRATIFIED
    nmap={sev:int(s[(s.severity==sev)].n.iloc[0]) for sev in ['CLEAN','MILD','MODERATE','SEVERE']}
    for sev,n in nmap.items(): t=t.replace(f'@N_{sev}@',latex_int(n))
    t=t.replace('@N_TOTAL@',latex_int(sum(nmap.values())))
    i=t.index('\\midrule'); j=t.index('\\bottomrule')
    out=t[:i]+'\\midrule\n'+'\n'.join(body)+'\n'+t[j:]
    open(os.path.join(A.out_dir,'tab_completeness_stratified.tex'),'w').write(out); print('wrote tab_completeness_stratified.tex; n=',nmap)
