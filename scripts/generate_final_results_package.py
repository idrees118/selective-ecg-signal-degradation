#!/usr/bin/env python3
from __future__ import annotations
import ast, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_prediction import grouped_risk_coverage, midrank_ecdf

SEEDS = (20260826, 20260827, 20260828)
FINAL_DIR = PROJECT_ROOT / "results/final_evaluation/v1/test"
FINAL_SUMMARY = FINAL_DIR / "summary.json"
CONDITION_TABLE = FINAL_DIR / "condition_metrics.csv"
L_TEST = FINAL_DIR / "test_annotation_confidence.csv"
COMBO_REF = PROJECT_ROOT / "results/selective_prediction_waveform_q/v1/validation/combination_reference.npz"
BOOT_DIR = PROJECT_ROOT / "results/bootstrap/v1/final_test"
BOOT_TABLE = BOOT_DIR / "primary_aurc_bootstrap_summary.csv"
BOOT_SUMMARY = BOOT_DIR / "summary.json"
REPORT_DIR = PROJECT_ROOT / "results/reporting/v1"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = PROJECT_ROOT / "figures/reporting/v1"
SUMMARY_OUT = REPORT_DIR / "FINAL_RESULTS_SUMMARY.json"

METHODS = ("confidence","U","L","Q_signal","L+U","L+Q_signal","Q_signal+U","L+Q_signal+U")
PRIMARY = ("confidence","U","L+U","L+Q_signal+U")
SEV = {"mild":1,"moderate":2,"severe":3}
CORRUPTIONS = ("baseline_wander","additive_white_noise","amplitude_clipping","lead_masking")

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path} must contain an object")
    return d

def parse_ci(v):
    x = ast.literal_eval(v) if isinstance(v, str) else v
    if not isinstance(x,(list,tuple)) or len(x)!=2:
        raise ValueError(f"Invalid CI {v!r}")
    lo, hi = float(x[0]), float(x[1])
    if not np.isfinite([lo,hi]).all() or lo>hi:
        raise ValueError(f"Invalid CI {v!r}")
    return lo, hi

def save_fig(fig, stem):
    fig.savefig(FIG_DIR/f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def seed_table():
    rows=[]
    for seed in SEEDS:
        p=PROJECT_ROOT/f"results/baseline/v1/seed_{seed}/run_metadata.json"
        d=load_json(p)
        if int(d["seed"])!=seed or int(d["test_records_used"])!=0:
            raise ValueError(f"Bad metadata for seed {seed}")
        best=float(d["best_validation_macro_auroc"])
        rep=float(d["reproduced_validation_macro_auroc"])
        if not np.isclose(best,rep,rtol=0,atol=1e-15):
            raise ValueError(f"Seed {seed} checkpoint mismatch")
        rows.append({
            "seed":seed,
            "best_epoch":int(d["best_epoch"]),
            "epochs_completed":int(d["epochs_completed"]),
            "best_validation_macro_auroc":best,
            "reproduced_validation_macro_auroc":rep,
            "test_records_used":int(d["test_records_used"]),
            "best_checkpoint_sha256":str(d["best_checkpoint_sha256"]),
        })
    df=pd.DataFrame(rows)
    v=df["best_validation_macro_auroc"].to_numpy(float)
    stats={
        "n_seeds":len(v),
        "mean_validation_macro_auroc":float(v.mean()),
        "sample_sd_validation_macro_auroc":float(v.std(ddof=1)),
        "min_validation_macro_auroc":float(v.min()),
        "max_validation_macro_auroc":float(v.max()),
        "range_validation_macro_auroc":float(np.ptp(v)),
    }
    return df, stats

def selectors(condition, L, ref):
    d=np.load(FINAL_DIR/condition/"results.npz", allow_pickle=False)
    loss=np.asarray(d["hamming_error"],float)
    conf=np.asarray(d["confidence"],float)
    Q=np.asarray(d["Q_signal"],float)
    U=np.asarray(d["U"],float)
    Lp=midrank_ecdf(np.asarray(ref["L_badness_reference"],float),1-L)
    Qp=midrank_ecdf(np.asarray(ref["Q_signal_badness_reference"],float),1-Q)
    Up=midrank_ecdf(np.asarray(ref["U_badness_reference"],float),U)
    return loss,{
        "confidence":conf,
        "U":-U,
        "L+U":-(Lp+Up)/2,
        "L+Q_signal+U":-(Lp+Qp+Up)/3,
    }

def main():
    TABLE_DIR.mkdir(parents=True,exist_ok=True)
    FIG_DIR.mkdir(parents=True,exist_ok=True)

    final=load_json(FINAL_SUMMARY)
    boot=load_json(BOOT_SUMMARY)
    if final["status"]!="FROZEN" or boot["status"]!="FROZEN":
        raise ValueError("Frozen source artifacts required")

    cond=pd.read_csv(CONDITION_TABLE,float_precision="round_trip")
    bt=pd.read_csv(BOOT_TABLE,float_precision="round_trip")
    if cond.shape!=(13,19): raise ValueError(f"Condition shape {cond.shape}")
    if bt.shape!=(39,9): raise ValueError(f"Bootstrap shape {bt.shape}")

    outputs={}
    seeds, seed_stats=seed_table()
    p=TABLE_DIR/"seed_reproducibility.csv"; seeds.to_csv(p,index=False,float_format="%.17g"); outputs["seed_reproducibility"]=p

    clean=cond.loc[cond["condition"].eq("clean")]
    if len(clean)!=1: raise ValueError("Expected one clean row")
    c=clean.iloc[0]
    clean_overall=pd.DataFrame([{
        "records":int(final["records"]),
        "patients":int(final["patients"]),
        "macro_auroc":float(c["macro_auroc"]),
        "macro_average_precision":float(c["macro_average_precision"]),
        "macro_f1":float(c["macro_f1"]),
        "full_hamming_risk":float(c["full_hamming_risk"]),
        "mean_confidence":float(c["mean_confidence"]),
        "mean_L":float(c["mean_L"]),
        "mean_Q_signal":float(c["mean_Q_signal"]),
        "mean_U":float(c["mean_U"]),
    }])
    p=TABLE_DIR/"clean_test_overall.csv"; clean_overall.to_csv(p,index=False,float_format="%.17g"); outputs["clean_test_overall"]=p

    cw=final["condition_results"]["clean"]["classification"]["classwise"]
    cwdf=pd.DataFrame([{"class":k,"auroc":float(v["auroc"]),"average_precision":float(v["average_precision"]),"f1":float(v["f1"])} for k,v in cw.items()])
    p=TABLE_DIR/"clean_test_classwise.csv"; cwdf.to_csv(p,index=False,float_format="%.17g"); outputs["clean_test_classwise"]=p

    corr=cond.loc[~cond["condition"].eq("clean")].copy()
    for m in ("macro_auroc","macro_average_precision","macro_f1","full_hamming_risk","mean_confidence","mean_Q_signal","mean_U"):
        corr[f"delta_{m}_vs_clean"]=corr[m]-float(c[m])
    p=TABLE_DIR/"corruption_test_performance.csv"; corr.to_csv(p,index=False,float_format="%.17g"); outputs["corruption_test_performance"]=p

    rows=[]
    for _,r in cond.iterrows():
        for m in METHODS:
            rows.append({"condition":r["condition"],"corruption_type":r["corruption_type"],"severity":r["severity"],"method":m,"aurc":float(r[f"aurc_{m}"])})
    aurc=pd.DataFrame(rows)
    p=TABLE_DIR/"selective_aurc_all_conditions.csv"; aurc.to_csv(p,index=False,float_format="%.17g"); outputs["selective_aurc_all_conditions"]=p

    b=bt.copy()
    ci=np.asarray([parse_ci(v) for v in b["ci_95_percentile"]],float)
    b["ci_lower"]=ci[:,0]; b["ci_upper"]=ci[:,1]
    b["confidence_better_supported"]=b["ci_lower"]>0
    b["alternative_better_supported"]=b["ci_upper"]<0
    p=TABLE_DIR/"bootstrap_primary_aurc_comparisons.csv"; b.to_csv(p,index=False,float_format="%.17g"); outputs["bootstrap_primary_aurc_comparisons"]=p

    winners=[]
    for name,g in aurc.groupby("condition",sort=False):
        g=g.sort_values("aurc")
        winners.append({
            "condition":name,
            "lowest_aurc_method":g.iloc[0]["method"],
            "lowest_aurc":float(g.iloc[0]["aurc"]),
            "second_lowest_aurc_method":g.iloc[1]["method"],
            "second_lowest_aurc":float(g.iloc[1]["aurc"]),
        })
    wdf=pd.DataFrame(winners)
    p=TABLE_DIR/"selective_aurc_rank_summary.csv"; wdf.to_csv(p,index=False,float_format="%.17g"); outputs["selective_aurc_rank_summary"]=p

    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(seeds["seed"].astype(str),seeds["best_validation_macro_auroc"],marker="o")
    ax.set_xlabel("Training seed"); ax.set_ylabel("Best validation macro-AUROC")
    ax.set_title("Baseline reproducibility across three seeds"); ax.grid(True,alpha=.25)
    save_fig(fig,"seed_reproducibility")

    display={"baseline_wander":"Baseline wander","additive_white_noise":"White noise","amplitude_clipping":"Amplitude clipping","lead_masking":"Lead masking"}
    metrics=[
        ("macro_auroc","Macro-AUROC","corruption_macro_auroc"),
        ("full_hamming_risk","Full-coverage Hamming risk","corruption_hamming_risk"),
        ("mean_confidence","Mean ordinary confidence","corruption_mean_confidence"),
        ("mean_U","Mean MC-dropout mutual information","corruption_mean_uncertainty"),
        ("mean_Q_signal","Mean waveform quality score","corruption_mean_q_signal"),
    ]
    for metric,ylabel,stem in metrics:
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        for kind in CORRUPTIONS:
            g=corr.loc[corr["corruption_type"].eq(kind)].copy()
            g["x"]=g["severity"].map(SEV); g=g.sort_values("x")
            ax.plot(g["x"],g[metric],marker="o",label=display[kind])
        ax.set_xticks([1,2,3],["Mild","Moderate","Severe"])
        ax.set_xlabel("Corruption severity"); ax.set_ylabel(ylabel); ax.legend(frameon=False); ax.grid(True,alpha=.25)
        save_fig(fig,stem)

    order=["clean","baseline_wander_mild","baseline_wander_moderate","baseline_wander_severe","additive_white_noise_mild","additive_white_noise_moderate","additive_white_noise_severe","amplitude_clipping_mild","amplitude_clipping_moderate","amplitude_clipping_severe","lead_masking_mild","lead_masking_moderate","lead_masking_severe"]
    short={"clean":"Clean","baseline_wander_mild":"BW-Mild","baseline_wander_moderate":"BW-Mod","baseline_wander_severe":"BW-Severe","additive_white_noise_mild":"WN-Mild","additive_white_noise_moderate":"WN-Mod","additive_white_noise_severe":"WN-Severe","amplitude_clipping_mild":"Clip-Mild","amplitude_clipping_moderate":"Clip-Mod","amplitude_clipping_severe":"Clip-Severe","lead_masking_mild":"Lead-Mild","lead_masking_moderate":"Lead-Mod","lead_masking_severe":"Lead-Severe"}
    stems={"L+U_vs_confidence":"bootstrap_delta_lu_vs_confidence","U_vs_confidence":"bootstrap_delta_u_vs_confidence","L+Q_signal+U_vs_confidence":"bootstrap_delta_lqu_vs_confidence"}
    for comp,stem in stems.items():
        g=bt.loc[bt["comparison"].eq(comp)].set_index("condition").loc[order].reset_index()
        ci=np.asarray([parse_ci(v) for v in g["ci_95_percentile"]],float)
        point=g["point_delta"].to_numpy(float); yerr=np.vstack((point-ci[:,0],ci[:,1]-point)); x=np.arange(len(g))
        fig,ax=plt.subplots(figsize=(11,5))
        ax.errorbar(x,point,yerr=yerr,fmt="o",capsize=3); ax.axhline(0,linewidth=1)
        ax.set_xticks(x,[short[v] for v in g["condition"]],rotation=45,ha="right")
        ax.set_ylabel("ΔAURC (method − confidence)"); ax.set_xlabel("Condition"); ax.set_title(comp.replace("_"," ")); ax.grid(True,axis="y",alpha=.25)
        save_fig(fig,stem)

    L=pd.read_csv(L_TEST,float_precision="round_trip")["label_confidence"].to_numpy(float)
    ref=np.load(COMBO_REF,allow_pickle=False)
    for name in ("clean","baseline_wander_severe","additive_white_noise_severe","amplitude_clipping_severe","lead_masking_severe"):
        loss,s=selectors(name,L,ref)
        fig,ax=plt.subplots(figsize=(7,5))
        for m in PRIMARY:
            curve=grouped_risk_coverage(s[m],loss)
            ax.plot(curve["coverage"],curve["risk"],label=m)
        ax.set_xlabel("Coverage"); ax.set_ylabel("Hamming risk"); ax.set_title(f"Risk–coverage: {name.replace('_',' ')}"); ax.set_xlim(0,1); ax.legend(frameon=False); ax.grid(True,alpha=.25)
        save_fig(fig,f"risk_coverage_{name}")

    monotonic={}
    for kind in CORRUPTIONS:
        g=corr.loc[corr["corruption_type"].eq(kind)].copy()
        g["x"]=g["severity"].map(SEV); g=g.sort_values("x")
        monotonic[kind]={
            "risk_nondecreasing":bool(np.all(np.diff(g["full_hamming_risk"].to_numpy(float))>=0)),
            "confidence_nonincreasing":bool(np.all(np.diff(g["mean_confidence"].to_numpy(float))<=0)),
            "Q_signal_nonincreasing":bool(np.all(np.diff(g["mean_Q_signal"].to_numpy(float))<=0)),
            "U_nondecreasing":bool(np.all(np.diff(g["mean_U"].to_numpy(float))>=0)),
        }

    lower=ci_lower=np.asarray([parse_ci(v)[0] for v in bt["ci_95_percentile"]],float)
    upper=np.asarray([parse_ci(v)[1] for v in bt["ci_95_percentile"]],float)
    confidence_better=int(np.sum(lower>0)); alternative_better=int(np.sum(upper<0)); inconclusive=int(len(bt)-confidence_better-alternative_better)

    winner_map={}
    for _,r in wdf.iterrows():
        winner_map[str(r["condition"])]={"method":str(r["lowest_aurc_method"]),"aurc":float(r["lowest_aurc"])}

    generated={}
    for p in outputs.values():
        generated[str(p.relative_to(PROJECT_ROOT))]=sha256_file(p)
    for p in sorted(FIG_DIR.glob("*")):
        if p.is_file():
            generated[str(p.relative_to(PROJECT_ROOT))]=sha256_file(p)

    report={
        "reporting_id":"ptbxl_final_reporting_v1",
        "source_status":{"final_test":final["status"],"bootstrap":boot["status"]},
        "records":int(final["records"]),
        "patients":int(final["patients"]),
        "clean_test":{
            "macro_auroc":float(c["macro_auroc"]),
            "macro_average_precision":float(c["macro_average_precision"]),
            "macro_f1":float(c["macro_f1"]),
            "full_hamming_risk":float(c["full_hamming_risk"]),
            "aurc_confidence":float(c["aurc_confidence"]),
            "aurc_U":float(c["aurc_U"]),
            "aurc_L_plus_U":float(c["aurc_L+U"]),
            "aurc_L_plus_Q_signal_plus_U":float(c["aurc_L+Q_signal+U"]),
        },
        "three_seed_reproducibility":seed_stats,
        "bootstrap_primary_comparisons":{
            "total":int(len(bt)),
            "confidence_lower_aurc_supported":confidence_better,
            "alternative_lower_aurc_supported":alternative_better,
            "inconclusive":inconclusive,
        },
        "lowest_point_aurc_method_by_condition":winner_map,
        "test_corruption_monotonicity":monotonic,
        "interpretation_constraints":{
            "L":"Retrospective annotation-aware confidence/ambiguity; not available for a truly unlabeled deployment ECG.",
            "Q_signal":"Study-specific training-referenced waveform quality score; not a clinically calibrated universal SQI.",
            "U":"Mean classwise MC-dropout mutual information from the two existing head dropout modules.",
        },
        "generated_artifact_sha256":generated,
        "status":"GENERATED_FROM_FROZEN_RESULTS",
    }
    with SUMMARY_OUT.open("w",encoding="utf-8") as f:
        json.dump(report,f,indent=2,sort_keys=True); f.write("\n")

    print("FINAL REPORTING PACKAGE GENERATED")
    print("Tables:",TABLE_DIR)
    print("Figures:",FIG_DIR)
    print("Summary:",SUMMARY_OUT)
    print("No training or model inference was performed.")

if __name__=="__main__":
    main()
