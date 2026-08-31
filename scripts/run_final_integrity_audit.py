#!/usr/bin/env python3
from __future__ import annotations
import ast, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT

N=2158; P=1877; C=13; B=5000; BC=3; BR=39
SEEDS=(20260826,20260827,20260828)
FINAL_CONFIG=PROJECT_ROOT/"configs/final_evaluation_v1.yaml"
CORRUPTION_CONFIG=PROJECT_ROOT/"configs/corruption_v1.yaml"
FINAL_DIR=PROJECT_ROOT/"results/final_evaluation/v1/test"
FINAL_SUMMARY=FINAL_DIR/"summary.json"
COND=FINAL_DIR/"condition_metrics.csv"
L_TEST=FINAL_DIR/"test_annotation_confidence.csv"
BOOT_DIR=PROJECT_ROOT/"results/bootstrap/v1/final_test"
BOOT_SUMMARY=BOOT_DIR/"summary.json"
BOOT_TABLE=BOOT_DIR/"primary_aurc_bootstrap_summary.csv"
BOOT_REPS=BOOT_DIR/"primary_aurc_bootstrap_replicates.npz"
REPORT_SUMMARY=PROJECT_ROOT/"results/reporting/v1/FINAL_RESULTS_SUMMARY.json"
LOG=PROJECT_ROOT/"logs/final_integrity_audit.json"
DOC=PROJECT_ROOT/"docs/FINAL_INTEGRITY_AUDIT.md"

def j(path):
    with path.open("r",encoding="utf-8") as f:d=json.load(f)
    if not isinstance(d,dict):raise ValueError(f"{path} must contain object")
    return d

def ci(v):
    x=ast.literal_eval(v) if isinstance(v,str) else v
    return float(x[0]),float(x[1])

class A:
    def __init__(self):self.rows=[]
    def check(self,name,ok,detail=""):
        ok=bool(ok); self.rows.append({"name":name,"passed":ok,"detail":detail})
        print(f"{'PASS' if ok else 'FAIL':4s}  {name}"+(f" — {detail}" if detail else ""))
        if not ok:raise AssertionError(f"{name}: {detail}")

def main():
    a=A(); print("FINAL INTEGRITY AUDIT\n")
    fc=yaml.safe_load(FINAL_CONFIG.read_text()); cc=yaml.safe_load(CORRUPTION_CONFIG.read_text())
    fs=j(FINAL_SUMMARY); bs=j(BOOT_SUMMARY)

    a.check("final config frozen",fc["status"]=="FROZEN")
    a.check("development closed",fc["development_closed"] is True)
    a.check("post-test tuning forbidden",fc["post_test_tuning_allowed"] is False)
    a.check("corruption protocol frozen",cc["status"]=="FROZEN")
    a.check("final evaluation frozen",fs["status"]=="FROZEN")
    a.check("test-set ECDF fitting absent",fs["test_set_ecdf_fitting_used"] is False)
    a.check("final record count",int(fs["records"])==N,str(fs["records"]))
    a.check("final patient count",int(fs["patients"])==P,str(fs["patients"]))
    a.check("final condition count",int(fs["conditions"])==C,str(fs["conditions"]))

    fmap={
        "final_config":FINAL_CONFIG,
        "corruption_config":CORRUPTION_CONFIG,
        "manifest":PROJECT_ROOT/"data/processed/cohort/v1/ptbxl_superdiagnostic_cohort.csv",
        "normalization":PROJECT_ROOT/"data/processed/preprocessing/v1/global_zscore_stats.json",
        "checkpoint":PROJECT_ROOT/"results/baseline/v1/seed_20260826/best_checkpoint.pt",
        "thresholds":PROJECT_ROOT/"results/baseline/v1/seed_20260826/validation_thresholds.json",
        "validation_L":PROJECT_ROOT/"results/label_confidence/v1/validation/label_confidence.csv",
        "training_Q_reference":PROJECT_ROOT/"data/processed/waveform_signal_quality/v1/training_reference.npz",
        "validation_combination_reference":PROJECT_ROOT/"results/selective_prediction_waveform_q/v1/validation/combination_reference.npz",
        "script":PROJECT_ROOT/"scripts/run_final_test_evaluation.py",
    }
    for k,p in fmap.items():a.check(f"final source hash: {k}",sha256_file(p)==fs["source_sha256"][k])
    a.check("final condition-table hash",sha256_file(COND)==fs["condition_table_sha256"])

    ct=pd.read_csv(COND,float_precision="round_trip")
    a.check("condition table shape",ct.shape==(13,19),str(ct.shape))
    names=ct["condition"].astype(str).tolist()
    a.check("condition names unique",len(set(names))==C)

    eref=pref=tref=None
    for name in names:
        art=FINAL_DIR/name/"results.npz"
        a.check(f"{name}: artifact exists",art.exists())
        a.check(f"{name}: artifact hash",sha256_file(art)==fs["condition_results"][name]["results_npz_sha256"])
        d=np.load(art,allow_pickle=False)
        e=np.asarray(d["ecg_ids"],np.int64); p=np.asarray(d["patient_ids"],np.int64); t=np.asarray(d["targets"],np.int64)
        prob=np.asarray(d["probabilities"],float); mc=np.asarray(d["mc_probabilities"],float)
        a.check(f"{name}: ECG count",e.shape==(N,))
        a.check(f"{name}: probability shape",prob.shape==(N,5))
        a.check(f"{name}: MC shape",mc.shape==(30,N,5))
        a.check(f"{name}: finite probabilities",np.isfinite(prob).all())
        a.check(f"{name}: finite MC probabilities",np.isfinite(mc).all())
        if eref is None: eref=e; pref=p; tref=t
        else:
            a.check(f"{name}: ECG ordering",np.array_equal(e,eref))
            a.check(f"{name}: patient ordering",np.array_equal(p,pref))
            a.check(f"{name}: targets unchanged",np.array_equal(t,tref))
    a.check("unique fold-10 patients in artifacts",np.unique(pref).size==P,str(np.unique(pref).size))

    ld=pd.read_csv(L_TEST,float_precision="round_trip")
    a.check("test L record count",len(ld)==N)
    a.check("test L ECG ordering",np.array_equal(ld["ecg_id"].to_numpy(np.int64),eref))
    a.check("test L hash",sha256_file(L_TEST)==fs["annotation_confidence"]["artifact_sha256"])

    a.check("bootstrap frozen",bs["status"]=="FROZEN")
    a.check("bootstrap unit patient",bs["unit"]=="patient")
    a.check("bootstrap paired",bs["paired"] is True)
    a.check("bootstrap replicate count",int(bs["replicates"])==B)

    bmap={
        "final_config":FINAL_CONFIG,
        "final_summary":FINAL_SUMMARY,
        "condition_table":COND,
        "test_L":L_TEST,
        "validation_combination_reference":PROJECT_ROOT/"results/selective_prediction_waveform_q/v1/validation/combination_reference.npz",
        "script":PROJECT_ROOT/"scripts/run_final_patient_bootstrap.py",
    }
    for k,p in bmap.items():a.check(f"bootstrap source hash: {k}",sha256_file(p)==bs["source_sha256"][k])
    a.check("bootstrap replicate artifact hash",sha256_file(BOOT_REPS)==bs["replicates_sha256"])
    a.check("bootstrap summary-table hash",sha256_file(BOOT_TABLE)==bs["summary_table_sha256"])

    bt=pd.read_csv(BOOT_TABLE,float_precision="round_trip")
    a.check("bootstrap table shape",bt.shape==(BR,9),str(bt.shape))
    rd=np.load(BOOT_REPS,allow_pickle=False); deltas=np.asarray(rd["deltas"],float)
    a.check("bootstrap replicate array shape",deltas.shape==(B,C,BC),str(deltas.shape))
    a.check("bootstrap replicates finite",np.isfinite(deltas).all())

    cb=ab=inc=0
    for r in bt.itertuples(index=False):
        exp=float(fs["condition_results"][r.condition]["selective_prediction"][r.method]["aurc"])-float(fs["condition_results"][r.condition]["selective_prediction"][r.baseline]["aurc"])
        a.check(f"bootstrap point delta: {r.condition}/{r.comparison}",np.isclose(float(r.point_delta),exp,rtol=0,atol=1e-12))
        lo,hi=ci(r.ci_95_percentile)
        if lo>0: cb+=1; expect="method_higher_aurc"
        elif hi<0: ab+=1; expect="method_lower_aurc"
        else: inc+=1; expect="inconclusive_ci_crosses_zero"
        a.check(f"bootstrap conclusion: {r.condition}/{r.comparison}",r.conclusion==expect)
    a.check("bootstrap comparison accounting",cb+ab+inc==BR)

    for seed in SEEDS:
        rd=PROJECT_ROOT/f"results/baseline/v1/seed_{seed}"
        m=j(rd/"run_metadata.json")
        a.check(f"seed {seed}: metadata seed",int(m["seed"])==seed)
        a.check(f"seed {seed}: 50 epochs completed",int(m["epochs_completed"])==50)
        a.check(f"seed {seed}: no test records used",int(m["test_records_used"])==0)
        a.check(f"seed {seed}: checkpoint reproduction",np.isclose(float(m["best_validation_macro_auroc"]),float(m["reproduced_validation_macro_auroc"]),rtol=0,atol=1e-15))
        a.check(f"seed {seed}: checkpoint hash",sha256_file(rd/"best_checkpoint.pt")==m["best_checkpoint_sha256"])

    if REPORT_SUMMARY.exists():
        rp=j(REPORT_SUMMARY)
        a.check("reporting status",rp["status"]=="GENERATED_FROM_FROZEN_RESULTS")
        for rel,h in rp["generated_artifact_sha256"].items():
            p=PROJECT_ROOT/rel
            a.check(f"report artifact hash: {rel}",p.exists() and sha256_file(p)==h)

    print("\nRunning full automated test suite...")
    proc=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=PROJECT_ROOT,capture_output=True,text=True)
    detail=(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip())
    a.check("pytest suite",proc.returncode==0,detail)

    passed=sum(x["passed"] for x in a.rows); total=len(a.rows)
    result={
        "audit_id":"ptbxl_final_integrity_audit_v1",
        "checks_passed":passed,
        "checks_total":total,
        "all_passed":passed==total,
        "bootstrap_outcome_counts":{"confidence_lower_aurc_supported":cb,"alternative_lower_aurc_supported":ab,"inconclusive":inc},
        "checks":a.rows,
        "status":"PASS" if passed==total else "FAIL",
    }
    LOG.parent.mkdir(parents=True,exist_ok=True); DOC.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("w",encoding="utf-8") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    DOC.write_text(
        "# Final Integrity Audit\n\n"
        f"- Status: **{result['status']}**\n"
        f"- Checks passed: **{passed}/{total}**\n"
        f"- Final test records: **{N}**\n"
        f"- Final test patients: **{P}**\n"
        f"- Final conditions: **{C}**\n"
        f"- Patient-level bootstrap: **{B} replicates**\n"
        f"- Primary AURC comparisons supporting lower ordinary-confidence AURC: **{cb}/39**\n"
        f"- Primary AURC comparisons supporting lower alternative-method AURC: **{ab}/39**\n"
        f"- Inconclusive primary comparisons: **{inc}/39**\n\n"
        "This audit verifies frozen source hashes, final fold-10 artifacts, "
        "cross-condition identities, bootstrap artifacts and point estimates, "
        "three-seed checkpoint reproduction, generated reporting hashes, and "
        "the automated test suite.\n",
        encoding="utf-8"
    )
    print("\nFINAL AUDIT COMPLETE")
    print("Status:",result["status"])
    print(f"Checks: {passed}/{total}")
    print("JSON:",LOG)
    print("Report:",DOC)

if __name__=="__main__":
    main()
