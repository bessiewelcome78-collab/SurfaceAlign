#!/usr/bin/env python3
from pathlib import Path
import os,re,csv
project=Path(os.environ.get("PROJECT",Path(__file__).resolve().parents[3]))
root=Path(os.environ.get("RESULT_ROOT",project/"surface_main3_stepE30_s42_results"))
study=os.environ.get("STUDY_ID") or (project/"SURFACE_MAIN3_STEPE30_S42_ACTIVE.txt").read_text().strip()
pat=re.compile(r"Average (DSC|NSD).*?:\s*([0-9.]+)%")
rows=[]
for ds in ("Kvasir","ISIC","BTMRI"):
    rr=root/study/ds/"FULL_STEP_E30"/"seed42"/"logs"
    def m(p):
        if not p.exists(): return {}
        return {k:float(v) for k,v in pat.findall(p.read_text(errors="replace"))}
    t=m(rr/"eval_test_mc30_true2d.log"); l=m(rr/"eval_test_mc30_paper_legacy.log")
    v=m(rr/"eval_val_mc10_true2d.log")
    rows.append({"dataset":ds,"val_dsc":v.get("DSC",""),"val_true2d_nsd":v.get("NSD",""),"test_dsc":t.get("DSC",""),"test_true2d_nsd":t.get("NSD",""),"test_legacy_nsd":l.get("NSD","")})
out=root/study/"table_main3_stepE30_s42.csv"; out.parent.mkdir(parents=True,exist_ok=True)
with out.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(out)
for r in rows: print(r)
