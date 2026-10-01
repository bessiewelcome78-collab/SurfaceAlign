#!/usr/bin/env python3
"""Resolve the single source-of-truth SemLT ablation specification."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

VARIANT_KEYS = {
    "operator": "M1.GEOTR_M1_OPERATOR",
    "transport_space": "M1.GEOTR_M1_TRANSPORT_SPACE",
    "support_ratio": "M1.GEOTR_M1_UNIFIED_SUPPORT_RATIO",
    "tangent_ratio": "M1.GEOTR_M1_UNIFIED_TANGENT_RATIO",
}

def load(path: Path):
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != "SemLT_locked_causal_ablation_v1":
        raise SystemExit("[FAIL] unexpected ablation spec protocol")
    return data

def fmt(v):
    if isinstance(v,bool): return "true" if v else "false"
    return str(v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/SEMLT_ABLATION_LOCKED_VARIANTS.json")
    ap.add_argument("--variant", default="")
    ap.add_argument("--emit-overrides", action="store_true")
    ap.add_argument("--emit-json", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sha256", action="store_true")
    a=ap.parse_args(); path=Path(a.spec); data=load(path)
    if a.sha256:
        print(hashlib.sha256(path.read_bytes()).hexdigest()); return
    if a.list:
        for key,item in data["variants"].items():
            print(f"{key}\t{item['paper_name']}")
        return
    if not a.variant or a.variant not in data["variants"]:
        raise SystemExit(f"[FAIL] choose --variant from {', '.join(data['variants'])}")
    item=data["variants"][a.variant]
    merged=dict(data["fixed_overrides"])
    for src,dst in VARIANT_KEYS.items(): merged[dst]=item[src]
    if a.emit_overrides:
        for k,v in merged.items(): print(f"{k}\t{fmt(v)}")
    elif a.emit_json:
        print(json.dumps({"variant":a.variant,"metadata":item,"overrides":merged},ensure_ascii=False,indent=2))
    else:
        print(item["paper_name"])
if __name__=="__main__": main()
