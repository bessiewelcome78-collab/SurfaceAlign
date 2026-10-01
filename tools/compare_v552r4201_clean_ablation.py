#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path

PAIR = re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
VAL = re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')

def parse(p: Path):
    lines = p.read_text(errors='replace').splitlines()
    rows = [{k: float(v) for k, v in PAIR.findall(x)} for x in lines if 'M1_DIAG:' in x]
    vals = [tuple(map(float, m.groups())) for x in lines if (m := VAL.search(x))]
    r = rows[-1] if rows else {}
    teacher = r.get('v538_action_realizable_teacher_oracle_gain', math.nan)
    student = r.get('v538_component_oracle_gain', math.nan)
    native = r.get('v552r420_native_mask_matching_dice', r.get('v552r419_native_mask_matching_dice', math.nan))
    purity = r.get('v552r420_native_mask_soft_purity', r.get('v552r419_native_mask_soft_purity', math.nan))
    coverage = r.get('v552r420_native_mask_soft_coverage', r.get('v552r419_native_mask_soft_coverage', math.nan))
    return dict(
        route_live=r.get('v552r4201_objective_route_live', math.nan),
        routed_ratio=r.get('v552r4201_m1_routed_ratio', math.nan),
        proposal_loss=r.get('v552r4201_proposal_loss_abs', math.nan),
        m1_grad=r.get('v538_m1_grad_norm', math.nan),
        teacher_built=r.get('v552r4201_forward_teacher_built', math.nan),
        teacher_valid=r.get('v552r4201_teacher_valid_count', math.nan),
        teacher_err=r.get('v552r4201_teacher_error_fraction', math.nan),
        target_mass=r.get('v552r4201_location_target_mass', math.nan),
        loc=r.get('v552r417_location_center_recall', math.nan),
        peak=r.get('v552r417_pre_topk_peak_recall', math.nan),
        native=native, purity=purity, coverage=coverage,
        capture=r.get('v538_component_capture_ratio', math.nan), student=student,
        teacher=teacher,
        retain=student / teacher if math.isfinite(teacher) and abs(teacher) > 1e-12 else math.nan,
        gain=r.get('v552r44_quality_candidate_audit_mean_gain', math.nan),
        harm=r.get('v552r44_quality_candidate_audit_harm_rate', math.nan),
        best=max((v[3] for v in vals), default=math.nan),
        complete=any('TRAIN COMPLETE' in x for x in lines),
        fatal=any(s in x for x in lines for s in ('Traceback (most recent call last)', 'CUDA out of memory', 'RuntimeError:', 'ValueError:')),
        path=p,
    )

def F(x): return 'n/a' if not math.isfinite(x) else f'{x:.4f}'

def routing_ok(d):
    if not math.isfinite(d.get('route_live', math.nan)):
        return True  # historical A0 predates the R4201 fail-closed routing diagnostic
    return (d['route_live'] >= .5 and d.get('routed_ratio',0)>0 and d.get('proposal_loss',0)>0 and d.get('m1_grad',0)>0)

def ownership_ok(d):
    if not math.isfinite(d.get('teacher_built', math.nan)):
        return True  # historical A0 does not expose R4201 ownership diagnostics
    err = d.get('teacher_err', math.nan); valid = d.get('teacher_valid', math.nan); mass = d.get('target_mass', math.nan)
    return d['teacher_built'] >= .5 and ((not math.isfinite(err)) or err <= 1e-9 or valid > 0) and ((not math.isfinite(err)) or err <= 1e-9 or mass > 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='SMOKE20', choices=['SMOKE20','FORMAL150'])
    ap.add_argument('--stamp', required=True)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(); root = Path('logs')
    specs = [
        ('A0_R419_Control', f'V552R4201_A0_R419_Control_{a.mode}_seed{a.seed}_gpu*_{a.stamp}.log'),
        ('A1_CleanDynamicMask', f'V552R4201_A1_CleanDynamicMask_{a.mode}_seed{a.seed}_gpu*_{a.stamp}.log'),
        ('A2_CleanDynamicTypeDecoupled', f'V552R4201_A2_CleanDynamicTypeDecoupled_{a.mode}_seed{a.seed}_gpu*_{a.stamp}.log'),
    ]
    data = {}
    for name, pat in specs:
        fs = sorted(root.glob(pat)); data[name] = parse(fs[-1]) if fs else {}
    print('Variant                         Route Own LocR PeakR NativeMatch Purity Coverage Capture SOracle Retain CandGain Harm BestM2 Complete')
    for name, _ in specs:
        d = data[name]
        print(f'{name:<31} {str(routing_ok(d)):<5} {str(ownership_ok(d)):<4} {F(d.get("loc",math.nan)):>5} {F(d.get("peak",math.nan)):>5} {F(d.get("native",math.nan)):>11} {F(d.get("purity",math.nan)):>6} {F(d.get("coverage",math.nan)):>8} {F(d.get("capture",math.nan)):>7} {F(d.get("student",math.nan)):>7} {F(d.get("retain",math.nan)):>6} {F(d.get("gain",math.nan)):>8} {F(d.get("harm",math.nan)):>5} {F(d.get("best",math.nan)):>6} {d.get("complete",False) and not d.get("fatal",False)}')
    a0, a1, a2 = (data[n] for n, _ in specs)
    if not routing_ok(a1) or not routing_ok(a2):
        print('\n[CAUSAL INTERPRETATION BLOCKED]')
        print('R4.20.1 objective-routing contract failed in a clean arm. M1 was not truly optimized; do NOT interpret decoder/type deltas.')
        raise SystemExit(3)
    if not ownership_ok(a1) or not ownership_ok(a2):
        print('\n[CAUSAL INTERPRETATION BLOCKED]')
        print('R4.20.1 ownership contract failed in a clean arm. Do NOT compare decoder/type deltas; fix Teacher/target plumbing first.')
        raise SystemExit(3)
    def D(k,x,y):
        u=x.get(k,math.nan); v=y.get(k,math.nan); return v-u if math.isfinite(u) and math.isfinite(v) else math.nan
    print('\n[Causal deltas after ownership is healthy]')
    print(f'A1-A0 clean dynamic-mask decoder: ΔLocR={F(D("loc",a0,a1))} ΔNativeMatch={F(D("native",a0,a1))} ΔPurity={F(D("purity",a0,a1))} ΔCapture={F(D("capture",a0,a1))} ΔStudentOracle={F(D("student",a0,a1))}')
    print(f'A2-A1 type/shape decoupling:      ΔLocR={F(D("loc",a1,a2))} ΔNativeMatch={F(D("native",a1,a2))} ΔPurity={F(D("purity",a1,a2))} ΔCapture={F(D("capture",a1,a2))} ΔStudentOracle={F(D("student",a1,a2))}')
    ready = all([
        a2.get('complete',False), not a2.get('fatal',True), routing_ok(a2), ownership_ok(a2),
        a2.get('loc',0)>=.90, a2.get('peak',0)>=.75, a2.get('native',0)>=.20,
        a2.get('purity',0)>=.12, a2.get('capture',0)>=.15, a2.get('student',0)>=.005,
        a2.get('retain',0)>=.20, a2.get('gain',-1)>0, a2.get('harm',1)<=.10,
    ])
    print('\n[FORMAL READINESS]', 'READY' if ready else 'NOT_READY')
    if not ready:
        print('Do NOT run FORMAL150 or loosen gates. Follow the first failed stage in diagnose_v552r4201_log.py.')
if __name__ == '__main__': main()
