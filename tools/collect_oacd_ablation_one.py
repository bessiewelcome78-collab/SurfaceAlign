#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--arm',required=True); p.add_argument('--result-dir',required=True); p.add_argument('--output',required=True); a=p.parse_args()
r=Path(a.result_dir)
def stats(name):
 f=r/name
 if not f.is_file(): raise SystemExit(f'missing {f}')
 d=pd.read_csv(f)
 cols={c.lower():c for c in d.columns}
 dc=cols.get('dsc') or cols.get('dice'); nc=cols.get('nsd')
 return {'n':len(d),'DSC':float(d[dc].mean()),'NSD':float(d[nc].mean())}
out={'arm':a.arm,'Base_true2d':stats('test_BaseNative_true2d.csv'),'M1_true2d':stats('test_M1Native_true2d.csv'),'Base_legacy':stats('test_BaseNative_paper_legacy.csv'),'M1_legacy':stats('test_M1Native_paper_legacy.csv')}
out['delta_true2d']={'DSC':out['M1_true2d']['DSC']-out['Base_true2d']['DSC'],'NSD':out['M1_true2d']['NSD']-out['Base_true2d']['NSD']}
Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
