"""Confirm modified plan/evidence cannot open allowlisted main inputs."""
import json,tempfile
from pathlib import Path
from sp27_plan_guard import file_sha256,guarded_input

def main():
 with tempfile.TemporaryDirectory() as d:
  r=Path(d);plan=r/'plan.md';cfg=r/'config.json';e=r/'null.json';data=r/'labels.csv';plan.write_text('frozen');e.write_text('{}');data.write_text('toy only')
  c=dict(status='FROZEN',S2_review_approved=True,analysis_plan_sha256=file_sha256(plan),calibration_evidence=[dict(path=str(e),sha256=file_sha256(e))],inputs={'toy':{'labels':dict(path=str(data),sha256=file_sha256(data))}})
  cfg.write_text(json.dumps(c));assert guarded_input(plan,cfg,'toy','labels')==data
  for target in (plan,e,data):
   old=target.read_bytes();target.write_bytes(old+b'changed')
   try:guarded_input(plan,cfg,'toy','labels')
   except PermissionError:pass
   else:raise AssertionError('modified input accepted')
   target.write_bytes(old)
  c.pop('analysis_plan_sha256');cfg.write_text(json.dumps(c))
  try:guarded_input(plan,cfg,'toy','labels')
  except PermissionError:pass
  else:raise AssertionError('missing hash accepted')
 print('PASS: valid freeze; missing hash; modified plan, calibration and input rejected')
if __name__=='__main__':main()
