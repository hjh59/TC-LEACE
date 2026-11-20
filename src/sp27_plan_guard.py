"""Fail closed for SP27 main-analysis inputs and frozen calibration evidence."""
import json,hashlib
from pathlib import Path

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def require_frozen_plan(plan_path,config_path):
    plan_path,config_path=Path(plan_path),Path(config_path)
    if not plan_path.is_file() or not config_path.is_file():raise PermissionError('SP27_PLAN_NOT_FROZEN')
    config=json.loads(config_path.read_text());expected=config.get('analysis_plan_sha256')
    if config.get('status')!='FROZEN':raise PermissionError('SP27_PLAN_NOT_FROZEN')
    if not isinstance(expected,str) or len(expected)!=64:raise PermissionError('SP27_PLAN_HASH_MISSING')
    if file_sha256(plan_path)!=expected:raise PermissionError('SP27_PLAN_HASH_MISMATCH')
    if not config.get('S2_review_approved',False):raise PermissionError('SP27_S2_REVIEW_NOT_APPROVED')
    evidence=config.get('calibration_evidence')
    if not evidence:raise PermissionError('SP27_CALIBRATION_EVIDENCE_MISSING')
    for item in evidence:
        if not Path(item['path']).is_file() or file_sha256(item['path'])!=item['sha256']:raise PermissionError('SP27_CALIBRATION_HASH_MISMATCH')
    return config

def guarded_input(plan_path,config_path,dataset,key):
    """Check the freeze BEFORE a caller may parse any main input contents."""
    config=require_frozen_plan(plan_path,config_path)
    item=config.get('inputs',{}).get(dataset,{}).get(key)
    if not item:raise PermissionError('SP27_INPUT_NOT_ALLOWLISTED')
    path=Path(item['path'])
    if not path.is_file() or file_sha256(path)!=item['sha256']:raise PermissionError('SP27_INPUT_HASH_MISMATCH')
    return path
