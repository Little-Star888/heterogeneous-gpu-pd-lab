import hashlib
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RUN = ROOT / 'hetero-v2/xyvllm-dspark-20260920'
records = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source, destination, phase):
    source = ROOT / source
    target = BASE / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)
    records.append({'path': destination, 'source': source.relative_to(ROOT).as_posix(),
                    'sha256': sha(target), 'bytes': target.stat().st_size, 'phase': phase})


for source, destination, phase in [
    ('hetero-v2/xyvllm-overlay-20260920.tgz', 'packages/xyvllm-overlay-20260920.tgz', 'transport_before_dspark'),
    ('hetero-v2/xyvllm-dspark-compute-delta-20260920.tgz', 'packages/xyvllm-dspark-compute-delta-20260920.tgz', 'final_delta'),
    ('hetero-v2/xyvllm-results-20260920/REPORT.md', 'evidence/transport-report.md', 'transport_before_dspark'),
]:
    freeze(source, destination, phase)
for source, destination in [
    ('REPORT.md', 'evidence/final-report.md'),
    ('fast/results/effective.json', 'evidence/effective.json'),
    ('fast/results/summary.json', 'evidence/summary.json'),
    ('fast/golden-repro-summary.json', 'evidence/golden-repro-summary.json'),
    ('final-health-d.json', 'evidence/final-health-d.json'),
    ('final-health-p.json', 'evidence/final-health-p.json'),
    ('final-health-rank0.json', 'evidence/final-health-rank0.json'),
    ('payload/runtime.env', 'runtime.env'),
]:
    freeze((RUN / source).relative_to(ROOT), destination, 'final')
patch = ROOT / 'hetero-v2/xyvllm-patch'
for source in sorted(patch.rglob('*')):
    if source.is_file() and source.suffix in {'.py', '.sh', '.md', '.add'} and '__pycache__' not in source.parts:
        freeze(source.relative_to(ROOT), 'patch/' + source.relative_to(patch).as_posix(), 'final_source')
with tarfile.open(BASE / 'packages/xyvllm-dspark-compute-delta-20260920.tgz') as archive:
    matched = set()
    for member in archive.getmembers():
        name = Path(member.name).name
        if member.isfile() and name in {'decode.py', 'utils.py', 'deepseek_v4_hook.py', 'runtime.env'}:
            target = BASE / ('runtime.env' if name == 'runtime.env' else 'patch/' + name)
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == sha(target), name
            matched.add(name)
    assert matched == {'decode.py', 'utils.py', 'deepseek_v4_hook.py', 'runtime.env'}
audit_source = ROOT / 'hetero-v2/xyvllm-results-20260920/config-audit-live.json'
audit = json.loads(audit_source.read_text(encoding='utf-8-sig'))
env_rows = []
for row in audit:
    selected = dict(line.split('=', 1) for line in row.get('stdout', '').splitlines()
                    if line.startswith(('XY_PD_TAIL=', 'CHUNKED_PREFILL_SIZE=', 'MAX_RUNNING_REQUESTS=')))
    if selected:
        assert selected['XY_PD_TAIL'] == '256'
        env_rows.append({'node': row['node'], 'environment': selected})
assert len(env_rows) == 4
(BASE / 'evidence/tail-environment.json').write_text(json.dumps({
    'source': audit_source.relative_to(ROOT).as_posix(), 'source_sha256': sha(audit_source),
    'phase': 'before_dspark; tail unchanged by final runtime.env', 'nodes': env_rows}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
effective = json.loads((BASE / 'evidence/effective.json').read_text())
manifest = {
    'version': 'V8', 'name': 'ds4.1flash6卡（4spark+2张6000Dpro）V8版本',
    'named_at': datetime.now().astimezone().isoformat(),
    'entry_document': 'ds4.1flash6gpu-v8.md',
    'status': 'accepted_local_baseline_snapshot', 'runtime_evidence_date': '2026-09-20',
    'runtime_rechecked_during_naming': False,
    'architecture': {'P': {'engine': 'vLLM', 'layers': 21, 'tp': 2, 'hardware': '2x RTX 6000Dpro'},
                     'D': {'engine': 'SGLang', 'layers': 40, 'tp': 4, 'ep': 4, 'hardware': '4x DGX Spark'},
                     'transfer': 'cross-engine NIXL with KV layout/state conversion'},
    'effective_d': effective,
    'tail_tokens_configured': 256, 'tail_evidence': 'evidence/tail-environment.json',
    'validation': {'final_correctness_checks': 4, 'final_matrix_cases': 10, 'final_matrix_requests': 55,
                   'final_matrix_max_input_tokens': 32768, 'final_golden_requests': 27,
                   'earlier_transport_cases': 24, 'earlier_transport_requests': 172,
                   'earlier_transport_max_input_tokens': 100000},
    'performance': {'golden_c1_decode_tps': 72.83333333333333,
                    'golden_c8_end_to_end_aggregate_tps': 231.83333333333331,
                    'v7_speedup': None, 'reason': 'No matched V7/V8 performance experiment'},
    'restore_scope': 'Existing role images and external model/Engram data required; no new full-image export or cold-restore test',
    'package_order': ['packages/xyvllm-overlay-20260920.tgz', 'packages/xyvllm-dspark-compute-delta-20260920.tgz'],
    'artifacts': records,
}
(BASE / 'baseline.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
files = sorted(p for p in BASE.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts)
lines = [sha(p) + '  ' + p.relative_to(BASE).as_posix() for p in files]
entry = ROOT / manifest['entry_document']
lines.append(sha(entry) + '  ../' + entry.name)
(BASE / 'SHA256SUMS').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Frozen V8: {len(records)} source artifacts, {len(lines)} checksums; delta/source consistency PASS')
