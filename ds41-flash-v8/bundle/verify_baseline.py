import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    data = json.loads((BASE / 'baseline.json').read_text(encoding='utf-8-sig'))
    assert data['version'] == 'V8'
    checks = 0
    for line in (BASE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (BASE / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        assert digest(path) == expected, relative
        checks += 1
    for item in data['artifacts']:
        assert digest(BASE / item['path']) == item['sha256'], item['path']
    for doc in [BASE / 'README.md', ROOT / data['entry_document']]:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8-sig')):
            if not re.match(r'https?://', target):
                assert (doc.parent / target.split('#')[0]).exists(), target
    effective = json.loads((BASE / 'evidence/effective.json').read_text())
    assert effective['speculative_algorithm'] == 'DSPARK'
    assert effective['speculative_dspark_block_size'] == 5
    assert effective['disable_cuda_graph'] is False
    assert effective['disable_overlap_schedule'] is False
    assert effective['fp8_gemm_runner_backend'] == 'flashinfer_cutlass'
    assert effective['disable_shared_experts_fusion'] is True
    rows = json.loads((BASE / 'evidence/summary.json').read_text())
    assert len(rows) == 10 and sum(x['success'] for x in rows) == 55
    assert all(x['failed'] == 0 and x['success'] == x['requests'] for x in rows)
    health = json.loads((BASE / 'evidence/final-health-d.json').read_text())
    for record in health:
        assert record['ok'] is True
        for line in record['output'].splitlines():
            if re.match(r'^[a-f0-9]{64}  ', line):
                expected, path = line.split('  ', 1)
                assert digest(BASE / 'patch' / Path(path).name) == expected
    print(f'PASS: V8; SHA256={checks}; links=OK; final configuration=OK; matrix=55/55; four-node patch hashes=OK')


if __name__ == '__main__':
    verify()
