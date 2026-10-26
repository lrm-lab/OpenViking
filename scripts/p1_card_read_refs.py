"""Read-only P1 identity replay. Writes only regression projections and analysis artifacts.

Run from OpenViking-main: python scripts/p1_card_read_refs.py [workspace-root]
Historical decisive_ao_refs are a guarded actual-read lower bound, not the entire raw-read history.
"""
import pathlib,json,hashlib,sys,os
from types import SimpleNamespace as NS
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openviking.session.card_proposer import read_ao_ids
base=pathlib.Path(sys.argv[1]).resolve() if len(sys.argv)>1 else pathlib.Path(__file__).resolve().parents[2]
fixtures=[]
for domain,account in [('swe','evo-swe-d-20261025'),('ir','evo-ir-d-20261024')]:
 root=base/'experiments/evo-kw-rerun-20261020/local-state'/account/'user/eval-admin/sessions'
 n=0
 for d in sorted(root.iterdir()):
  for p in sorted((d/'adjudications').glob('*.json')):
   a=json.loads(p.read_text());trace=a.get('card_trace') or {}
   if a.get('status')!='frozen':continue
   if domain=='swe' and not any(x.get('evidence_ref','').endswith(':cards-v95') for x in a.get('evaluator_results',[])):continue
   if domain=='ir' and not (trace.get('verdict')=='proposals_all_rejected' and trace.get('read_aos')==0):continue
   report=d/'aig-paper-audit'/a['task_run_id']/'reader-report.json'
   if not report.exists():print('MISSING',p);continue
   refs=json.loads(report.read_text())['report']['decisive_ao_refs']
   views=json.loads((d/'views.json').read_text())
   revs=[r for k in views['keys'] for r in k['revisions'] if r['revision_id']==a['revision_id']]
   native=json.loads((d/'native-edges.json').read_text()) if (d/'native-edges.json').exists() else {}
   revs=[r['body'] for r in native.get('body',{}).get('revisions',[]) if r['body']['revision_id']==a['revision_id']] or revs
   assert len(revs)==1,(p,len(revs))
   records=[json.loads(l) for l in (d/'ao-ledger.jsonl').read_text().splitlines() if l]
   fixtures.append(dict(domain=domain,adjudication_id=a['adjudication_id'],source=str(p.relative_to(base)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),verdict=trace.get('verdict'),rejected=trace.get('rejected',[]),recorded_read_aos=trace.get('read_aos'),raw_read_lower_bound=refs,records=[dict(ao_id=r['ao_id'],sequence=r['sequence']) for r in records],blocks=[dict(block_id=b['block_id'],source_refs=[dict(ao_id=r['ao_id'],sequence=r['sequence']) for r in b['source_refs']]) for b in revs[0]['blocks']]))
   n+=1
 print(domain,n)
out=base/'OpenViking-main/tests/fixtures/card-read-ref-adjudications'
out.mkdir(exist_ok=True)
for fixture in fixtures:
 (out/(fixture['domain']+'-'+fixture['adjudication_id']+'.json')).write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n')
(base/'OpenViking-main/tests/fixtures/card-read-ref-adjudications.json').unlink(missing_ok=True)

results=[]
for c in fixtures:
 revision=NS(blocks=[NS(source_refs=[NS(**r) for r in b['source_refs']]) for b in c['blocks']])
 snapshot=NS(records=[NS(**r) for r in c['records']])
 refs=c['raw_read_lower_bound']
 os.environ.pop('OPENVIKING_CARD_READ_REF_ZERO_BASED',None)
 legacy=read_ao_ids(revision,refs,snapshot=snapshot)
 os.environ['OPENVIKING_CARD_READ_REF_ZERO_BASED']='1'
 corrected=read_ao_ids(revision,refs,snapshot=snapshot)
 actual={snapshot.records[i].ao_id for i in refs}
 assert set(corrected)<=actual
 checks=0
 for i,record in enumerate(snapshot.records):
  mapped=read_ao_ids(revision,[i],snapshot=snapshot)
  assert set(mapped)<={record.ao_id}
  checks+=1
 for bad in [-1,len(snapshot.records)]:
  try:read_ao_ids(revision,[bad],snapshot=snapshot)
  except ValueError:pass
  else:raise AssertionError('invalid index accepted')
 results.append(dict(domain=c['domain'],adjudication_id=c['adjudication_id'],raw_read_lower_bound=refs,legacy=list(legacy),corrected=list(corrected),recorded_read_aos=c['recorded_read_aos'],legacy_outside_confirmed_reads=sorted(set(legacy)-actual),singleton_checks=checks,source_sha256=c['sha256']))
output=base/'experiments/evo-swe-par2-20261025/analysis-20261026/P1-B1-parsed.json'
output.write_text(json.dumps(results,indent=2)+'\n')
for domain in ['swe','ir']:
 rows=[r for r in results if r['domain']==domain]
 print(domain,dict(cases=len(rows),indices=sum(len(r['raw_read_lower_bound']) for r in rows),valid=sum(len(r['raw_read_lower_bound']) for r in rows),legacy_zero=sum(not r['legacy'] for r in rows),corrected_zero=sum(not r['corrected'] for r in rows),changed=sum(r['legacy']!=r['corrected'] for r in rows),singleton_checks=sum(r['singleton_checks'] for r in rows),potential_false_baseline=sum(len(r['legacy_outside_confirmed_reads']) for r in rows)))

# Verify off-mode bytes against the exact pre-fix function for every IR source session.
import ast
import subprocess
legacy_source = subprocess.check_output(['git', 'show', 'a7bcb1781:openviking/session/card_proposer.py'], text=True)
legacy_node = next(n for n in ast.parse(legacy_source).body if isinstance(n, ast.FunctionDef) and n.name == 'read_ao_ids')
namespace = {}
exec('from __future__ import annotations\n' + ast.unparse(legacy_node), namespace)
legacy_function = namespace['read_ao_ids']
os.environ.pop('OPENVIKING_CARD_READ_REF_ZERO_BASED', None)
ir_root = base/'experiments/evo-kw-rerun-20261020/local-state/evo-ir-d-20261024/user/eval-admin/sessions'
checks = []
for directory in sorted(ir_root.iterdir()):
    ledger = directory/'ao-ledger.jsonl'
    if not ledger.exists():
        raise AssertionError(f'missing ledger: {directory.name}')
    records = [json.loads(line) for line in ledger.read_text().splitlines() if line]
    revisions = []
    views = directory/'views.json'
    if views.exists():
        revisions.extend(r for key in json.loads(views.read_text())['keys'] for r in key['revisions'])
    native = directory/'native-edges.json'
    if native.exists():
        revisions.extend(r['body'] for r in json.loads(native.read_text())['body']['revisions'])
    for raw in revisions:
        revision = NS(blocks=[NS(source_refs=[NS(**r) for r in block['source_refs']]) for block in raw['blocks']])
        refs = list(range(len(records)))
        before = json.dumps(legacy_function(revision, refs), separators=(',', ':'))
        after = json.dumps(read_ao_ids(revision, refs), separators=(',', ':'))
        assert before == after
        checks.append({'session': directory.name, 'revision': raw['revision_id'], 'sha256': hashlib.sha256(after.encode()).hexdigest()})
manifest = {'sessions': len(list(ir_root.iterdir())), 'revision_comparisons': len(checks), 'mismatches': 0, 'checks': checks}
(base/'experiments/evo-swe-par2-20261025/analysis-20261026/P1-IR-off-baseline.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('IR off baseline:', manifest['sessions'], 'sessions;', manifest['revision_comparisons'], 'revisions; mismatches=0')
