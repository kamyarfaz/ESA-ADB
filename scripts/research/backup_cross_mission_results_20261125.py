"""Archive immutable v2 evidence in <2 GiB assets and verify fresh downloads."""
import argparse
import hashlib
import json
import tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results_longrun/cross_mission_transfer_20261120_v2'
SOURCE=ROOT/'results_longrun/context_expansion_20261104_v1'
T=ROOT/'docs/research/tables/cross-mission-2026-11-20-v2'
O=ROOT/'outputs/cross-mission-study-20261125'
DEST=Path('/home/k_faz/project-backups/cross-mission-v2-20261120/release-20261125')

def digest(stream):
    h=hashlib.sha256()
    for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def sha(p):
    with p.open('rb') as f:return digest(f)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','verify']);parser.add_argument('--download',type=Path);args=parser.parse_args()
    if args.mode=='build':
        for n in ('verification.json','wandb-cloud-verification.json','workbook-verification.json'):assert json.loads((T/n).read_text())['verified']
        assert json.loads((T/'all-preparation-verification.json').read_text())['passed']
        complete=json.loads((RUN/'complete.json').read_text());plan=json.loads((RUN/'frozen/protocol.json').read_text())
        files={RUN/n for n in complete['files']}|{RUN/'complete.json'}
        files|={ROOT/n for n in plan['source_sha256']}
        files|=set(RUN.glob('prepared/*/*/complete.json'))
        files|={p for p in T.rglob('*') if p.is_file()}
        files|=set((ROOT/'scripts/research').glob('*cross_mission*20261125.py'))
        files|={ROOT/n for n in ('scripts/research/verify_mission2_study_20261115.py','requirements-test.txt','README.md','AGENTS.md','MIGRATION.md','ESA_thesis_results_full_and_historical.xlsx','docs/research/cross-mission-results-2026-11-25.md','docs/research/cross-mission-channel-research-2026-11-18.md','docs/research/cross-mission-whole-system-protocol-2026-11-18.md','esa_thesis/cross_mission_study.py','scripts/research/verify_cross_mission_v2_amendment_20261120.py')}
        files|={p for p in (ROOT/'results_longrun/cross_mission_transfer_20261118_v1/research').rglob('*') if p.is_file()}
        files|={ROOT/'results_longrun/all_channel_correlations_20261102_v1'/n for n in ('channel_inventory.csv','mission1_block_matrices.npz','mission2_block_matrices.npz','complete.json')}
        files|={ROOT/n for n in json.loads((RUN/'preparation.json').read_text())['raw'] if n.endswith('.csv')}
        files|={p for p in (ROOT/'timeeval/metrics').rglob('*.py')}
        files|={p for p in O.glob('*') if p.is_file() and p.suffix in ('.json','.mjs','.png','.txt','.log','.xlsx')}
        files|={p for p in (O/'logs').glob('*') if p.is_file()}
        selection=json.loads((RUN/'selection_complete.json').read_text())
        files.add(SOURCE/'selection_complete.json')
        for s in selection['sources']:
            files|={SOURCE/s['job']['name']/n for n in ('model.pt','frozen_rule.json','trained.json')}
        # Preserve a source/failure snapshot of v1 without duplicating all old arrays.
        old=ROOT/'results_longrun/cross_mission_transfer_20261118_v1'
        files|={p for p in old.glob('*.json') if p.is_file()}
        files|={p for p in old.glob('preparation_frozen/*.json') if p.is_file()}
        DEST.mkdir(parents=True,exist_ok=True)
        groups=[];group=[];size=0
        for p in sorted(files):
            assert p.is_file() and not p.is_symlink(),str(p)
            if group and size+p.stat().st_size>1700*1024**2:groups.append(group);group=[];size=0
            group.append(p);size+=p.stat().st_size
        if group:groups.append(group)
        records=[];assets=[]
        for i,paths in enumerate(groups,1):
            target=DEST/f'cross-mission-v2-part-{i:02d}-20261125.tar.gz';assert not target.exists()
            with tarfile.open(target,'w:gz',compresslevel=1) as archive:
                for p in paths:
                    member=str(p.relative_to(ROOT));before=p.stat();d=sha(p);archive.add(p,arcname=member,recursive=False);after=p.stat()
                    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                    records.append(dict(archive=target.name,member=member,bytes=after.st_size,sha256=d))
            assert target.stat().st_size<2*1024**3
            assets.append(dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target),members=len(paths)))
            print('Archived',target.name,len(paths),'members',target.stat().st_size,'bytes',flush=True)
        manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),assets=assets,members=records,
          scope='All v2 completion files, prepared arrays, candidate/selected/terminal weights, calibration scores/grids, predictions and event IDs; frozen sources and runtime records; nine historical transfer weights/rules; channel research; independent audits; report and workbook.',
          exclusions='Raw channel zip archives, historical non-selected checkpoint arrays, W&B internal logs and unrelated experiments are not duplicated. Their raw fingerprints and original GitHub Releases remain preserved. Existing external-drive copy is not modified; this is not a new full-project migration verification.')
        (DEST/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        checks=assets+[dict(name='artifact-manifest.json',sha256=sha(DEST/'artifact-manifest.json'))]
        (DEST/'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {r["name"]}\n' for r in checks))
    else:
        assert args.download is not None
        manifest=json.loads((args.download/'artifact-manifest.json').read_text())
        assert sha(args.download/'artifact-manifest.json')==sha(DEST/'artifact-manifest.json')
        assert (args.download/'SHA256SUMS').read_bytes()==(DEST/'SHA256SUMS').read_bytes()
        checked=[]
        for a in manifest['assets']:
            p=args.download/a['name'];assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256']
            expected={r['member']:r for r in manifest['members'] if r['archive']==a['name']}
            with tarfile.open(p,'r:gz') as archive:
                members=archive.getmembers();assert len(members)==len(expected)
                for member in members:
                    assert member.isfile() and member.name in expected
                    r=expected[member.name];assert member.size==r['bytes'] and digest(archive.extractfile(member))==r['sha256'],member.name
            checked.append(dict(**a,download_sha256_verified=True,all_member_hashes_verified=True))
            print('Fresh download verified',a['name'],len(expected),'members',flush=True)
        receipt=dict(verified=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),release_url='https://github.com/kamyarfaz/ESA-ADB/releases/tag/cross-mission-whole-system-2026-11-25',fresh_download_directory=str(args.download),assets=checked,manifest_sha256=sha(args.download/'artifact-manifest.json'),checksums_sha256=sha(args.download/'SHA256SUMS'),member_count=len(manifest['members']))
        (T/'release-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
