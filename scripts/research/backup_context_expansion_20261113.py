"""Archive immutable evidence, then verify fresh Release downloads and members."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results_longrun/context_expansion_20261104_v1'
TABLE=ROOT/'docs/research/tables/context-expansion-2026-11-04'
DEST=Path('/home/k_faz/project-backups/context-expansion-20261104/release-20261113')
METHODS=['short_no_roll','pretrain_finetune','expanded_short_no_roll']


def digest(stream):
    h=hashlib.sha256()
    for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def sha(p):
    with p.open('rb') as f:return digest(f)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','verify'])
    parser.add_argument('--download',type=Path);args=parser.parse_args()
    if args.mode=='build':
        for name in ['verification.json','wandb-cloud-verification.json','workbook-verification.json']:
            assert json.loads((TABLE/name).read_text())['verified']
        DEST.mkdir(parents=True,exist_ok=True)
        complete=json.loads((RUN/'complete.json').read_text())
        files={RUN/name for name in complete['files']}|{RUN/'complete.json'}
        groups={method:sorted(p for p in files if p.relative_to(RUN).parts[0].startswith(method+'_s')) for method in METHODS}
        allocated=set().union(*map(set,groups.values()))
        plan=json.loads((RUN/'protocol.json').read_text())
        provenance=(files-allocated)|{ROOT/name for section in ['source_sha256','annotations','dependencies'] for name in plan[section]}
        provenance|={p for p in TABLE.rglob('*') if p.is_file()}
        provenance|={ROOT/'data/frozen/mission1-reference-v1/manifest.json',ROOT/'requirements-test.txt',ROOT/'README.md',ROOT/'AGENTS.md',ROOT/'MIGRATION.md',ROOT/'AI_HANDOFF.md',ROOT/'ESA_thesis_results_full_and_historical.xlsx',ROOT/'docs/research/context-expansion-results-2026-11-13.md',ROOT/'tests/thesis/test_context_expansion_verifier.py'}
        provenance|=set((ROOT/'scripts/research').glob('*context_expansion_20261113.py'))
        provenance|={ROOT/'outputs/context-expansion-20261113'/n for n in ['workbook.mjs','workbook-before.xlsx','workbook-data.json','pooled.png','annual.png','notes.png']}
        groups['provenance']=sorted(provenance)
        records=[];assets=[]
        for group,paths in groups.items():
            target=DEST/f'context-expansion-{group}-20261113.tar.gz'
            assert not target.exists(),'Refuse overwrite of existing archive'
            with tarfile.open(target,'w:gz',compresslevel=1) as archive:
                for path in paths:
                    member=str(path.relative_to(ROOT));before=path.stat();h=sha(path)
                    archive.add(path,arcname=member,recursive=False)
                    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                    records.append(dict(archive=target.name,member=member,bytes=after.st_size,sha256=h))
            assets.append(dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target),members=len(paths)))
            print('Archived',group,len(paths),'files',target.stat().st_size,'bytes',flush=True)
        value=dict(created_utc=datetime.now(timezone.utc).isoformat(),assets=assets,members=records,
                   scope='All 600 non-W&B campaign files, all evaluated/selected/terminal weights, scores, predictions and grids; frozen source, training channel-selection inputs, original annotations, audit/report/workbook',
                   exclusions='Prepared train/calibration/testing CSVs and older experiments are not duplicated; their frozen manifest is included. Existing external-drive copy is preserved; this is not a full-project migration or a new external-drive verification.')
        (DEST/'artifact-manifest.json').write_text(json.dumps(value,indent=2)+'\n')
        assets.append(dict(name='artifact-manifest.json',sha256=sha(DEST/'artifact-manifest.json')))
        (DEST/'SHA256SUMS').write_text(''.join(f'{a["sha256"]}  {a["name"]}\n' for a in assets))
    else:
        assert args.download is not None
        manifest=json.loads((args.download/'artifact-manifest.json').read_text())
        assert sha(args.download/'artifact-manifest.json')==sha(DEST/'artifact-manifest.json')
        assert (args.download/'SHA256SUMS').read_bytes()==(DEST/'SHA256SUMS').read_bytes()
        checked=[]
        for asset in manifest['assets']:
            p=args.download/asset['name'];assert p.stat().st_size==asset['bytes'] and sha(p)==asset['sha256']
            expected={r['member']:r for r in manifest['members'] if r['archive']==asset['name']}
            with tarfile.open(p,'r:gz') as archive:
                members=archive.getmembers();assert len(members)==len(expected)
                for member in members:
                    assert member.isfile() and member.name in expected
                    r=expected[member.name];assert member.size==r['bytes']
                    assert digest(archive.extractfile(member))==r['sha256'],member.name
            checked.append(dict(**asset,download_sha256_verified=True,all_member_hashes_verified=True))
            print('Fresh download verified',asset['name'],len(expected),'members',flush=True)
        receipt=dict(verified=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),
                     release_url='https://github.com/kamyarfaz/ESA-ADB/releases/tag/context-expansion-2026-11-13',
                     fresh_download_directory=str(args.download),assets=checked,
                     manifest_sha256=sha(args.download/'artifact-manifest.json'),
                     checksums_sha256=sha(args.download/'SHA256SUMS'),member_count=len(manifest['members']))
        (TABLE/'release-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
