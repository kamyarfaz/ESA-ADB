"""Wait for the active campaign, verify it, and write its report automatically."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-pid',type=int,required=True)
    a=p.parse_args()
    root=Path('results_longrun/development/inner_isolation_cuda_v1')
    control=Path('results_longrun/development/inner_selection_cuda_v2')
    # PID plus command guards against unrelated PID reuse.
    while True:
        proc=Path(f'/proc/{a.training_pid}/cmdline')
        if not proc.exists() or b'esa_thesis.isolation_comparison' not in proc.read_bytes():break
        time.sleep(10)
    if not (root/'comparison_protocol.json').exists():
        raise RuntimeError('Training did not finish successfully; inspect /tmp/esa-isolation-run.log. Preserving all artifacts.')
    review=Path('docs/research/tables/isolation-verification-2026-10-08')
    tables=Path('docs/research/tables/channel-isolation-results-2026-10-08')
    env=dict(os.environ,PYTHONPATH='.')
    subprocess.run([sys.executable,'scripts/research/review_inner_validation.py','--source',str(root),'--output',str(review)],check=True,env=env)
    subprocess.run([sys.executable,'scripts/research/review_isolation.py','--control',str(control),'--candidate',str(root),'--candidate-review',str(review),'--output',str(tables)],check=True,env=env)
    shutil.copy2('/tmp/esa-isolation-run.log',root/'execution.log')
    summary=pd.read_csv(tables/'summary.csv')
    decision=json.loads((root/'isolation_decision.json').read_text())
    passed=decision['advancement_gate_passed']
    lines=['# Channel-isolation comparison: verified results','',
        'All six candidate runs completed. Primary policy: calibration-selected checkpoints.',
        'Control and candidate use loss weight 1, the same real periods and all three seeds.',
        'Both models have 1,333,392 parameters; diagonal channel attention changes functional capacity.',
        'No level centering or channel removal was applied.','',
        '| Fold (Jul–Dec) | Arm | Scope | Mean F0.5 | SD | Mean FP | Mean recall | Mean nominal false-alarm fraction |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f'| {r.fold} | {r.arm} | {r.scope} | {r.f05_mean:.6f} | {r.f05_sd:.6f} | {r.fp_mean:.3f} | {r.recall_mean:.6f} | {r.nominal_false_alarm_rate_mean:.6f} |')
    lines += ['',f'**Predeclared advancement gate: {"passed" if passed else "failed"}.**',
        'Gate: higher mean system F0.5 in both folds without increased mean false-event counts.',
        'The main full-period baseline is unchanged. No automatic promotion or outer-period run.',
        'Fixed-epoch-five results are secondary and were not selected after viewing validation.',
        'Internal periods were previously inspected for development; these are not untouched tests.',
        '', '## Verification', '',
        '92 tests passed before launch, including cross-channel output/gradient isolation and identical seeded parameters.',
        'Verified 30 calibration epochs, 24 candidate metric rows, both arms’ artifact manifests, paired windows/scalers/timestamps and 60 loss decompositions.',
        '', '[Protocol](channel-isolation-plan-2026-10-08.md).',
        '[All primary rows](tables/channel-isolation-results-2026-10-08/results.csv).',
        '[Verification](tables/channel-isolation-results-2026-10-08/verification.json).',
        '', '![Results](tables/channel-isolation-results-2026-10-08/comparison.png)',
        '', '**Replay warning:** instantiate ChannelIsolatedAE for candidate checkpoints; matching tensor keys do not make them joint-model checkpoints.','']
    Path('docs/research/channel-isolation-results-2026-10-08.md').write_text('\n'.join(lines))
    p=Path('AI_HANDOFF.md');s=p.read_text();key='## Start here: current user intent\n'
    note=f'''\n8 October CHANNEL ISOLATION COMPLETE AND VERIFIED by automatic completion pipeline.
All6 candidate runs complete at inner_isolation_cuda_v1. Advancement gate passed:
{passed}. No automatic promotion or outerrun. Report docs/research/channel-isolation-
results-2026-10-08.md; tables channel-isolation-results-2026-10-08. All30calepochs,
24candidate metricrows,60lossdecompositions and matching windows/scalers/timestamps
verified. Replay candidate with ChannelIsolatedAE. This supersedes ACTIVE notes
below. Main baseline unchanged; inspect report/plot before further work. No push.\n\n'''
    p.write_text(s.replace(key,key+note,1))
    p=Path('docs/README.md');p.write_text(p.read_text()+'\n- [Verified channel-isolation results](research/channel-isolation-results-2026-10-08.md).\n')
    print('VERIFIED COMPLETE; advancement gate:',passed,flush=True)


if __name__=='__main__':main()
