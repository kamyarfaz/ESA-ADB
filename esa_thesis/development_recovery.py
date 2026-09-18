"""Atomic epoch checkpoints and strict AE development recovery checks."""
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Cannot restore a CUDA run without CUDA')
        torch.cuda.set_rng_state_all(state['cuda'])


def check_protocol(output, expected, resume):
    """Keep the original manifest; record a verified recovery implementation separately."""
    output = Path(output)
    expected = json.loads(json.dumps(expected, default=str))
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        (output/'protocol.json').write_text(json.dumps(expected, indent=2))
        (output/'resume_protocol.json').write_text(json.dumps(expected, indent=2))
        return
    path = output/'resume_protocol.json'
    legacy = not path.exists()
    previous = json.loads((output/'protocol.json' if legacy else path).read_text())
    if legacy:
        # The only intentional algorithm-file change is checkpoint plumbing in development.py.
        ignored = {'development.py', 'development_recovery.py'}
        previous['source_sha256'] = {k: v for k, v in previous['source_sha256'].items() if k not in ignored}
        compare = {**expected, 'source_sha256': {k: v for k, v in expected['source_sha256'].items() if k not in ignored}}
    else:
        compare = expected
    if previous != compare:
        raise ValueError('Recovery protocol differs: preserve data, settings, device, folds and source code; use a new output for a changed experiment')
    if legacy:
        path.write_text(json.dumps(expected, indent=2))


def prepare_fold(output, resume, restart_incomplete):
    output = Path(output)
    required = ['results.json', 'model_checkpoint.pt', 'calibration_candidates.csv',
                'history.csv', 'frozen_rule.json', 'assessment_score.npy', 'assessment_prediction.npy']
    if resume and all((output/name).is_file() for name in required):
        result = json.loads((output/'results.json').read_text())
        return result
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError('Refusing existing fold output')
        if not (output/'checkpoint_last.pt').exists():
            if not restart_incomplete:
                raise ValueError(f'{output}: no full epoch checkpoint. Use --restart-incomplete to archive this partial fold and retrain it from scratch')
            archive = output.parent/'interrupted'
            archive.mkdir(exist_ok=True)
            target = archive/(output.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            output.rename(target)
            print(f'Preserved incomplete fold at {target}; restarting from epoch 1', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    return None
