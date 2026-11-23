"""Held-out channel-wise permutation importance.

Swaps entire HxW input channels between held-out samples, repeats the
permutation, and aggregates AUC loss across folds and seeds. Uses checkpoints
produced by scripts/spatial_cv.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spatial_cv import (
    load_dataset, build_model, select_device, FEATURE_NAMES, VERSION,
)


def load_model(path, device):
    import torch
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        raise RuntimeError('PyTorch with weights_only support required; upgrade PyTorch')
    for key in ('model_params', 'model_state', 'test_idx', 'version'):
        if key not in ck:
            raise ValueError(f'Checkpoint missing {key}')
    if ck['version'] != VERSION:
        raise ValueError(
            f'Checkpoint version {ck["version"]!r} differs from {VERSION}'
        )
    model = build_model(**ck['model_params'])
    model.load_state_dict(ck['model_state'])
    model.to(device).eval()
    return model, ck


def predict_patches(model, env, insar, device, batch_size):
    import torch
    n = len(env)
    values = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            e = torch.as_tensor(
                np.asarray(env[i:i + batch_size], dtype=np.float32).copy(),
                device=device,
            )
            s = torch.as_tensor(
                np.asarray(insar[i:i + batch_size], dtype=np.float32).copy(),
                device=device,
            )
            values.append(model(e, s).reshape(-1).detach().cpu().numpy())
    return np.concatenate(values)


def importance_for_checkpoint(data, checkpoint, *, device='cpu',
                              repeats=3, seed=42, batch_size=8):
    model, ck = load_model(checkpoint, device)
    te = np.asarray(ck['test_idx'], dtype=int)
    if len(te) < 2 or te.min() < 0 or te.max() >= data.n or len(np.unique(te)) != len(te):
        raise ValueError('Invalid or duplicated held-out indices in checkpoint')
    yt = np.asarray(data.y[te]).astype(int)
    if len(np.unique(yt)) != 2:
        raise ValueError('Cannot calculate ROC AUC on a one-class test fold')

    env = np.asarray(data.env[te], dtype=np.float32).copy()
    ins = np.asarray(data.insar[te], dtype=np.float32).copy()

    baseline = float(roc_auc_score(yt, predict_patches(model, env, ins, device, batch_size)))

    rng = np.random.default_rng(seed)
    rows = []
    for k, name in enumerate(FEATURE_NAMES):
        for rep in range(repeats):
            order = rng.permutation(len(te))
            if k < 14:
                e = env.copy()
                e[:, k, :, :] = env[order, k, :, :]
                ss = ins
            else:
                ss = ins.copy()
                ss[:, 0, :, :] = ins[order, 0, :, :]
                e = env
            score = float(roc_auc_score(yt, predict_patches(model, e, ss, device, batch_size)))
            rows.append({
                'feature': name,
                'channel': k,
                'repeat': rep + 1,
                'baseline_AUC': baseline,
                'permuted_AUC': score,
                'AUC_drop': baseline - score,
                'checkpoint': str(checkpoint),
                'n_test': len(te),
            })
    return rows


def _main():
    p = argparse.ArgumentParser(
        description='Per-fold held-out spatial-channel permutation importance'
    )
    p.add_argument('--data-dir', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', help='One checkpoint saved by spatial_cv.py')
    group.add_argument('--cv-root',
                       help='Spatial CV output directory containing fold*/full/seed*/model.pt')
    p.add_argument('--out', default='outputs/permutation_importance')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()

    if a.repeats < 1 or a.batch_size < 1:
        raise ValueError('repeats and batch-size must be positive')

    d = load_dataset(a.data_dir)
    device = select_device(a.device)

    if a.checkpoint:
        paths = [Path(a.checkpoint)]
    else:
        paths = sorted(Path(a.cv_root).glob('fold*/full/seed*/model.pt'))
    if not paths or not all(x.is_file() for x in paths):
        raise FileNotFoundError(
            'No full-model checkpoints found. Run scripts/spatial_cv.py --train first.'
        )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for num, path in enumerate(paths):
        print(f'Permutation importance: {num + 1}/{len(paths)} {path}')
        current = importance_for_checkpoint(
            d, path, device=device, repeats=a.repeats,
            seed=a.seed + 100 * num, batch_size=a.batch_size,
        )
        fold_name = next(
            (q for q in path.parts if q.startswith('fold') and q[4:].isdigit()),
            'single',
        )
        seed_name = path.parent.name
        for r in current:
            r.update({'fold': fold_name, 'train_seed': seed_name})
        rows.extend(current)
        pd.DataFrame(rows).to_csv(out / 'permutations.csv', index=False)

    df = pd.DataFrame(rows)

    per_seed = df.groupby(
        ['fold', 'train_seed', 'feature', 'channel'], as_index=False,
    ).agg(
        mean_AUC_drop=('AUC_drop', 'mean'),
        repeat_sd=('AUC_drop', 'std'),
    )
    per_seed.to_csv(out / 'by_fold_seed.csv', index=False)

    per_fold = per_seed.groupby(
        ['fold', 'feature', 'channel'], as_index=False,
    ).agg(mean_AUC_drop=('mean_AUC_drop', 'mean'))
    per_fold.to_csv(out / 'per_fold.csv', index=False)

    summary = per_fold.groupby(
        ['feature', 'channel'], as_index=False,
    ).agg(
        AUC_drop_mean=('mean_AUC_drop', 'mean'),
        AUC_drop_sd_across_folds=('mean_AUC_drop', 'std'),
        n_folds=('fold', 'nunique'),
    )
    summary.sort_values('AUC_drop_mean', ascending=False).to_csv(
        out / 'importance_summary.csv', index=False,
    )

    (out / 'RUN_STATUS.json').write_text(json.dumps({
        'script_version': VERSION,
        'n_checkpoints': len(paths),
        'repeats': a.repeats,
        'permutation_unit':
            'One entire HxW input channel patch, swapped between held-out samples',
        'aggregation':
            'Mean of repeats per model seed; mean of seeds per fold; mean+SD of fold means',
    }, indent=2), encoding='utf-8')

    print(f'Saved: {out / "importance_summary.csv"}')


if __name__ == '__main__':
    _main()