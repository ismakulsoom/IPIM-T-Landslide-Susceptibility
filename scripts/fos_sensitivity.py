"""One-factor-at-a-time FoS sensitivity analysis.


"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from spatial_cv import (
    load_dataset, train_evaluate, VERSION,
    DEFAULT_GAMMA, DEFAULT_DEPTH, DEFAULT_FS_THRESHOLD,
    DEFAULT_PROB_THRESHOLD, DEFAULT_LAMBDA,
    DEFAULT_COHESION_INIT, DEFAULT_FRICTION_INIT,
)

BASE = dict(
    c=DEFAULT_COHESION_INIT,
    phi=DEFAULT_FRICTION_INIT,
    gamma=DEFAULT_GAMMA,
    depth=DEFAULT_DEPTH,
    fs_threshold=DEFAULT_FS_THRESHOLD,
    prob_threshold=DEFAULT_PROB_THRESHOLD,
    lam=DEFAULT_LAMBDA,
)


def parameter_grid():
    yield 'reference', BASE.copy()
    for x in (1.2, 1.3, 2.0):
        yield f'fs_threshold_{x}', dict(BASE, fs_threshold=x)
    for x in (2., 5., 20.):
        yield f'depth_{int(x)}', dict(BASE, depth=x)
    for x in (.01, .05, .5):
        yield f'lambda_{x}', dict(BASE, lam=x)


def check_split(source, n, y):
    with np.load(source, allow_pickle=False) as z:
        keys = ('train_idx', 'val_idx', 'test_idx')
        if not all(x in z for x in keys):
            raise ValueError(f'Saved split {source} needs keys {keys}')
        tr, va, te = (np.asarray(z[k], dtype=int).reshape(-1) for k in keys)
    if min(map(len, (tr, va, te))) == 0:
        raise ValueError('Empty saved split')
    if np.min(np.r_[tr, va, te]) < 0 or np.max(np.r_[tr, va, te]) >= n:
        raise ValueError('Invalid split indices')
    if len(np.unique(np.r_[tr, va, te])) != len(tr) + len(va) + len(te):
        raise ValueError('Saved split has duplicate indices across or within partitions')
    for idx in (tr, va, te):
        if len(np.unique(y[idx])) != 2:
            raise ValueError('Each saved partition must contain both classes')
    return tr, va, te


def _main():
    p = argparse.ArgumentParser(
        description='One-factor FoS sensitivity (Table S4 settings)'
    )
    p.add_argument('--data-dir', required=True,
                   help='Original .npy patches and RAW slope_deg')
    p.add_argument('--split-npz', required=True,
                   help='Saved train/val/test indices')
    p.add_argument('--split-protocol',
                   choices=['original_random', 'spatial_fold', 'other'],
                   required=True,
                   help='Use original_random with the saved random split to reproduce Table S4')
    p.add_argument('--out', default='outputs/fos_sensitivity')
    p.add_argument('--only', nargs='+', default=None,
                   help='Subset setting tags for a smoke test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--transformer-layers', type=int, default=6)
    p.add_argument('--transformer-dim', type=int, default=128)
    p.add_argument('--transformer-heads', type=int, default=8)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()

    d = load_dataset(a.data_dir, require_slope=True)
    tr, va, te = check_split(a.split_npz, d.n, d.y)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    all_names = [name for name, _ in parameter_grid()]
    if a.only and not set(a.only).issubset(set(all_names)):
        raise ValueError(f'Invalid settings requested: {set(a.only) - set(all_names)}')

    records = []
    for setting, phys in parameter_grid():
        if a.only and setting not in a.only:
            continue
        print(f'FoS sweep: training {setting}; '
              f'depth={phys["depth"]}, FS={phys["fs_threshold"]}, lambda={phys["lam"]}')
        m = train_evaluate(
            d, tr, va, te, out / setting,
            kind='full', epochs=a.epochs, batch_size=a.batch_size,
            lr=a.lr, weight_decay=a.weight_decay, seed=a.seed,
            device=a.device, phys=phys,
            transformer_layers=a.transformer_layers,
            transformer_dim=a.transformer_dim,
            transformer_heads=a.transformer_heads,
        )
        m.update({
            'setting': setting,
            'depth_m': phys['depth'],
            'fs_threshold': phys['fs_threshold'],
            'lambda': phys['lam'],
            'cohesion_init_kPa': phys['c'],
            'friction_init_deg': phys['phi'],
            'gamma_kN_m3': phys['gamma'],
            'split_protocol': a.split_protocol,
            'split_file': str(Path(a.split_npz).resolve()),
        })
        records.append(m)
        pd.DataFrame(records).to_csv(out / 'fos_sensitivity_results.csv', index=False)

    if not records:
        raise ValueError('No FoS settings processed')

    (out / 'RUN_STATUS.json').write_text(json.dumps({
        'script_version': VERSION,
        'n_settings': len(records),
        'full_one_factor_sweep': len(records) == 10,
        'split_protocol': a.split_protocol,
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'param_grid': BASE,
    }, indent=2), encoding='utf-8')

    print(f'FoS sensitivity results saved to {out}')


if __name__ == '__main__':
    _main()