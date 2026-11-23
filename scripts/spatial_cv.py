"""Five-fold spatial block cross-validation for IPIM-T with a 1 km buffer.

"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, accuracy_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FEATURE_NAMES = (
    'slope', 'aspect', 'curvature', 'twi', 'profile_curvature',
    'plan_curvature', 'roughness', 'elevation', 'lithology', 'land_cover',
    'distance_roads', 'distance_streams', 'distance_faults', 'rainfall',
    'insar_los_velocity',
)
VERSION = 'R03_20251123'

# Default physical constants for the reference FoS setting.
DEFAULT_GAMMA = 20.0          # kN/m^3
DEFAULT_DEPTH = 11.0          # m
DEFAULT_FS_THRESHOLD = 1.5
DEFAULT_PROB_THRESHOLD = 0.5
DEFAULT_LAMBDA = 0.1
DEFAULT_COHESION_INIT = 20.0  # kPa
DEFAULT_FRICTION_INIT = 30.0  # deg
COHESION_MIN, COHESION_MAX = 5.0, 50.0
FRICTION_MIN, FRICTION_MAX = 20.0, 45.0


def _load_arr(root: Path, name: str):
    p = root / name
    return np.load(p, mmap_mode='r', allow_pickle=False) if p.exists() else None


@dataclass
class Dataset:
    env: np.ndarray
    insar: np.ndarray
    y: np.ndarray
    xy: np.ndarray
    rowcol: np.ndarray
    slope: np.ndarray | None
    blocks: np.ndarray | None
    groups: np.ndarray
    path: Path
    metadata: dict

    @property
    def n(self):
        return int(len(self.y))


def load_dataset(data_dir, require_slope=False) -> Dataset:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset folder missing: {root}')

    meta_path = root / 'dataset_manifest.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}

    combined = _load_arr(root, 'X.npy')
    env = _load_arr(root, 'X_env.npy')
    insar = _load_arr(root, 'X_insar.npy')
    if env is None or insar is None:
        if combined is None or combined.ndim != 4 or combined.shape[1] != 15:
            raise ValueError(
                'Require X.npy [N,15,H,W] OR X_env.npy [N,14,H,W] '
                'and X_insar.npy [N,1,H,W]'
            )
        env, insar = combined[:, :14], combined[:, 14:15]

    n = len(env)
    if env.ndim != 4 or insar.ndim != 4 or env.shape[1] != 14 or insar.shape[1] != 1:
        raise ValueError(
            'Expected channel-first [N,14,H,W] environment and [N,1,H,W] InSAR arrays'
        )
    if len(insar) != n or env.shape[-2:] != insar.shape[-2:]:
        raise ValueError('Patch dimensions or sample counts differ')
    if env.shape[-1] != env.shape[-2] or env.shape[-1] < 8 or env.shape[-1] % 8:
        raise ValueError('Require square patches with side divisible by 8 (32 for paper)')

    sample_file = root / 'samples.csv'
    frame = pd.read_csv(sample_file) if sample_file.exists() else None

    def from_array_or_column(name, cols, *, required=True):
        a = _load_arr(root, name)
        if a is not None:
            return np.asarray(a)
        if frame is not None and all(c in frame for c in cols):
            return frame[list(cols)].to_numpy()
        if required:
            raise FileNotFoundError(f'Missing {name}; or samples.csv with columns {cols}')
        return None

    y = from_array_or_column('y.npy', ['label']).reshape(-1).astype(int)
    xy = from_array_or_column('xy.npy', ['x', 'y']).astype(float)
    rc = from_array_or_column('rowcol.npy', ['row', 'col']).astype(int)
    slope = from_array_or_column('slope_deg.npy', ['slope_deg'], required=require_slope)
    blocks = from_array_or_column('block.npy', ['block'], required=False)
    groups = from_array_or_column('group_id.npy', ['group_id'], required=False)
    if groups is None:
        groups = np.arange(n)
        warnings.warn(
            'group_id.npy absent: identical-event grouping cannot be guaranteed; '
            'provide inventory-level IDs for reproducible experiments.'
        )
    groups = np.asarray(groups).reshape(-1).astype(str)
    if blocks is not None:
        blocks = np.asarray(blocks).reshape(-1).astype(int)
    if slope is not None:
        slope = np.asarray(slope).reshape(-1).astype(float)

    if n < 10 or len(y) != n or len(xy) != n or len(rc) != n or len(groups) != n:
        raise ValueError('Missing samples or sample counts do not agree')
    if xy.shape != (n, 2) or rc.shape != (n, 2):
        raise ValueError('xy/rowcol arrays must be N x 2')
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError('Labels must have both binary classes 0 and 1')
    if not np.isfinite(xy).all() or (np.max(xy, axis=0) - np.min(xy, axis=0)).max() < 100:
        raise ValueError(
            'xy must be projected coordinates in METRES with realistic extent; '
            'no lon/lat degrees'
        )
    if np.all((abs(xy[:, 0]) <= 180) & (abs(xy[:, 1]) <= 90)):
        raise ValueError('Coordinates appear to be lon/lat degrees; project them into a metric CRS')
    if slope is not None and (
        len(slope) != n or not np.isfinite(slope).all()
        or np.any((slope < 0) | (slope > 90))
    ):
        raise ValueError('Invalid slope_deg values; never use normalized slope for FoS')
    if blocks is not None and (len(blocks) != n or len(np.unique(blocks)) != 5):
        raise ValueError('block.npy must hold five distinct geographic group labels')
    if meta.get('synthetic', False):
        warnings.warn('Dataset marked SYNTHETIC; results are not KKH field-derived')

    return Dataset(env, insar, y, xy, rc, slope, blocks, groups, root, meta)


def make_blocks(d: Dataset, seed=42):
    if d.blocks is not None:
        labels = np.unique(d.blocks)
        means = {k: np.mean(d.xy[d.blocks == k, 1]) for k in labels}
        rank = {k: i for i, k in enumerate(sorted(labels, key=lambda x: means[x]))}
        return (
            np.asarray([rank[k] for k in d.blocks], dtype=int),
            'saved geographic block labels, ordered south-to-north',
        )
    cluster = KMeans(n_clusters=5, random_state=seed, n_init=10).fit(d.xy)
    labels = cluster.labels_
    rank = {k: i for i, k in enumerate(np.argsort(cluster.cluster_centers_[:, 1]))}
    return (
        np.asarray([rank[k] for k in labels], dtype=int),
        'KMeans projected-centre clusters, ordered south-to-north',
    )


def _filter_distance(candidate, reference, xy, buffer_m):
    if not len(candidate) or not len(reference):
        return np.asarray(candidate, dtype=int)
    distances = cKDTree(xy[reference]).query(xy[candidate], workers=1)[0]
    return np.asarray(candidate, dtype=int)[distances > buffer_m]


def patch_overlap_pairs(a, b, rowcol, patch_size):
    if not len(a) or not len(b):
        return 0
    aa = np.asarray(rowcol[a], dtype=int)
    bb = np.asarray(rowcol[b], dtype=int)
    tree = cKDTree(bb)
    nearby = tree.query_ball_point(aa, np.sqrt(2) * patch_size)
    return int(sum(
        np.count_nonzero(
            (np.abs(bb[ix, 0] - row[0]) < patch_size)
            & (np.abs(bb[ix, 1] - row[1]) < patch_size)
        )
        for row, ix in zip(aa, nearby) if ix
    ))


def _filter_exact_patch(candidate, reference, rc, patch_size):
    if not len(candidate) or not len(reference):
        return np.asarray(candidate, dtype=int)
    aa = rc[candidate]
    bb = rc[reference]
    tree = cKDTree(bb)
    near = tree.query_ball_point(aa, np.sqrt(2) * patch_size)
    keep = [
        not (len(ix) and np.any(
            (np.abs(bb[ix, 0] - row[0]) < patch_size)
            & (np.abs(bb[ix, 1] - row[1]) < patch_size)
        ))
        for row, ix in zip(aa, near)
    ]
    return np.asarray(candidate, dtype=int)[keep]


def _group_safe_filter(candidate, reference, groups):
    if not len(candidate) or not len(reference):
        return np.asarray(candidate, dtype=int)
    return np.asarray(candidate, dtype=int)[
        ~np.isin(groups[candidate], np.unique(groups[reference]))
    ]


def _partition_check(d, train, val, test, exact=True):
    for name, inds in [('train', train), ('validation', val), ('test', test)]:
        if not len(inds) or len(np.unique(d.y[inds])) != 2:
            raise ValueError(f'{name} partition empty or lacks a class')
    if len(set(train) & set(val)) or len(set(train) & set(test)) or len(set(val) & set(test)):
        raise AssertionError('Sample indices leak between splits')
    if any(
        set(d.groups[a]) & set(d.groups[b])
        for a, b in ((train, val), (train, test), (val, test))
    ):
        raise AssertionError('group_id leaks between splits')
    if exact and any(
        patch_overlap_pairs(a, b, d.rowcol, d.env.shape[-1])
        for a, b in ((train, val), (train, test), (val, test))
    ):
        raise AssertionError('Spatial split still has overlapping raster patches')


def make_folds(d, buffer_m=1000., seed=42, validation_mode='training_blocks', exact_guard=True):
    if buffer_m < 0:
        raise ValueError('Buffer must be non-negative')
    block, source = make_blocks(d, seed)
    for fold in range(5):
        test = np.flatnonzero(block == fold)
        candidates = np.flatnonzero(block != fold)
        candidates = _group_safe_filter(candidates, test, d.groups)
        candidates = _filter_distance(candidates, test, d.xy, buffer_m)
        if exact_guard:
            candidates = _filter_exact_patch(candidates, test, d.rowcol, d.env.shape[-1])

        if validation_mode == 'offset_block':
            val = candidates[block[candidates] == ((fold + 2) % 5)]
            train = candidates[block[candidates] != ((fold + 2) % 5)]
        elif validation_mode == 'training_blocks':
            val, train = None, None
            for attempt in range(80):
                if len(np.unique(d.groups[candidates])) < 3:
                    break
                split = GroupShuffleSplit(
                    n_splits=1, test_size=.20,
                    random_state=seed + fold * 100 + attempt,
                )
                ti, vi = next(split.split(candidates, d.y[candidates], groups=d.groups[candidates]))
                if len(np.unique(d.y[candidates[ti]])) == 2 and \
                   len(np.unique(d.y[candidates[vi]])) == 2:
                    train, val = candidates[ti], candidates[vi]
                    break
            if val is None:
                raise ValueError(
                    'Unable to form a two-class, group-disjoint validation set; '
                    'provide more data/groups'
                )
        else:
            raise ValueError('Unknown validation_mode')

        train = _group_safe_filter(train, val, d.groups)
        train = _filter_distance(train, val, d.xy, buffer_m)
        if exact_guard:
            train = _filter_exact_patch(train, val, d.rowcol, d.env.shape[-1])

        _partition_check(d, train, val, test, exact=exact_guard)
        yield {
            'fold': fold, 'train': train, 'val': val, 'test': test,
            'block': block, 'block_source': source,
            'excluded': int(len(d.y) - len(train) - len(val) - len(test)),
        }


def matched_random(d, split, seed=42):
    rng = np.random.default_rng(seed)
    out = [[], [], []]
    for cls in (0, 1):
        pool = rng.permutation(np.flatnonzero(d.y == cls))
        sizes = [int(np.count_nonzero(d.y[idx] == cls)) for idx in split]
        if sum(sizes) > len(pool):
            raise ValueError('Insufficient samples for class-matched random comparison')
        start = 0
        for j, size in enumerate(sizes):
            out[j].extend(pool[start:start + size].tolist())
            start += size
    return tuple(np.asarray(x, dtype=int) for x in out)


def split_diagnostic(d, fold, protocol, train, val, test, buffer_m):
    record = {'fold': int(fold), 'protocol': protocol, 'buffer_m': buffer_m}
    for title, x in [('train', train), ('val', val), ('test', test)]:
        record[f'{title}_n'] = len(x)
        record[f'{title}_positive'] = int(d.y[x].sum())
        record[f'{title}_negative'] = int(len(x) - d.y[x].sum())
    for title, a, b in [('train_test', train, test),
                        ('train_val', train, val),
                        ('val_test', val, test)]:
        record[f'{title}_patch_overlap_pairs'] = patch_overlap_pairs(a, b, d.rowcol, d.env.shape[-1])
        record[f'{title}_group_overlap'] = int(len(set(d.groups[a]) & set(d.groups[b])))
        if len(a) and len(b):
            record[f'{title}_min_centre_distance_m'] = float(
                cKDTree(d.xy[b]).query(d.xy[a])[0].min()
            )
    return record


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def build_model(kind='full', transformer_layers=6, transformer_dim=128,
                transformer_heads=8, use_physics=True,
                cohesion_init=DEFAULT_COHESION_INIT,
                friction_init=DEFAULT_FRICTION_INIT):
    import torch
    from torch import nn
    from models.ipim_t import IPIMT, CNNEncoder
    from types import SimpleNamespace

    if kind == 'full':
        conf = SimpleNamespace(
            TOPO_INDICES=list(range(8)),
            ENV_INDICES=list(range(15)),
            TRANSFORMER_DIM=transformer_dim,
            TRANSFORMER_HEADS=transformer_heads,
            TRANSFORMER_LAYERS=transformer_layers,
            USE_PHYSICS=use_physics,
            COHESION_INIT=cohesion_init,
            FRICTION_INIT=friction_init,
        )
        return IPIMT(conf)

    if kind == 'static':
        class StaticTwoStream(nn.Module):
            def __init__(self):
                super().__init__()
                self.topo = CNNEncoder(8)
                self.geo = CNNEncoder(6)
                self.merge = nn.Conv2d(256, 128, 1)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 1), nn.Sigmoid(),
                )

            def forward(self, env, insar):
                return self.head(
                    self.merge(torch.cat([self.topo(env[:, :8]),
                                          self.geo(env[:, 8:14])], dim=1))
                ).squeeze(-1)

        return StaticTwoStream()

    raise ValueError(f'Unsupported model kind: {kind}')


def fos_numpy(slope, c=20., phi=30., gamma=20., depth=11.):
    theta = np.deg2rad(np.asarray(slope, dtype=float))
    return (c + gamma * depth * np.cos(theta) ** 2 * np.tan(np.deg2rad(phi))) / (
        gamma * depth * np.sin(theta) * np.cos(theta) + 1.e-4
    )


def fos_torch(slope, c, phi, gamma, depth):
    import torch
    theta = torch.deg2rad(slope)
    return (
        c + gamma * depth * torch.cos(theta).square() * torch.tan(torch.deg2rad(phi))
    ) / (
        gamma * depth * torch.sin(theta) * torch.cos(theta) + 1.e-4
    )


def make_loader(d, ids, batch_size, shuffle=False):
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader

    class Loader(TorchDataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            k = int(ids[i])
            return (
                torch.from_numpy(np.asarray(d.env[k], dtype='float32').copy()),
                torch.from_numpy(np.asarray(d.insar[k], dtype='float32').copy()),
                torch.tensor(float(d.y[k]), dtype=torch.float32),
                torch.tensor(
                    float(d.slope[k]) if d.slope is not None else 0.,
                    dtype=torch.float32,
                ),
            )

    drop_singleton = bool(shuffle and len(ids) > batch_size and len(ids) % batch_size == 1)
    return DataLoader(
        Loader(), batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_singleton, num_workers=0,
    )


def select_device(device):
    torch = _torch()
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA but no GPU available')
    return device


def predict(model, loader, device):
    torch = _torch()
    model.eval()
    yy, pp = [], []
    with torch.no_grad():
        for env, insar, lab, _ in loader:
            p = model(env.to(device), insar.to(device))
            pp.extend(p.detach().cpu().reshape(-1).numpy())
            yy.extend(lab.numpy())
    return np.asarray(yy, dtype=int), np.asarray(pp, dtype=float)


def score(y, p, *, slope=None, phys=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float).reshape(-1)
    if not np.isfinite(p).all() or len(np.unique(y)) != 2:
        raise ValueError('Need valid test probabilities and both binary classes')
    binary = p >= .5
    result = {
        'auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'accuracy': float(accuracy_score(y, binary)),
        'f1': float(f1_score(y, binary, zero_division=0)),
        'mcc': float(matthews_corrcoef(y, binary)),
        'n_test': int(len(y)),
    }
    if slope is not None and phys is not None:
        fs = fos_numpy(
            slope,
            c=phys.get('c', 20.),
            phi=phys.get('phi', 30.),
            gamma=phys.get('gamma', DEFAULT_GAMMA),
            depth=phys.get('depth', DEFAULT_DEPTH),
        )
        high = binary
        result['pir_percent'] = 100 * float(
            np.sum(high & (fs > phys.get('fs_threshold', DEFAULT_FS_THRESHOLD)))
        ) / max(int(np.sum(high)), 1)
    return result


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_evaluate(
    d, tr, va, te, dest, *,
    kind='full', epochs=50, batch_size=8, lr=.001,
    weight_decay=.0001, seed=42, device='auto',
    phys=None, transformer_layers=6,
    transformer_dim=128, transformer_heads=8,
):
    torch = _torch()
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch size must be >= 1')
    if not len(tr) or not len(va) or not len(te):
        raise ValueError('Empty split')
    if any(len(np.unique(d.y[idx])) != 2 for idx in (tr, va, te)):
        raise ValueError('All partitions must contain both classes')

    use_physics = bool(kind == 'full' and phys is not None and phys.get('lam', 0) > 0)
    if use_physics and d.slope is None:
        raise ValueError('slope_deg required for physics regularization')

    phys = dict(phys or {})
    gamma = float(phys.get('gamma', DEFAULT_GAMMA))
    depth = float(phys.get('depth', DEFAULT_DEPTH))
    fs_threshold = float(phys.get('fs_threshold', DEFAULT_FS_THRESHOLD))
    prob_threshold = float(phys.get('prob_threshold', DEFAULT_PROB_THRESHOLD))
    lam = float(phys.get('lam', DEFAULT_LAMBDA))
    cohesion_init = float(phys.get('c', DEFAULT_COHESION_INIT))
    friction_init = float(phys.get('phi', DEFAULT_FRICTION_INIT))
    if use_physics and (depth <= 0 or gamma <= 0 or lam < 0):
        raise ValueError('Invalid physical settings')

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device(device)
    model_params = dict(
        kind=kind,
        transformer_layers=transformer_layers,
        transformer_dim=transformer_dim,
        transformer_heads=transformer_heads,
        use_physics=use_physics,
        cohesion_init=cohesion_init,
        friction_init=friction_init,
    )
    model = build_model(**model_params).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos = int(d.y[tr].sum())
    neg = len(tr) - pos
    pos_weight = float(neg / pos)

    train_loader = make_loader(d, tr, batch_size, shuffle=True)
    val_loader = make_loader(d, va, batch_size, shuffle=False)
    criterion = torch.nn.BCELoss(reduction='none')

    best_loss, best_state, history = float('inf'), None, []

    for epoch in range(epochs):
        model.train()
        total, seen = 0., 0
        for env, insar, lab, slope in train_loader:
            env, insar, lab, slope = [t.to(device) for t in (env, insar, lab, slope)]
            opt.zero_grad(set_to_none=True)

            p = model(env, insar).reshape(-1).clamp(1.e-6, 1 - 1.e-6)
            bce = (criterion(p, lab) * torch.where(lab > .5, pos_weight, 1.)).mean()

            if use_physics and lam > 0:
                c = torch.clamp(model.cohesion, COHESION_MIN, COHESION_MAX)
                phi = torch.clamp(model.friction, FRICTION_MIN, FRICTION_MAX)
                fs = fos_torch(slope, c, phi, gamma, depth)
                mask = ((fs > fs_threshold) & (p > prob_threshold)).float()
                penalty = lam * (mask * (p - prob_threshold).square()).mean()
            else:
                penalty = p.new_zeros(())

            loss = bce + penalty
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(lab)
            seen += len(lab)

        model.eval()
        val_loss, nv = 0., 0
        with torch.no_grad():
            for env, insar, lab, _ in val_loader:
                env, insar, lab = env.to(device), insar.to(device), lab.to(device)
                p = model(env, insar).reshape(-1).clamp(1.e-6, 1 - 1.e-6)
                weighted = criterion(p, lab) * torch.where(lab > .5, pos_weight, 1.)
                val_loss += float(weighted.sum().cpu())
                nv += len(lab)
        val_loss /= nv
        history.append({
            'epoch': epoch + 1,
            'train_loss': total / seen,
            'validation_bce': val_loss,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)

    if use_physics:
        final_c = float(torch.clamp(model.cohesion, COHESION_MIN, COHESION_MAX).detach().cpu())
        final_phi = float(torch.clamp(model.friction, FRICTION_MIN, FRICTION_MAX).detach().cpu())
    else:
        final_c, final_phi = DEFAULT_COHESION_INIT, DEFAULT_FRICTION_INIT

    y, prob = predict(model, make_loader(d, te, batch_size), device)
    metrics = score(
        y, prob,
        slope=d.slope[te] if d.slope is not None else None,
        phys={**phys, 'c': final_c, 'phi': final_phi},
    )
    metrics.update({
        'model': kind, 'seed': seed, 'epochs': epochs,
        'n_train': len(tr), 'n_val': len(va),
        'pos_weight_train': pos_weight,
        'model_parameters': sum(x.numel() for x in model.parameters()),
        'val_loss': best_loss,
        'architecture': 'IPIMT' if kind == 'full' else 'StaticTwoStream',
        'learned_c_kPa': final_c,
        'learned_phi_deg': final_phi,
        'transformer_layers': transformer_layers,
        'version': VERSION,
    })

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(dest / 'history.csv', index=False)
    pd.DataFrame({
        'sample_index': te, 'y_true': y, 'p_landslide': prob,
    }).to_csv(dest / 'heldout_predictions.csv', index=False)

    ck = {
        'model_state': best_state,
        'model_params': model_params,
        'train_idx': tr.tolist(),
        'val_idx': va.tolist(),
        'test_idx': te.tolist(),
        'pos_weight': pos_weight,
        'dataset_meta': d.metadata,
        'data_path': str(d.path),
        'phys': phys,
        'learned_c_kPa': final_c,
        'learned_phi_deg': final_phi,
        'version': VERSION,
        'model_parameter_count': metrics['model_parameters'],
    }
    torch.save(ck, dest / 'model.pt')
    (dest / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def _main():
    parser = argparse.ArgumentParser(
        description='Five-fold spatial block CV and patch-leakage audit for IPIM-T'
    )
    parser.add_argument('--data-dir', required=True, help='Exported .npy files; see README.md')
    parser.add_argument('--out', default='outputs/spatial_cv')
    parser.add_argument('--buffer-m', type=float, default=1000)
    parser.add_argument('--validation-mode',
                        choices=['training_blocks', 'offset_block'],
                        default='training_blocks')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare-random', action='store_true')
    parser.add_argument('--no-exact-patch-guard', action='store_true',
                        help='Report overlaps but do not purge extra patch-overlap candidates')
    parser.add_argument('--train', action='store_true',
                        help='Train IPIMT and a reference static two-stream comparator')
    parser.add_argument('--models', nargs='+', choices=['full', 'static'],
                        default=['full', 'static'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--transformer-layers', type=int, default=6)
    parser.add_argument('--transformer-dim', type=int, default=128)
    parser.add_argument('--transformer-heads', type=int, default=8)
    parser.add_argument('--training-seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    data = load_dataset(args.data_dir, require_slope=args.train)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    diagnostics, metrics = [], []
    folds = list(make_folds(
        data, args.buffer_m, args.seed,
        args.validation_mode,
        exact_guard=not args.no_exact_patch_guard,
    ))

    for fold in folds:
        k = fold['fold']
        if k not in args.folds:
            continue
        tr, va, te = (fold[key] for key in ('train', 'val', 'test'))

        row = split_diagnostic(data, k, 'spatial', tr, va, te, args.buffer_m)
        row['excluded'] = fold['excluded']
        diagnostics.append(row)

        if args.compare_random:
            ra, rb, rc = matched_random(data, (tr, va, te), args.seed + 100 * k)
            diagnostics.append(
                split_diagnostic(data, k, 'class_matched_RANDOM', ra, rb, rc, args.buffer_m)
            )

        np.savez_compressed(out / f'spatial_fold{k}_split.npz',
                            train_idx=tr, val_idx=va, test_idx=te)

        print(f'Fold {k}: {len(tr)} train, {len(va)} val, {len(te)} test; '
              f'patch train-test overlaps={row["train_test_patch_overlap_pairs"]}')

        if args.train:
            for kind in args.models:
                for seed in args.training_seeds:
                    dest = out / f'fold{k}' / kind / f'seed{seed}'
                    phys = ({
                        'c': DEFAULT_COHESION_INIT,
                        'phi': DEFAULT_FRICTION_INIT,
                        'gamma': DEFAULT_GAMMA,
                        'depth': DEFAULT_DEPTH,
                        'fs_threshold': DEFAULT_FS_THRESHOLD,
                        'prob_threshold': DEFAULT_PROB_THRESHOLD,
                        'lam': DEFAULT_LAMBDA,
                    } if kind == 'full' else None)
                    rec = train_evaluate(
                        data, tr, va, te, dest,
                        kind=kind, seed=seed,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, weight_decay=args.weight_decay,
                        device=args.device, phys=phys,
                        transformer_layers=args.transformer_layers,
                        transformer_dim=args.transformer_dim,
                        transformer_heads=args.transformer_heads,
                    )
                    rec.update({'fold': k, 'checkpoint': str(dest / 'model.pt')})
                    metrics.append(rec)
                    pd.DataFrame(metrics).to_csv(out / 'spatial_cv_fold_metrics.csv', index=False)

    if not diagnostics:
        raise ValueError('No folds selected')

    pd.DataFrame(diagnostics).to_csv(
        out / 'spatial_cv_overlap_random_vs_spatial.csv', index=False,
    )

    if metrics:
        df = pd.DataFrame(metrics)
        summary = []
        for kind, g in df.groupby('model'):
            per_fold = g.groupby('fold')[['auc', 'pr_auc', 'mcc', 'f1', 'accuracy']].mean()
            entry = {
                'model': kind,
                'folds': len(per_fold),
                'seeds_per_fold': len(args.training_seeds),
            }
            for col in per_fold:
                entry[col + '_mean'] = float(per_fold[col].mean())
                entry[col + '_sd_across_folds'] = (
                    float(per_fold[col].std(ddof=1)) if len(per_fold) > 1 else None
                )
            summary.append(entry)
        pd.DataFrame(summary).to_csv(out / 'spatial_cv_summary.csv', index=False)

    (out / 'RUN_STATUS.json').write_text(json.dumps({
        'version': VERSION,
        'data_dir': str(data.path),
        'data_marker': data.metadata,
        'n_total': data.n,
        'folds_run': args.folds,
        'buffer_m': args.buffer_m,
        'strict_patch_guard': not args.no_exact_patch_guard,
        'training_ran': args.train,
        'full_model': 'IPIMT',
        'static_baseline': 'static two-stream comparator',
        'analysis_protocol': 'five-fold spatial block cross-validation with 1 km buffer',
    }, indent=2), encoding='utf-8')

    print(f'Spatial CV results saved to {out}')


if __name__ == '__main__':
    _main()