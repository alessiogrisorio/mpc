"""Plot offline dei risultati MPC. Nessun solver o BRT viene ricalcolato."""
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from plot_utils import set_style, export_figure

LABELS = {
    'euclidean': 'Euclidean', 'dce': 'DCE', 'ttc': 'TTC', 'rss': 'RSS',
    'btn': 'BTN', 'stn': 'STN', 'long_lat': 'min(BTN, STN)',
    'eggert': 'Eggert', 'time_eggert': 'TBR', 'mixed': 'Mixed',
    'multi_space': 'Spatial metrics', 'multi_time': 'Temporal metrics',
    'multi_all': 'All metrics',
}
PALETTE = dict(zip(
    ['euclidean', 'dce', 'ttc', 'rss', 'btn', 'stn', 'long_lat', 'eggert', 'time_eggert', 'mixed'],
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#08519c'],
))
STATE_LABELS = [r'$X_E\ [\mathrm{m}]$', r'$Y_E\ [\mathrm{m}]$',
                r'$\psi_E\ [\mathrm{rad}]$', r'$v_E\ [\mathrm{m/s}]$',
                r'$\delta_E\ [\mathrm{rad}]$']
INPUT_LABELS = [r'$\dot\delta_E\ [\mathrm{rad/s}]$', r'$a_E\ [\mathrm{m/s^2}]$']
_HJ_CLASS = {}


def repository_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / 'mpc_multimetric_th' / 'results').is_dir():
            return folder
    raise FileNotFoundError('Apri i notebook nella cartella grafici della radice di mpc.')


def _diagnostic_class(root):
    path = (Path(root) / 'mpc_multimetric_th/hj_diagnostics.py').resolve()
    if path not in _HJ_CLASS:
        spec = importlib.util.spec_from_file_location('_thesis_plot_hj', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HJ_CLASS[path] = module.HJDiagnostics
    return _HJ_CLASS[path]


@dataclass
class Run:
    name: str
    folder: Path
    sim: dict
    diagnostic: object
    hj: dict
    metadata: dict

    @property
    def metrics(self):
        return tuple(self.diagnostic.metrics)

    @property
    def dt(self):
        return float(self.sim['dt'])

    @property
    def time(self):
        return np.arange(len(self.sim['simX'])) * self.dt

    @property
    def input_time(self):
        return self.hj['time']


def load_run(name, root=None):
    root = Path(root) if root is not None else repository_root()
    folder = root / 'mpc_multimetric_th/results' / name
    for file in ('simulation.npz', 'hj.npz'):
        if not (folder / file).is_file():
            raise FileNotFoundError(folder / file)
    with np.load(folder / 'simulation.npz', allow_pickle=False) as data:
        sim = {key: data[key].copy() for key in data.files}
    with np.load(folder / 'hj.npz', allow_pickle=False) as data:
        metadata = json.loads(data['metadata_json'].item())
    diagnostic = _diagnostic_class(root).load(folder / 'hj.npz')
    hj = diagnostic.arrays()
    n = len(sim['simU'])
    if sim['simX'].shape != (n + 1, 5) or sim['simU'].shape != (n, 2):
        raise ValueError(f'{name}: dimensioni di stati/comandi inattese.')
    if len(hj['time']) != n:
        raise ValueError(f'{name}: lunghezze simulation/HJ diverse.')
    if not np.isclose(float(sim['dt']), diagnostic.dt):
        raise ValueError(f'{name}: passi temporali diversi.')
    if not np.allclose(hj['time'], np.arange(n)*diagnostic.dt):
        raise ValueError(f'{name}: griglia temporale HJ non coerente.')
    if tuple(sim['metrics'].tolist()) != tuple(diagnostic.metrics):
        raise ValueError(f'{name}: elenco metriche diverso nei due archivi.')
    if not np.allclose(sim['simX'][:-1], hj['state'], rtol=1e-6, atol=1e-7):
        raise ValueError(f'{name}: stati diversi nei due archivi.')
    if not np.allclose(sim['simU'], hj['u'][:, 0], rtol=1e-6, atol=1e-7):
        raise ValueError(f'{name}: comandi diversi nei due archivi.')
    return Run(name, folder, sim, diagnostic, hj, metadata)


def run_manifest(runs):
    return pd.DataFrame([{
        'run': run.name, 'metrics': ', '.join(run.metrics),
        'duration_s': run.time[-1], 'dt_s': run.dt, 'steps': len(run.sim['simU']),
        'predictions': 'simX_pred' in run.sim,
        'costs': 'simOcpCost' in run.sim,
        'recorded_timers': ', '.join(k for k in ('solve_time', 'hj_time', 'controller_time') if k in run.sim),
    } for run in runs])


def compare_scenarios(runs):
    reference = runs[0]
    rows = []
    for run in runs[1:]:
        for key in ('dt', 'v_ref', 'v_H', 'X_H_initial', 'Y_H_initial'):
            rows.append({'run': run.name, 'quantity': key,
                         'same_as_reference': bool(np.allclose(run.sim[key], reference.sim[key]))})
        rows.append({'run': run.name, 'quantity': 'initial_ego_state',
                     'same_as_reference': bool(np.allclose(run.sim['simX'][0], reference.sim['simX'][0]))})
    result = pd.DataFrame(rows)
    if len(result) and not result['same_as_reference'].all():
        warnings.warn('I risultati confrontati hanno parametri di scenario diversi.')
    return result


def _finish(ax, legend=True, time_axis=True):
    if time_axis:
        ax.set_xlabel(r'$t\ [\mathrm{s}]$')
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best')
    ax.grid(False)


def _run_label(run):
    return LABELS.get(run.name, run.name)


def _selected_metrics(run, metrics):
    chosen = list(run.metrics if metrics is None else metrics)
    if not chosen or any(m not in run.metrics for m in chosen):
        raise ValueError(f'Metriche richieste non disponibili: {chosen}')
    return chosen


def plot_trajectory(run, figsize=(9, 3.8)):
    fig, ax = plt.subplots(figsize=figsize)
    xy = run.sim['simX'][:, :2]
    segments = np.stack((xy[:-1], xy[1:]), axis=1)
    line = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, run.time[-1]))
    line.set_array(run.time[:-1])
    line.set_linewidth(plt.rcParams['lines.linewidth'])
    ax.add_collection(line)
    ax.autoscale_view()
    ax.set_ylim(-2, 6)
    ax.axhline(2, color="gray", linestyle="--", linewidth=1.0, zorder=0)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, zorder=0)
    ax.axhline(4, color="gray", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlabel(r'$X_E\ [\mathrm{m}]$')
    ax.set_ylabel(r'$Y_E\ [\mathrm{m}]$')
    fig.colorbar(line, ax=ax, label=r'$t\ [\mathrm{s}]$', pad=0.025)
    return fig


def plot_trajectories(runs, figsize=(9, 3.8)):
    fig, ax = plt.subplots(figsize=figsize)
    for run in runs:
        ax.plot(run.sim['simX'][:, 0], run.sim['simX'][:, 1], label=_run_label(run))
    ax.set_ylim(-2, 6)
    ax.set_xlabel(r'$X_E\ [\mathrm{m}]$')
    ax.set_ylabel(r'$Y_E\ [\mathrm{m}]$')
    _finish(ax, time_axis=False)
    return fig


def _rectangle(ax, state, length, width, color):
    corners = np.array([[-length/2, -width/2], [length/2, -width/2],
                        [length/2, width/2], [-length/2, width/2]])
    c, s = np.cos(state[2]), np.sin(state[2])
    rotation = np.array([[c, -s], [s, c]])
    ax.add_patch(Polygon(corners @ rotation.T + state[:2], closed=True,
                         facecolor=color, edgecolor='black', linewidth=1.0, zorder=4))


def plot_frames(run, times=(5, 7, 9, 11), behind=10., ahead=25., prediction=True):
    if not len(times) or behind < 0 or ahead <= 0:
        raise ValueError('Tempi non vuoti e finestra spaziale valida richiesti.')
    if any(t < run.time[0] or t > run.time[-1] for t in times):
        raise ValueError(f'Tempi dei frame fuori dalla simulazione {run.name}.')
    fig, panels = plt.subplots(len(times), 1, figsize=(10, 1.65*len(times)), squeeze=False)
    for ax, requested in zip(panels[:, 0], times):
        i = int(np.argmin(abs(run.time-requested)))
        time = run.time[i]
        state = run.sim['simX'][i]
        human = np.array([float(run.sim['X_H_initial']) + float(run.sim['v_H'])*time,
                          float(run.sim['Y_H_initial']), 0.])
        for y in (-2, 6):
            ax.axhline(y, color="black", linestyle="-", linewidth=1.5, zorder=0)
        for y in (0, 4):
            ax.axhline(y, color="gray", linestyle=":", linewidth=1.5, zorder=0)
        ax.axhline(2., color='black', linestyle='--', linewidth=1.0)
        if prediction and 'simX_pred' in run.sim and i < len(run.sim['simX_pred']):
            pred = run.sim['simX_pred'][i]
            ax.plot(pred[:, 0], pred[:, 1], '--', color='tab:green', label='MPC prediction', zorder=3)
        _rectangle(ax, state, 4.68, 2.20, 'tab:blue')
        _rectangle(ax, human, 4.28, 1.80, 'tab:orange')
        ax.set_xlim(state[0]-behind, state[0]+ahead)
        ax.set_ylim(-2.15, 6.15)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([-2, 0, 2, 4, 6])
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_ylabel(r'$Y\ [\mathrm{m}]$')
        ax.set_title(rf'$t={time:.2f}\,\mathrm{{s}}$', loc='left', pad=5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='lower left', bbox_to_anchor=(0.18, 1.0), frameon=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    panels[-1, 0].set_xlabel(r'$X\ [\mathrm{m}]$')
    return fig


def plot_states(runs):
    runs = [runs] if isinstance(runs, Run) else list(runs)
    fig, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
    for run in runs:
        for k, ax in enumerate(axes):
            ax.plot(run.time, run.sim['simX'][:, k], label=_run_label(run))
    for ax, label in zip(axes, STATE_LABELS):
        ax.set_ylabel(label)
    axes[0].legend()
    axes[-1].set_xlabel(r'$t\ [\mathrm{s}]$')
    return fig


def plot_commands(runs):
    runs = [runs] if isinstance(runs, Run) else list(runs)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for run in runs:
        for k, ax in enumerate(axes):
            ax.step(run.input_time, run.sim['simU'][:, k], where='post', label=_run_label(run))
    for ax, label in zip(axes, INPUT_LABELS):
        ax.set_ylabel(label)
    axes[0].legend()
    axes[-1].set_xlabel(r'$t\ [\mathrm{s}]$')
    return fig


def plot_values(run, metrics=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for metric in _selected_metrics(run, metrics):
        k = run.metrics.index(metric)
        values = np.where(run.hj['active'][:, 0, k], run.hj['V'][:, k], np.nan)
        ax.plot(run.input_time, values, color=PALETTE.get(metric), label=LABELS.get(metric, metric))
    ax.axhline(0, color='black', linestyle='--', linewidth=0.9)
    ax.set_ylabel(r'$V$')
    _finish(ax, legend=False)
    if len(_selected_metrics(run, metrics)) > 3:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    else:
        ax.legend(loc='best')
    return fig


def plot_values_comparison(runs):
    # Pannelli separati: scale diverse non vengono normalizzate arbitrariamente.
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 3*len(runs)), sharex=True, squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        for k, metric in enumerate(run.metrics):
            value = np.where(run.hj['active'][:, 0, k], run.hj['V'][:, k], np.nan)
            ax.plot(run.input_time, value, color=PALETTE.get(metric), label=LABELS.get(metric, metric))
        ax.axhline(0, color='black', linestyle='--', linewidth=0.9)
        ax.set_ylabel(r'$V$')
        ax.legend()
    axes[-1, 0].set_xlabel(r'$t\ [\mathrm{s}]$')
    return fig


def _hj_window(run, stage=0, padding=1):
    available = np.flatnonzero(np.any(run.hj['active'][:, stage], axis=1))
    if not len(available):
        return (run.input_time[0], run.input_time[-1])
    return (max(run.input_time[0], run.input_time[available[0]]-padding),
            min(run.input_time[-1], run.input_time[available[-1]]+run.dt+padding))


def plot_hj_limits(run, metric=None, stage=0, softened=False, padding=1):
    if not 0 <= stage < run.diagnostic.stages:
        raise ValueError('Stage HJ non valido.')
    bounds = run.diagnostic.compute_limits(softened=softened)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, (ax, name) in enumerate(zip(axes, ('steering_rate', 'acceleration'))):
        data = bounds[name]
        if metric is None:
            lower, upper = data['lower'][:, stage], data['upper'][:, stage]
            feasible = data['feasible'][:, stage]
            active = np.any(run.hj['active'][:, stage], axis=1)
        else:
            k = run.metrics.index(metric)
            lower = np.maximum(run.diagnostic.u_lower[axis], data['metric_lower'][:, stage, k])
            upper = np.minimum(run.diagnostic.u_upper[axis], data['metric_upper'][:, stage, k])
            active = run.hj['active'][:, stage, k]
            M = run.hj['M'][:, stage, k]
            b = run.hj['b'][:, stage, k] + (run.hj['slack'][:, stage, k] if softened else 0)
            other = 1-axis
            constant = M[:, other]*run.hj['u'][:, stage, other] + b
            scale = (run.diagnostic.u_upper-run.diagnostic.u_lower)/2
            magnitude = np.max(np.abs(M)*scale, axis=-1)
            zero = abs(M[:, axis])*scale[axis] <= run.diagnostic.coefficient_tolerance*magnitude
            incompatible = active & zero & (constant < -run.diagnostic.tolerance*np.maximum(abs(constant), magnitude))
            feasible = (lower <= upper) & ~incompatible
        valid = active & feasible
        lo = np.where(valid, lower, np.nan)
        hi = np.where(valid, upper, np.nan)
        ax.fill_between(run.input_time, lo, hi, color='tab:green', alpha=.18,
                        label='HJ admissible interval')
        ax.plot(run.input_time, lo, color='tab:green', linewidth=1.)
        ax.plot(run.input_time, hi, color='tab:green', linewidth=1.)
        ax.step(run.input_time, run.hj['u'][:, stage, axis], where='post',
                color='tab:blue', label='MPC command')
        ax.axhline(run.diagnostic.u_lower[axis], color='tab:red', linestyle='--', label='Physical bounds')
        ax.axhline(run.diagnostic.u_upper[axis], color='tab:red', linestyle='--')
        if np.any(active & ~feasible):
            ax.fill_between(run.input_time, run.diagnostic.u_lower[axis],
                            run.diagnostic.u_upper[axis], where=active & ~feasible,
                            color='tab:red', alpha=.12, label='Empty HJ interval')
        ax.set_ylabel(INPUT_LABELS[axis])
        ax.set_xlim(_hj_window(run, stage, padding))
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    axes[-1].set_xlabel(r'$t\ [\mathrm{s}]$')
    return fig


def plot_command_planes(run, times=None, stage=0, softened=False):
    if times is None:
        active = np.flatnonzero(np.any(run.hj['active'][:, stage], axis=1))
        ids = (active[np.linspace(0, len(active)-1, 3).round().astype(int)]
               if len(active) else np.array([0, len(run.input_time)//2, len(run.input_time)-1]))
    else:
        if not len(times) or any(t < 0 or t > run.input_time[-1] for t in times):
            raise ValueError('Tempi del piano comandi fuori dalla simulazione.')
        ids = [int(np.argmin(abs(run.input_time-t))) for t in times]
    fig, axes = plt.subplots(1, len(ids), figsize=(4*len(ids), 4), squeeze=False)
    lo, hi = run.diagnostic.u_lower, run.diagnostic.u_upper
    for ax, i in zip(axes[0], ids):
        ax.add_patch(Rectangle(lo, *(hi-lo), facecolor='none', edgecolor='tab:red', linestyle='--'))
        polygon = run.diagnostic.control_polygon(i, stage, softened=softened)
        if len(polygon) >= 3:
            ax.fill(polygon[:, 0], polygon[:, 1], color='tab:green', alpha=.18)
        elif len(polygon):
            ax.plot(polygon[:, 0], polygon[:, 1], color='tab:green', marker='o')
        for k, metric in enumerate(run.metrics):
            if not run.hj['active'][i, stage, k]:
                continue
            M = run.hj['M'][i, stage, k]
            b = run.hj['b'][i, stage, k] + (run.hj['slack'][i, stage, k] if softened else 0)
            if abs(M[1]) > 1e-12:
                xx = np.linspace(lo[0], hi[0], 100)
                ax.plot(xx, -(M[0]*xx+b)/M[1], color=PALETTE.get(metric), label=LABELS.get(metric, metric))
            elif abs(M[0]) > 1e-12:
                ax.axvline(-b/M[0], color=PALETTE.get(metric), label=LABELS.get(metric, metric))
        ax.scatter(*run.hj['u'][i, stage], color='tab:blue', zorder=5, s=25)
        ax.set_xlim(lo[0]-.005, hi[0]+.005)
        ax.set_ylim(lo[1]-.25, hi[1]+.25)
        ax.set_xlabel(INPUT_LABELS[0])
        ax.set_ylabel(INPUT_LABELS[1])
        ax.set_title(rf'$t={run.input_time[i]:.2f}\,\mathrm{{s}}$')
    handles = [Line2D([], [], color=PALETTE.get(m), label=LABELS.get(m, m)) for m in run.metrics]
    handles += [Line2D([], [], color='tab:red', linestyle='--', label='Physical bounds'),
                Line2D([], [], color='tab:blue', marker='o', linestyle='none', label='MPC command')]
    fig.legend(handles=handles, loc='outside upper center', ncol=min(4, len(handles)))
    return fig


def plot_costs(run):
    if 'simOcpCost' not in run.sim:
        raise ValueError(f'{run.name}: costi non registrati.')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(run.input_time, run.sim['simOcpCost'], color='black', label='OCP objective')
    axes[0].set_ylabel(r'$J$')
    axes[0].legend()
    parts = {k[6:]: v for k, v in run.sim.items() if k.startswith('cost__')}
    for name, value in parts.items():
        axes[1].plot(run.input_time, value, label=name)
    axes[1].set_ylabel('Cost contribution')
    axes[1].set_xlabel(r'$t\ [\mathrm{s}]$')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    return fig


def cost_consistency(run):
    parts = [v for k, v in run.sim.items() if k.startswith('cost__')]
    if not parts or 'simOcpCost' not in run.sim:
        return {'run': run.name, 'max_abs_decomposition_error': np.nan}
    return {'run': run.name, 'max_abs_decomposition_error': float(np.max(abs(sum(parts)-run.sim['simOcpCost'])))}


def timing_statistics(runs, control_frequency=50.):
    if control_frequency <= 0:
        raise ValueError('La frequenza di controllo deve essere positiva.')
    deadline = 1000/control_frequency
    rows = []
    for run in runs:
        for key, label in [('solve_time', 'OCP solver'), ('hj_time', 'HJ preparation'),
                           ('controller_time', 'Controller total')]:
            if key not in run.sim:
                continue
            ms = np.asarray(run.sim[key], dtype=float)*1000
            if ms.ndim != 1 or len(ms) != len(run.sim['simU']) or not np.isfinite(ms).all() or np.any(ms < 0):
                raise ValueError(f'{run.name}: timer {key} non valido.')
            rows.append({
                'run': run.name, 'timer': label, 'samples': len(ms), 'mean_ms': ms.mean(),
                'median_ms': np.median(ms), 'p95_ms': np.percentile(ms, 95),
                'p99_ms': np.percentile(ms, 99), 'max_ms': ms.max(),
                'deadline_ms': deadline, 'within_deadline_pct': 100*np.mean(ms <= deadline),
                'deadline_misses': int(np.count_nonzero(ms > deadline)),
            })
    return pd.DataFrame(rows)


def plot_timings(runs, control_frequency=50.):
    fig, ax = plt.subplots(figsize=(9, 4))
    for run in runs:
        if 'solve_time' in run.sim:
            ax.plot(run.input_time, run.sim['solve_time']*1000, label=_run_label(run))
    ax.axhline(1000/control_frequency, color='tab:red', linestyle='--', label='Control period')
    ax.set_ylabel('OCP solve time [ms]')
    _finish(ax)
    return fig


def export_collection(figures, group, formats=('pdf',), tables=None):
    paths = []
    for name, fig in figures.items():
        paths.extend(export_figure(fig, name, group, formati=formats, sovrascrivi=True))
    folder = Path(__file__).resolve().parent / 'output' / group
    if tables:
        folder.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            path = folder / f'{name}.csv'
            table.to_csv(path, index=False)
            paths.append(path)
    return paths
