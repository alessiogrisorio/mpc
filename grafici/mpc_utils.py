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
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from plot_utils import set_style, export_figure

LABELS = {
    'euclidean': 'Euclidean', 'dce': 'DCE', 'ttc': 'TTC', 'rss': 'RSS',
    'btn': 'BTN', 'stn': 'STN', 'long_lat': 'min(BTN, STN)',
    'eggert': 'PRM', 'time_eggert': 'TBR', 'mixed': 'Mixed',
    'multi_space': 'Spatial metrics', 'multi_time': 'Temporal metrics',
    'multi_all': 'All metrics', 'tbr': 'TBR',
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
    ax.set_xlabel(r'$X\ [\mathrm{m}]$')
    ax.set_ylabel(r'$Y\ [\mathrm{m}]$')
    fig.colorbar(line, ax=ax, label=r'$t\ [\mathrm{s}]$', pad=0.025)
    return fig


def plot_trajectories(runs, figsize=(9, 3.8)):
    fig, ax = plt.subplots(figsize=figsize)
    for run in runs:
        ax.plot(run.sim['simX'][:, 0], run.sim['simX'][:, 1], label=_run_label(run))
    ax.set_ylim(-2, 6)
    for y in (0, 4):
        ax.axhline(y, color='gray', linestyle=':', linewidth=1.0, zorder=0)
    ax.axhline(2, color='gray', linestyle='--', linewidth=1.0, zorder=0)
    ax.set_xlabel(r'$X\ [\mathrm{m}]$')
    ax.set_ylabel(r'$Y\ [\mathrm{m}]$')
    _finish(ax, time_axis=False)
    return fig


def _rectangle(ax, state, length, width, color, zorder=4):
    corners = np.array([[-length/2, -width/2], [length/2, -width/2],
                        [length/2, width/2], [-length/2, width/2]])
    c, s = np.cos(state[2]), np.sin(state[2])
    rotation = np.array([[c, -s], [s, c]])
    ax.add_patch(Polygon(corners @ rotation.T + state[:2], closed=True,
                         facecolor=color, edgecolor='black', linewidth=1.0, zorder=zorder))


def _fixed_brt_slice(values, coordinates, fixed, theta_period):
    """Interpolate the four fixed coordinates, including periodic heading."""
    brackets = []
    for offset, value in enumerate(fixed):
        axis = coordinates[offset + 2]
        if offset == 0:
            value = axis[0] + (value - axis[0]) % theta_period
            if axis[-1] < axis[0] + theta_period - 1e-10:
                extended = np.append(axis, axis[0] + theta_period)
                high = min(np.searchsorted(extended, value, side='right'), len(axis))
                low = high - 1
                weight = (value - extended[low]) / (extended[high] - extended[low])
                brackets.append(((low, 1-weight), (high % len(axis), weight)))
                continue
        if not np.isfinite(value) or value < axis[0]-1e-9 or value > axis[-1]+1e-9:
            return None
        value = np.clip(value, axis[0], axis[-1])
        if len(axis) == 1:
            brackets.append(((0, 1.),))
        else:
            high = int(np.clip(np.searchsorted(axis, value, side='right'), 1, len(axis)-1))
            low = high-1
            weight = (value-axis[low]) / (axis[high]-axis[low])
            brackets.append(((low, 1-weight), (high, weight)))
    from itertools import product
    result = np.zeros(values.shape[:2], dtype=float)
    for combination in product(*brackets):
        indices, weights = zip(*combination)
        weight = float(np.prod(weights))
        if weight:
            result += weight * values[(slice(None), slice(None), *indices)]
    return result


def _frame_brt_slices(run, ids, metrics, brt_dir, smoothing, grid):
    """Load one value array at a time; gradients are not loaded."""
    result = {}
    for metric in metrics:
        k = run.metrics.index(metric)
        active_ids = [i for i in dict.fromkeys(ids)
                      if i < len(run.input_time) and run.hj['active'][i, 0, k]]
        if not active_ids:
            continue
        path = brt_dir / f'{metric}.npz'
        if not path.is_file():
            raise FileNotFoundError(
                f'BRT mancante: {path}. Specifica brt_dir oppure brt_metrics=[] '
                'per esportare frame senza contorni.')
        with np.load(path, allow_pickle=False) as data:
            coordinates = [np.asarray(data[name], dtype=float) for name in
                           ('x_rel', 'y_rel', 'theta_rel', 'v_H', 'delta_E', 'v_E')]
            values = data['BRT']
            if values.shape != tuple(len(axis) for axis in coordinates):
                raise ValueError(f'{path}: forma BRT non coerente con le coordinate.')
            if any(len(a) == 0 or not np.all(np.diff(a) > 0) for a in coordinates):
                raise ValueError(f'{path}: coordinate non strettamente crescenti.')
            period = float(data['grid_hi'][2] - data['grid_lo'][2])
            if period <= 0:
                raise ValueError(f'{path}: periodo angolare non valido.')
            xx, yy = np.meshgrid(
                np.linspace(coordinates[0][0], coordinates[0][-1], grid[0]),
                np.linspace(coordinates[1][0], coordinates[1][-1], grid[1]))
            points = np.column_stack((xx.ravel(), yy.ravel()))
            for i in active_ids:
                fixed = np.asarray(run.hj['relative_state'][i, 2:6], dtype=float)
                sliced = _fixed_brt_slice(values, coordinates, fixed, period)
                if sliced is None:
                    warnings.warn(f'{run.name}, {metric}, t={run.time[i]:.2f}: '
                                  'sezione fuori dal dominio BRT; contorno omesso.')
                    continue
                if not np.isfinite(sliced).all():
                    raise ValueError(f'{path}: sezione BRT con valori non finiti.')
                dense = RegularGridInterpolator(
                    tuple(coordinates[:2]), sliced, bounds_error=True)(points).reshape(xx.shape)
                # Smooth after resampling to round the interpolated grid corners.
                # Sigma remains expressed in ORIGINAL spatial grid cells.
                if smoothing:
                    sigma = (smoothing*(grid[1]-1)/(len(coordinates[1])-1),
                             smoothing*(grid[0]-1)/(len(coordinates[0])-1))
                    dense = gaussian_filter(dense, sigma=sigma, mode='nearest')
                if dense.min() < 0 < dense.max():
                    result[i, metric] = (xx, yy, dense)
    return result


def plot_frames(run, times=(5, 7, 9, 11), behind=10., ahead=25., prediction=True,
                ncols=1, *, brt_metrics=None, brt_smoothing=0.5,
                brt_grid=(241, 161), brt_dir=None):
    """Global frames; brt_metrics=None draws all active metrics, [] none.

    brt_smoothing is Gaussian sigma in original x/y grid cells; 0 disables it.
    Only the recorded current-state HJ activation (stage 0) enables contours.
    """
    if not len(times) or behind < 0 or ahead <= 0:
        raise ValueError('Tempi non vuoti e finestra spaziale valida richiesti.')
    if not isinstance(ncols, (int, np.integer)) or ncols < 1:
        raise ValueError('ncols deve essere un intero positivo.')
    if any(not np.isfinite(t) or t < run.time[0] or t > run.time[-1] for t in times):
        raise ValueError(f'Tempi dei frame fuori dalla simulazione {run.name}.')
    if not np.isfinite(brt_smoothing) or brt_smoothing < 0:
        raise ValueError('brt_smoothing deve essere non negativo.')
    if len(brt_grid) != 2 or any(int(n) != n or n < 2 for n in brt_grid):
        raise ValueError('brt_grid deve contenere due interi maggiori di uno.')

    chosen = list(run.metrics) if brt_metrics is None else list(brt_metrics)
    if any(m not in run.metrics for m in chosen):
        raise ValueError(f'Metriche BRT non presenti nella simulazione: {chosen}')
    chosen = list(dict.fromkeys(chosen))

    brt_dir = Path(brt_dir) if brt_dir is not None else run.folder.parents[2] / 'brt'
    ids = [int(np.argmin(abs(run.time-t))) for t in times]
    contours = _frame_brt_slices(
        run, ids, chosen, brt_dir, brt_smoothing, brt_grid
    )

    nrows = int(np.ceil(len(times)/ncols))
    fig, panels = plt.subplots(
        nrows, ncols,
        figsize=(10*ncols, 3*nrows),
        squeeze=False,
    )
    axes = panels.ravel()
    for ax in axes:
        ax.tick_params(axis='both', labelsize=22)
        ax.xaxis.label.set_fontsize(24)
        ax.yaxis.label.set_fontsize(24)
        ax.title.set_fontsize(24)
        ax._left_title.set_fontsize(24)
    linewidth = plt.rcParams['lines.linewidth']
    drawn_metrics = set()
    predictions_drawn = False

    for ax, i in zip(axes, ids):
        time = run.time[i]
        state = run.sim['simX'][i]
        human_speed = float(run.sim['v_H'])
        human = np.array([
            float(run.sim['X_H_initial']) + human_speed*time,
            float(run.sim['Y_H_initial']),
            0.,
        ])

        if i > 0:
            ax.plot(
                [float(run.sim['X_H_initial']), human[0]],
                [human[1], human[1]],
                color='tab:orange',
                linestyle='-',
                linewidth=linewidth,
                zorder=.2,
            )

        _rectangle(ax, human, 4.28, 1.80, 'tab:orange', zorder=.4)

        for y in (-2, 6):
            ax.axhline(
                y, color='black', linestyle='-',
                linewidth=1.5, zorder=.6,
            )
        for y in (0, 4):
            ax.axhline(
                y, color='gray', linestyle=':',
                linewidth=1.5, zorder=.6,
            )
        ax.axhline(
            2., color='black', linestyle='--',
            linewidth=1.0, zorder=.6,
        )

        if i > 0:
            history = run.sim['simX'][:i+1]
            ax.plot(
                history[:, 0], history[:, 1],
                '-', color='tab:blue',
                linewidth=linewidth, zorder=2.5,
            )

        if prediction and 'simX_pred' in run.sim and i < len(run.sim['simX_pred']):
            pred = run.sim['simX_pred'][i]
            horizon = (len(pred)-1) * run.diagnostic.stage_dt

            ax.plot(
                [human[0], human[0]+human_speed*horizon],
                [human[1], human[1]],
                '--', color='tab:orange',
                linewidth=linewidth, zorder=.3,
            )
            ax.plot(
                pred[:, 0], pred[:, 1],
                '--', color='black',
                linewidth=linewidth, zorder=3,
            )
            predictions_drawn = True

        for metric in chosen:
            if (i, metric) not in contours:
                continue

            xx, yy, values = contours[i, metric]
            c, s = np.cos(state[2]), np.sin(state[2])

            contour = ax.contour(
                state[0]+c*xx-s*yy,
                state[1]+s*xx+c*yy,
                values,
                levels=[0],
                colors=[PALETTE.get(metric, 'black')],
                linestyles='--',
                linewidths=1.2*linewidth,
                zorder=3.5,
            )
            road_clip = Rectangle(
                (0, -2),
                1,
                8,
                transform=ax.get_yaxis_transform(),
            )
            contour.set_clip_path(road_clip)
            drawn_metrics.add(metric)

        _rectangle(ax, state, 4.68, 2.20, 'tab:blue')

        ax.set_xlim(state[0]-behind, state[0]+ahead)
        ax.set_ylim(-2.15, 6.15)
        ax.set_aspect('equal', adjustable='box')
        xmin, xmax = ax.get_xlim()
        x_ticks = 5 * np.arange(np.ceil(xmin / 5), np.floor(xmax / 5) + 1)

        ax.set_xticks(x_ticks)
        ax.spines['bottom'].set_position(('data', -2))
        ax.tick_params(axis='x', which='both', length=0, pad=4)
        ax.set_yticks([-2, 0, 2, 4, 6])
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_ylabel(r'$Y\ [\mathrm{m}]$')
        ax.set_xlabel(r'$X\ [\mathrm{m}]$')
        ax.set_title(
            rf'$t={time:.2f}\,\mathrm{{s}}$',
            loc='left', pad=5, fontsize=28,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(times):]:
        ax.set_visible(False)

    handles = []
    if predictions_drawn:
        handles.extend([
            Line2D(
                [], [], color='black',
                linestyle='--', label='MPC prediction',
            ),
            Line2D(
                [], [], color='tab:orange',
                linestyle='--', label='Human prediction',
            ),
        ])

    handles.extend(
        Line2D(
            [], [],
            color=PALETTE.get(m, 'black'),
            linestyle='--',
            linewidth=1.2*linewidth,
            label=f'{LABELS.get(m, m)}',
        )
        for m in chosen if m in drawn_metrics
    )

    if handles:
        columns = min(4, len(handles))
        legend_rows = int(np.ceil(len(handles)/columns))
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height+.4*legend_rows)
        fig.legend(
            handles=handles,
            loc='outside upper center',
            bbox_to_anchor=(0.5, 1.15),
            ncol=columns,
            frameon=False,
            columnspacing=1.2,
            handlelength=2.2,
            fontsize=26,
        )

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
    fig, ax = plt.subplots(figsize=(6, 3))
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
    ax.grid(True, which="major", color="gray", alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig


def plot_values_comparison(runs):
    fig, ax = plt.subplots(figsize=(7, 3.2))

    for run in runs:
        for k, metric in enumerate(run.metrics):
            values = np.where(
                run.hj["active"][:, 0, k],
                run.hj["V"][:, k],
                np.nan,
            )

            ax.plot(
                run.input_time,
                values,
                color=PALETTE.get(metric),
                label=LABELS.get(metric, metric),
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9)

    ax.set_xlabel(r"$t\ [\mathrm{s}]$")
    ax.set_ylabel(r"$V$")

    ax.grid(True, which="major", color="gray", alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    entries = dict(zip(labels, handles))

    fig.legend(
        entries.values(),
        entries.keys(),
        loc="outside upper center",
        ncol=min(4, len(entries)),
        frameon=False,
    )

    return fig


def _hj_window(run, stage=0, padding=1):
    available = np.flatnonzero(np.any(run.hj['active'][:, stage], axis=1))
    if not len(available):
        return (run.input_time[0], run.input_time[-1])
    return (max(run.input_time[0], run.input_time[available[0]]-padding),
            min(run.input_time[-1], run.input_time[available[-1]]+run.dt+padding))


def _mark_hj_switches(axes, run, active):
    indices = np.flatnonzero(active)
    if not len(indices):
        return
    switches = [(float(run.input_time[indices[0]]), 'HJ\nON')]
    if indices[-1] + 1 < len(run.input_time):
        switches.append((float(run.input_time[indices[-1]+1]), 'HJ\nOFF'))
    bottom = axes[-1]
    left, right = bottom.get_xlim()
    switches = [(t, label) for t, label in switches if left <= t <= right]
    if not switches:
        return
    ticks = bottom.get_xticks()
    labels = bottom.xaxis.get_major_formatter().format_ticks(ticks)
    clearance = .045 * (right-left)
    normal = [(t, label) for t, label in zip(ticks, labels)
              if left <= t <= right and all(abs(t-s) > clearance for s, _ in switches)]
    entries = sorted(normal+switches)
    bottom.set_xticks([t for t, _ in entries], [label for _, label in entries])
    for ax in axes:
        for tick, (position, _) in zip(ax.xaxis.get_major_ticks(), entries):
            tick.update_position(position)
            if any(np.isclose(position, t, rtol=0, atol=1e-9) for t, _ in switches):
                tick.set_pad(10)
                tick.tick1line.set_color('tab:green')
                tick.tick1line.set_markeredgecolor('tab:green')
                tick.tick1line.set_markeredgewidth(3.)
                tick.tick1line.set_markersize(9.)
                tick.label1.set_color('tab:green')
                tick.label1.set_multialignment('center')
                tick.gridline.set_visible(False)


def plot_hj_limits(run, metric=None, stage=0, softened=False, padding=1, command=None):
    if not 0 <= stage < run.diagnostic.stages:
        raise ValueError('Stage HJ non valido.')
    bounds = run.diagnostic.compute_limits(softened=softened)
    commands = ("steering_rate", "acceleration")
    selected = [0, 1] if command is None else [commands.index(command)]
    fig, panels = plt.subplots(len(selected), 1, figsize=(9, 3.5 * len(selected)), sharex=True, squeeze=False)
    axes = panels[:, 0]
    for ax, axis in zip(axes, selected):
        name = commands[axis]
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
        ax.grid(True, which="major", color="gray", alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        if np.any(active & ~feasible):
            ax.fill_between(run.input_time, run.diagnostic.u_lower[axis],
                            run.diagnostic.u_upper[axis], where=active & ~feasible,
                            color='tab:red', alpha=.12, label='Empty HJ interval')
        ax.set_ylabel(INPUT_LABELS[axis])
        if axis == 0:
            ax.set_yticks([-0.087, -0.04, 0, 0.04, 0.087])
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        else:
            ax.set_yticks([-7, -6, -5, -3.75, -2.50, -1.25, 0, 1.25, 2.5])
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_xlim(_hj_window(run, stage, padding))
    entries = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles))
    fig.legend(entries.values(), entries.keys(), loc="outside upper center", ncol=2, frameon=False,)
    _mark_hj_switches(axes, run, active)
    axes[-1].set_xlabel(r'$t\ [\mathrm{s}]$')
    return fig


def plot_command_planes(run, times=None, stage=0, softened=False, *,
                        zoom_to_constraints=False, xlim=None, ylim=None):
    """Use zoom_to_constraints only in the desired comparison notebook.

    Expanded views share the same limits across panels. xlim/ylim override them.
    The admissible polygon remains clipped to the physical control rectangle.
    """
    if not 0 <= stage < run.diagnostic.stages:
        raise ValueError('Stage HJ non valido.')
    if times is None:
        active = np.flatnonzero(np.any(run.hj['active'][:, stage], axis=1))
        ids = (active[np.linspace(0, len(active)-1, 3).round().astype(int)]
               if len(active) else np.array([0, len(run.input_time)//2, len(run.input_time)-1]))
    else:
        if not len(times) or any(not np.isfinite(t) or t < 0 or t > run.input_time[-1] for t in times):
            raise ValueError('Tempi del piano comandi fuori dalla simulazione.')
        ids = [int(np.argmin(abs(run.input_time-t))) for t in times]
    lo, hi = run.diagnostic.u_lower, run.diagnostic.u_upper
    view_lo = lo - np.array([.005, .25])
    view_hi = hi + np.array([.005, .25])
    if zoom_to_constraints:
        center, scale = (lo+hi)/2, (hi-lo)/2
        extent_points = [np.array([-1., -1.]), np.array([1., 1.])]
        for i in ids:
            for k in range(len(run.metrics)):
                if not run.hj['active'][i, stage, k]:
                    continue
                M = run.hj['M'][i, stage, k]
                b = run.hj['b'][i, stage, k] + (run.hj['slack'][i, stage, k] if softened else 0)
                normal = M*scale
                norm = np.linalg.norm(normal)
                if not np.isfinite(norm) or norm == 0:
                    continue
                # Closest boundary point in units of the physical half-ranges.
                unit = normal/norm
                closest = -((M @ center + b)/norm)*unit
                tangent = np.array([-unit[1], unit[0]])
                extent_points.extend([closest-tangent, closest+tangent])
        extent_points = np.asarray(extent_points)
        zlo, zhi = extent_points.min(axis=0), extent_points.max(axis=0)
        margin = .08*(zhi-zlo)
        view_lo, view_hi = center+scale*(zlo-margin), center+scale*(zhi+margin)
    for axis, limit in enumerate((xlim, ylim)):
        if limit is not None:
            limit = np.asarray(limit, dtype=float)
            if limit.shape != (2,) or not np.isfinite(limit).all() or limit[0] >= limit[1]:
                raise ValueError('xlim e ylim devono essere coppie finite crescenti.')
            view_lo[axis], view_hi[axis] = limit
    fig, axes = plt.subplots(1, len(ids), figsize=(4*len(ids), 4), squeeze=False)
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
                xx = np.linspace(view_lo[0], view_hi[0], 100)
                ax.plot(xx, -(M[0]*xx+b)/M[1], color=PALETTE.get(metric), label=LABELS.get(metric, metric))
            elif abs(M[0]) > 1e-12:
                ax.axvline(-b/M[0], color=PALETTE.get(metric), label=LABELS.get(metric, metric))
        ax.scatter(*run.hj['u'][i, stage], color='tab:blue', zorder=5, s=25)
        ax.set_xlim(view_lo[0], view_hi[0])
        ax.set_ylim(view_lo[1], view_hi[1])
        if not zoom_to_constraints and xlim is None:
            ax.set_xticks([-0.087, 0, 0.087])
        if not zoom_to_constraints and ylim is None:
            ax.set_yticks([-7, -5, -3, 0, 2.5])
        ax.grid(True, which='major', color='gray', alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(INPUT_LABELS[0])
        ax.set_ylabel(INPUT_LABELS[1])
        ax.set_title(rf'$t={run.input_time[i]:.2f}\,\mathrm{{s}}$')
    handles = [Line2D([], [], color=PALETTE.get(m), label=LABELS.get(m, m)) for m in run.metrics]
    handles += [Line2D([], [], color='tab:red', linestyle='--', label='Physical bounds'),
                Line2D([], [], color='tab:blue', marker='o', linestyle='none', label='MPC command')]
    fig.legend(handles=handles, loc='outside upper center', ncol=min(4, len(handles)), frameon=False)
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
