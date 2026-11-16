# Cella da eseguire nel notebook Mixed, dopo la configurazione iniziale.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.patches import Polygon, Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from IPython.display import display
from mpc_utils import load_run, plot_frames, _fixed_brt_slice, LABELS, PALETTE
from mixed_background_frames import _covered_road_xlim

TIME = globals().get('COMPARE_TIME', 8.0)
METRICS_TO_PLOT = globals().get('COMPARE_METRICS', None)  # None: tutte quelle salvate nella costruzione Mixed.
N_COLS = globals().get('COMPARE_NCOLS', 2)
BEHIND, AHEAD = globals().get('COMPARE_VIEW', (10.0, 25.0))
CROP_TO_BRT = globals().get('COMPARE_CROP', True)
EDGE_FADE = globals().get('COMPARE_FADE', 3.0)         # Ignorato se CROP_TO_BRT=True.
SMOOTHING = globals().get('COMPARE_SMOOTHING', 0.5)         # 0: nessun filtro gaussiano.
GRID = (241, 161)
COLOR_LIMITS = globals().get('COMPARE_COLOR_LIMITS', None)     # Oppure (minimo, massimo), comuni a tutti i pannelli.
COLORBAR_TICKS = globals().get('COMPARE_COLORBAR_TICKS', None)   # Oppure una lista di tick.

run = load_run('mixed')
BRT_DIR = Path(globals().get('COMPARE_BRT_DIR', run.folder.parents[2] / 'brt'))
if not run.input_time[0] <= TIME <= run.input_time[-1]:
    raise ValueError('TIME fuori dalla simulazione registrata.')
i = int(np.argmin(abs(run.input_time-TIME)))
time = float(run.input_time[i])
if not run.hj['active'][i, 0, 0]:
    raise ValueError(f'HJ non attivo a t={time:.2f} s: scegli un altro istante.')
state = run.sim['simX'][i]
relative = run.hj['relative_state'][i]
road = (-2., 6.)

with np.load(BRT_DIR / 'mixed.npz', allow_pickle=False) as data:
    metadata = json.loads(data['metadata_json'].item())
    parameters = metadata['metric']['parameters']
    reference_axes = [data[k].copy() for k in
                      ('x_rel', 'y_rel', 'theta_rel', 'v_H', 'delta_E', 'v_E')]
    reference_period = float(data['grid_hi'][2]-data['grid_lo'][2])
    reference_horizon = float(data['target_time'])
    reference_dynamics = metadata['dynamics']['parameters']

if parameters.get('normalization') != 'percentile of absolute V0':
    raise ValueError('Normalizzazione Mixed diversa da quella attesa: controlla i metadati.')
source_metrics = parameters['metrics']
scales = dict(parameters['scales'])
# Mixed e' gia' propagato dal campo terminale normalizzato: non dividerlo ancora.
scales['mixed'] = 1.0
available_metrics = list(source_metrics) + ['mixed']
metrics = list(source_metrics if METRICS_TO_PLOT is None else METRICS_TO_PLOT)
if (not metrics or len(set(metrics)) != len(metrics)
        or any(m not in available_metrics for m in metrics)):
    raise ValueError(f'Scegli metriche distinte tra: {available_metrics}')
if not isinstance(N_COLS, int) or N_COLS < 1:
    raise ValueError('N_COLS deve essere un intero positivo.')
if not np.isfinite([SMOOTHING, EDGE_FADE]).all() or min(SMOOTHING, EDGE_FADE) < 0:
    raise ValueError('SMOOTHING e EDGE_FADE devono essere non negativi.')

nx, ny = GRID
xx, yy = np.meshgrid(
    np.linspace(reference_axes[0][0], reference_axes[0][-1], nx),
    np.linspace(reference_axes[1][0], reference_axes[1][-1], ny),
)
points = np.column_stack((xx.ravel(), yy.ravel()))
c, s = np.cos(state[2]), np.sin(state[2])
gx, gy = state[0]+c*xx-s*yy, state[1]+s*xx+c*yy
view = (state[0]-BEHIND, state[0]+AHEAD)
if CROP_TO_BRT:
    view = _covered_road_xlim(state, reference_axes, road, view)
visible = ((gx >= view[0]) & (gx <= view[1]) & (gy >= road[0]) & (gy <= road[1]))
if not visible.any():
    raise ValueError('Nessun campo BRT nella finestra selezionata.')

fields, rows = {}, []
for metric in metrics:
    scale = float(scales[metric])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'{metric}: divisore di normalizzazione non valido.')
    with np.load(BRT_DIR / f'{metric}.npz', allow_pickle=False) as data:
        coords = [data[k].copy() for k in
                  ('x_rel', 'y_rel', 'theta_rel', 'v_H', 'delta_E', 'v_E')]
        if any(not np.array_equal(a, b) for a, b in zip(coords, reference_axes)):
            raise ValueError(f'{metric}: griglia diversa da quella usata da Mixed.')
        meta = json.loads(data['metadata_json'].item())
        if meta['dynamics']['parameters'] != reference_dynamics:
            raise ValueError(f'{metric}: dinamiche diverse da Mixed.')
        if not np.isclose(float(data['target_time']), reference_horizon):
            raise ValueError(f'{metric}: orizzonte BRT diverso da Mixed.')
        period = float(data['grid_hi'][2]-data['grid_lo'][2])
        if not np.isclose(period, reference_period):
            raise ValueError(f'{metric}: periodo angolare diverso da Mixed.')
        values = data['BRT']
        sliced = _fixed_brt_slice(values, coords, relative[2:6], period)
        del values
    if sliced is None or not np.isfinite(sliced).all():
        raise ValueError(f'{metric}: sezione BRT non disponibile allo stato Mixed.')
    sliced /= scale
    interpolator = RegularGridInterpolator(tuple(coords[:2]), sliced)
    # Valore al baricentro human: senza smussamento, per controllo quantitativo.
    point_value = float(interpolator(relative[:2].reshape(1, 2))[0])
    dense = interpolator(points).reshape(xx.shape)
    if SMOOTHING:
        dense = gaussian_filter(
            dense,
            sigma=(SMOOTHING*(ny-1)/(len(coords[1])-1),
                   SMOOTHING*(nx-1)/(len(coords[0])-1)),
            mode='nearest',
        )
    fields[metric] = dense
    rows.append({'metric': metric, 'scale_from_mixed': scale,
                 'normalized_BRT_at_human': point_value})

if COLOR_LIMITS is None:
    vmin = min(float(v[visible].min()) for v in fields.values())
    vmax = max(float(v[visible].max()) for v in fields.values())
    if vmin == vmax:
        margin = max(abs(vmin)*.01, 1e-6)
        vmin, vmax = vmin-margin, vmax+margin
else:
    vmin, vmax = map(float, COLOR_LIMITS)
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError('COLOR_LIMITS deve essere una coppia finita crescente.')
norm = (TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax))
fade = np.ones_like(xx)
if not CROP_TO_BRT and EDGE_FADE:
    distance = np.minimum.reduce([
        xx-reference_axes[0][0], reference_axes[0][-1]-xx,
        yy-reference_axes[1][0], reference_axes[1][-1]-yy,
    ])
    fade = np.clip(distance/EDGE_FADE, 0., 1.)
    fade = fade**2*(3.-2.*fade)

# Stesso identico frame Mixed in tutti i pannelli: nessuna simulazione rilanciata.
fig = plot_frames(
    run, times=(time,)*len(metrics), behind=BEHIND, ahead=AHEAD,
    ncols=N_COLS, brt_metrics=[],
)
axes = fig.axes[:len(metrics)]
human = np.array([float(run.sim['X_H_initial'])+float(run.sim['v_H'])*time,
                  float(run.sim['Y_H_initial'])])
for ax, metric in zip(axes, metrics):
    ax.set_xlim(view)
    clip = Rectangle((view[0], road[0]), view[1]-view[0], road[1]-road[0],
                     transform=ax.transData)
    colors = plt.get_cmap('RdBu_r')(norm(fields[metric]))[..., :3]
    colors = 1.+(colors-1.)*fade[..., None]
    image = ax.pcolormesh(gx, gy, colors, shading='gouraud', zorder=-5, rasterized=True)
    image.set_clip_path(clip)
    if fields[metric].min() < 0 < fields[metric].max():
        contour = ax.contour(
            gx, gy, fields[metric], levels=[0], colors=[PALETTE.get(metric, 'black')],
            linestyles='--', linewidths=1.2*plt.rcParams['lines.linewidth'], zorder=3.5,
        )
        contour.set_clip_path(clip)
    for patch in ax.patches:
        if not isinstance(patch, Polygon):
            continue
        if np.allclose(patch.get_facecolor(), to_rgba('tab:orange')):
            patch.set_edgecolor('white')
            patch.set_linewidth(2.)
            patch.set_zorder(1.4)
        elif np.allclose(patch.get_facecolor(), to_rgba('tab:blue')):
            patch.set_edgecolor('white')
            patch.set_linewidth(2.)
    for line in ax.lines:
        if np.allclose(to_rgba(line.get_color()), to_rgba('tab:orange')):
            line.set_path_effects([
                pe.Stroke(linewidth=line.get_linewidth()+2.5, foreground='white'), pe.Normal(),
            ])
            line.set_zorder(1.2)
    centroid, = ax.plot(*human, marker='+', color='black', markersize=8,
                        markeredgewidth=1.8, linestyle='none', zorder=1.7)
    centroid.set_path_effects([pe.Stroke(linewidth=4.2, foreground='white'), pe.Normal()])
    ax.set_title(f'{LABELS.get(metric, metric)} — '+rf'$t={time:.2f}\,\mathrm{{s}}$',
                 loc='left', fontsize=26, pad=5)
    ax.set_xlim(view)

below = any(v[visible].min() < vmin for v in fields.values())
above = any(v[visible].max() > vmax for v in fields.values())
extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
bar = fig.colorbar(ScalarMappable(norm=norm, cmap='RdBu_r'), ax=axes,
                   location='right', pad=.02, fraction=.035, extend=extend)
bar_label = (r'$\widetilde V(-3,x_R)$' if 'mixed' in metrics
             else r'$V_m(-3,x_R)\,/\,p_{95}$')
bar.set_label(bar_label, rotation=270, labelpad=24, fontsize=26)
bar.ax.tick_params(labelsize=20)
if COLORBAR_TICKS is not None:
    bar.set_ticks(COLORBAR_TICKS)
elif vmin < 0 < vmax:
    bar.set_ticks(np.r_[np.linspace(vmin, 0, 4), np.linspace(0, vmax, 4)[1:]])
print(f'Stato e predizioni della simulazione Mixed a t={time:.2f} s.')
print(f"Divisori salvati: percentile {parameters['percentile']:g} di |V0|.")
display(pd.DataFrame(rows))
show(f'mixed_components_t_{time:.2f}'.replace('.', 'p'), fig)
