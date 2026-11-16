"""Frame Mixed con campo BRT sulla strada e human trasparente."""
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.patches import Polygon, Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from mpc_utils import plot_frames, _fixed_brt_slice


def _covered_road_xlim(state, coordinates, road_limits, view_limits):
    """Intervallo X coperto dalla griglia per ogni Y della strada."""
    c, s = np.cos(state[2]), np.sin(state[2])
    lower, upper = np.asarray(view_limits, dtype=float)-state[0]
    for y in road_limits:
        dy = y-state[1]
        for coefficient, offset, axis in (
            (c, s*dy, coordinates[0]),
            (-s, c*dy, coordinates[1]),
        ):
            if abs(coefficient) < 1e-12:
                if offset < axis[0]-1e-10 or offset > axis[-1]+1e-10:
                    raise ValueError('La griglia BRT non copre tutta la larghezza della strada.')
                continue
            ends = sorted(((axis[0]-offset)/coefficient,
                           (axis[-1]-offset)/coefficient))
            lower, upper = max(lower, ends[0]), min(upper, ends[1])
    if upper-lower <= 1e-9:
        raise ValueError('Nessun intervallo X interamente coperto dal BRT nella finestra richiesta.')
    return lower+state[0], upper+state[0]


def plot_mixed_background_frames(
    run, times=(5.52, 7., 9., 10.40), behind=10., ahead=25.,
    prediction=True, ncols=2, *, brt_dir=None, brt_smoothing=0.5,
    brt_grid=(241, 161), color_limits=None, cmap='RdBu_r',
    background_alpha=1., edge_fade=3., crop_to_brt=False, road_limits=(-2., 6.),
    colorbar_label=r'$V(-3, x_R)$', colorbar_ticks=None,
):
    """Mantiene lo stile di plot_frames e aggiunge una scala colori comune.

    color_limits=None: estremi comuni ai campi visibili nei frame selezionati.
    color_limits=(minimo, massimo): scala manuale, uguale in tutte le esportazioni.
    I frame con HJ spento e le aree esterne alla griglia rimangono senza campo.
    brt_smoothing=0 mostra il campo interpolato senza filtro gaussiano.
    edge_fade: metri interni alla griglia per dissolvere il colore verso il bianco;
    0 disattiva la dissolvenza. Valori BRT, contorno zero e colorbar non cambiano.
    crop_to_brt=True restringe X alla copertura completa della strada e ignora
    edge_fade. Non estende la finestra, non estrapola e non attiva HJ.
    """
    if tuple(run.metrics) != ('mixed',):
        raise ValueError('Questa funzione richiede una simulazione con la sola metrica mixed.')
    if not 0 <= background_alpha <= 1:
        raise ValueError('background_alpha deve essere tra 0 e 1.')
    if not np.isfinite(edge_fade) or edge_fade < 0:
        raise ValueError('edge_fade deve essere un numero non negativo.')
    if not isinstance(crop_to_brt, (bool, np.bool_)):
        raise ValueError('crop_to_brt deve essere True o False.')
    road_limits = np.asarray(road_limits, dtype=float)
    if (road_limits.shape != (2,) or not np.isfinite(road_limits).all()
            or road_limits[0] >= road_limits[1]):
        raise ValueError('road_limits deve essere una coppia finita crescente.')
    if color_limits is not None:
        color_limits = np.asarray(color_limits, dtype=float)
        if (color_limits.shape != (2,) or not np.isfinite(color_limits).all()
                or color_limits[0] >= color_limits[1]):
            raise ValueError('color_limits deve essere una coppia finita crescente.')

    # La funzione esistente conserva testi, legenda, code e contorno zero.
    fig = plot_frames(
        run, times, behind=behind, ahead=ahead, prediction=prediction,
        ncols=ncols, brt_metrics=['mixed'], brt_smoothing=brt_smoothing,
        brt_grid=brt_grid, brt_dir=brt_dir,
    )
    axes = list(fig.axes[:len(times)])
    ids = [int(np.argmin(abs(run.time-t))) for t in times]
    for ax, i in zip(axes, ids):
        human = np.array([
            float(run.sim['X_H_initial']) + float(run.sim['v_H'])*run.time[i],
            float(run.sim['Y_H_initial']),
        ])
        for patch in ax.patches:
            if (isinstance(patch, Polygon)
                    and np.allclose(patch.get_facecolor(), to_rgba('tab:blue'))):
                patch.set_edgecolor('white')
                patch.set_linewidth(2.)
            if (isinstance(patch, Polygon)
                    and np.allclose(patch.get_facecolor(), to_rgba('tab:orange'))):
                patch.set_facecolor('tab:orange')
                patch.set_edgecolor('tab:orange')
                patch.set_linewidth(2.6)
                patch.set_zorder(1.4)
                patch.set_path_effects([
                    pe.Stroke(linewidth=5.6, foreground='white'), pe.Normal(),
                ])
        for line in ax.lines:
            if np.allclose(to_rgba(line.get_color()), to_rgba('tab:orange')):
                width = max(line.get_linewidth(), 2.6)
                line.set_linewidth(width)
                line.set_zorder(1.2)
                line.set_path_effects([
                    pe.Stroke(linewidth=width+2.5, foreground='white'), pe.Normal(),
                ])
        # Baricentro nero con alone bianco, senza riempire la sagoma.
        marker, = ax.plot(
            *human, marker='o', markerfacecolor='none', markeredgecolor='black',
            markersize=7, markeredgewidth=1.7, linestyle='none', zorder=1.6,
        )
        marker.set_path_effects([
            pe.Stroke(linewidth=4.2, foreground='white'), pe.Normal(),
        ])
        center, = ax.plot(
            *human, marker='+', color='black', markersize=7,
            markeredgewidth=1.7, linestyle='none', zorder=1.7,
        )
        center.set_path_effects([
            pe.Stroke(linewidth=4.2, foreground='white'), pe.Normal(),
        ])

    for legend in fig.legends:
        for handle in legend.legend_handles:
            if (hasattr(handle, 'get_color')
                    and np.allclose(to_rgba(handle.get_color()), to_rgba('tab:orange'))):
                handle.set_path_effects([
                    pe.Stroke(linewidth=handle.get_linewidth()+2.5,
                              foreground='white'), pe.Normal(),
                ])

    active_ids = [i for i in dict.fromkeys(ids)
                  if i < len(run.input_time) and run.hj['active'][i, 0, 0]]
    if not active_ids:
        warnings.warn('Nessun frame a HJ attivo: sfondo BRT e barra colori non aggiunti.')
        return fig
    folder = Path(brt_dir) if brt_dir is not None else run.folder.parents[2] / 'brt'
    fields = {}
    with np.load(folder / 'mixed.npz', allow_pickle=False) as data:
        coords = [np.asarray(data[key], dtype=float) for key in
                  ('x_rel', 'y_rel', 'theta_rel', 'v_H', 'delta_E', 'v_E')]
        values = data['BRT']
        period = float(data['grid_hi'][2]-data['grid_lo'][2])
        nx, ny = map(int, brt_grid)
        xx, yy = np.meshgrid(np.linspace(coords[0][0], coords[0][-1], nx),
                             np.linspace(coords[1][0], coords[1][-1], ny))
        if edge_fade and not crop_to_brt:
            distance = np.minimum.reduce([
                xx-coords[0][0], coords[0][-1]-xx,
                yy-coords[1][0], coords[1][-1]-yy,
            ])
            fade = np.clip(distance/edge_fade, 0., 1.)
            fade = fade**2 * (3.-2.*fade)
        else:
            fade = np.ones_like(xx)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        for i in active_ids:
            sliced = _fixed_brt_slice(values, coords, run.hj['relative_state'][i, 2:6], period)
            if sliced is None:
                continue
            dense = RegularGridInterpolator(tuple(coords[:2]), sliced)(points).reshape(xx.shape)
            if brt_smoothing:
                dense = gaussian_filter(
                    dense,
                    sigma=(brt_smoothing*(ny-1)/(len(coords[1])-1),
                           brt_smoothing*(nx-1)/(len(coords[0])-1)),
                    mode='nearest',
                )
            state = run.sim['simX'][i]
            c, s = np.cos(state[2]), np.sin(state[2])
            fields[i] = (state[0]+c*xx-s*yy, state[1]+s*xx+c*yy, dense)

    if crop_to_brt:
        for ax, i in zip(axes, ids):
            if i in fields:
                ax.set_xlim(_covered_road_xlim(
                    run.sim['simX'][i], coords, road_limits, ax.get_xlim(),
                ))

    visible = []
    for ax, i in zip(axes, ids):
        if i not in fields:
            continue
        gx, gy, dense = fields[i]
        left, right = ax.get_xlim()
        mask = ((gx >= left) & (gx <= right)
                & (gy >= road_limits[0]) & (gy <= road_limits[1])
                & np.isfinite(dense))
        if mask.any():
            visible.append(dense[mask])
    if not visible:
        warnings.warn('Nessun valore BRT disponibile nella parte di strada visibile.')
        return fig
    if color_limits is None:
        vmin = min(float(v.min()) for v in visible)
        vmax = max(float(v.max()) for v in visible)
        if vmin == vmax:
            margin = max(abs(vmin)*.01, 1e-6)
            vmin, vmax = vmin-margin, vmax+margin
    else:
        vmin, vmax = color_limits
    norm = (TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
            if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax))

    image = None
    for ax, i in zip(axes, ids):
        if i not in fields:
            continue
        gx, gy, dense = fields[i]
        left, right = ax.get_xlim()
        # Clip esatto: nessun colore oltre Y=-2 e Y=6 o fuori dalla griglia.
        clip = Rectangle((left, road_limits[0]), right-left,
                         road_limits[1]-road_limits[0], transform=ax.transData)
        # Composizione sul bianco prima del rendering: evita artefatti della
        # trasparenza variabile tra triangoli Gouraud nell'esportazione PDF.
        colors = plt.get_cmap(cmap)(norm(dense))[..., :3]
        opacity = (background_alpha*fade)[..., None]
        colors = 1. + (colors-1.)*opacity
        image = ax.pcolormesh(
            gx, gy, colors, shading='gouraud', zorder=-5, rasterized=True,
        )
        image.set_clip_path(clip)
        ax.set_xlim(left, right)

    below = any(v.min() < vmin for v in visible)
    above = any(v.max() > vmax for v in visible)
    extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
    # Barra pienamente opaca: rappresenta i valori, non la dissolvenza ai bordi.
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(mappable, ax=axes, location='right', pad=.02,
                       fraction=.035, extend=extend)
    bar.set_label(colorbar_label, rotation=270, labelpad=20, fontsize=26)
    if colorbar_ticks is not None:
        bar.set_ticks(colorbar_ticks)
    elif vmin < 0 < vmax:
        bar.set_ticks(np.r_[np.linspace(vmin, 0, 4), np.linspace(0, vmax, 4)[1:]])
    return fig
