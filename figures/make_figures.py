"""Figures of the study, 4 Nov 2026 (Fig. 7 with the two lung cohorts of plan 7j; supplementary figures numbered by first citation: S1 mask QC, S2 fields by class in five lung datasets, S3 acquisition decoding). notebook 10's figures redrawn to the journal's artwork rules, from the same verified result files.

Artwork rules (174 mm wide, Arial, EPS and 600-dpi TIFF): width 39, 84, 129 or 174 mm, height at most 234 mm;
Arial lettering 8-12 pt; lines at least 0.3 pt; RGB; vector EPS preferred, TIFF for halftone/combination (600 dpi);
no titles inside figures; panels a, b, c; meaning not carried by colour alone (accessibility; print may be greyscale).
Nothing is refitted: every plotted value is read from notebook 03-10 outputs whose MD5 matches their run record.

Usage: python -I figures/make_figures.py <project folder> <output folder>
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

PROJECT, OUTDIR = Path(sys.argv[1]), Path(sys.argv[2])
RES = PROJECT / 'results'
IN = {'01': RES / '01_inventory', '02': RES / '02_tasks_features', '03': RES / '03_audit', '04': RES / '04_posthoc',
      '05': RES / '05_image_check', '06': RES / '06_image_posthoc', '07': RES / '07_screen_validation',
      '08': RES / '08_transfer', '09': RES / '09_transfer_posthoc', '10': RES / '10_figures_tables',
      '11': RES / '11_external_lung', '12': RES / '12_review_checks'}
for sub in ('figures_eps', 'figures_tiff', 'preview_png', 'esm_png'):
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(folder):
    man = json.loads((folder / 'run_manifest.json').read_text())
    bad = [f['file'] for f in man['outputs'] if md5(folder / f['file']) != f['md5']]
    assert not bad, f'changed since the run: {folder} {bad}'
    return man


man = {nb: verify(IN[nb]) for nb in ('02', '03', '10')}
for nb in ('05', '06', '07', '08', '09', '11'):
    man[nb] = verify(IN[nb] / 'real')
    assert man[nb]['in_colab'] and man[nb]['labels'] == 'real'
man['07d'] = verify(IN['07'] / 'decoding')
assert man['10']['in_colab'], 'notebook 10 outputs must be the Colab run'

audit = pd.read_csv(IN['03'] / 'audit_primary.csv')
tasks_final = pd.read_csv(IN['02'] / 'tasks_final.csv')
groups_fg = pd.read_csv(IN['03'] / 'audit_feature_groups.csv')
sites = pd.read_csv(IN['03'] / 'audit_sites.csv')
probe05 = pd.read_csv(IN['05'] / 'real' / 'image_probe.csv')
sil06 = pd.read_csv(IN['06'] / 'real' / 'silhouette_probe.csv')
sv = pd.read_csv(IN['07'] / 'real' / 'screen_validation_tasks.csv')
decoding = pd.read_csv(IN['07'] / 'decoding' / 'acquisition_decoding.csv')
transfer = pd.read_csv(IN['08'] / 'real' / 'transfer.csv')
sil09 = pd.read_csv(IN['09'] / 'real' / 'silhouette_transfer.csv')
hb09 = pd.read_csv(IN['09'] / 'real' / 'header_body_transfer.csv')
fields09 = pd.read_csv(IN['09'] / 'real' / 'field_differences_numeric.csv')
transfer11 = pd.read_csv(IN['11'] / 'real' / 'transfer.csv')
fields11 = pd.read_csv(IN['11'] / 'real' / 'field_differences_numeric.csv')
_C = ['external_auc', 'external_ci_low', 'external_ci_high', 'target_internal_auc']
for _pair in ('lung: TCGA -> NSCLC Radiogenomics', 'lung: TCGA -> CPTAC'):     # notebook 11 reproduces notebook 08
    _a = transfer[transfer.pair == _pair].set_index(['encoder', 'input']).sort_index()[_C]
    _b = transfer11[transfer11.pair == _pair].set_index(['encoder', 'input']).sort_index()[_C]
    assert np.allclose(_a.to_numpy(), _b.to_numpy(), atol=1e-9), _pair
robust = pd.read_csv(IN['10'] / 'robustness_screen_validation.csv')

# ---- style ------------------------------------------------------------------------------------------------------
MM = 1 / 25.4
FULL = 174 * MM
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#3d3c3a', '#6f6e69', '#d9d8d0', '#9a9990'
BLUE, ORANGE, AQUA = '#2a78d6', '#d9541e', '#12936a'          # contrast >= 3:1 on white for marks; text is ink
INPUT_COLOR = {'image': BLUE, 'background': ORANGE, 'silhouette': AQUA}
INPUT_MARK = {'image': 'o', 'background': 's', 'silhouette': '^'}
INPUT_NAME = {'image': 'full image', 'background': 'background only', 'silhouette': 'silhouette'}
ENCODERS = ('ctfm', 'merlin')
ENC = {'ctfm': 'CT-FM', 'merlin': 'Merlin'}
RAMP = ['#e3eefc', '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
        '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('blue_seq', RAMP)
FS = 8           # every piece of lettering is 8-9 pt at final size
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': FS, 'axes.titlesize': 9,
    'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 0,
    'text.color': INK, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'ps.fonttype': 42, 'pdf.fonttype': 42,
    'hatch.linewidth': 0.6, 'lines.linewidth': 1.0})

SHORT = {'renal subtype (KIRC vs KIRP)': 'TCGA renal subtype (KIRC vs KIRP)',
         'lung histology (LUAD vs LUSC)': 'TCGA lung histology (LUAD vs LUSC)'}


def short(t):
    return SHORT.get(t, t).replace(' | ', ': ')


def axis_style(ax, grid='x'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def chance(ax, orient='v', flag_line=True):
    line = ax.axvline if orient == 'v' else ax.axhline
    line(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if flag_line:
        line(0.6, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)


def ci_point(ax, x, lo, hi, y, color, filled=True, marker='o', size=22, lw=1.1):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='butt', zorder=2)
    ax.scatter([x], [y], s=size, marker=marker, facecolor=color if filled else 'white', edgecolor=color,
               linewidth=1.0, zorder=3)


def mk(marker, color, filled=True, ms=5, label=''):
    return Line2D([], [], marker=marker, ls='', mfc=color if filled else 'white', mec=color, mew=1.0, ms=ms,
                  label=label)


SAVED = {}


def save(fig, name, esm=False, check_height=True):
    w, h = fig.get_size_inches()
    assert abs(w - FULL) < 1e-6 and (h * 25.4 <= 234.0 or not check_height), (name, w * 25.4, h * 25.4)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())      # inches; nothing may fall outside the page
    assert tb.x0 >= -0.01 and tb.y0 >= -0.01 and tb.x1 <= w + 0.01 and tb.y1 <= h + 0.01, \
        (name, [round(v * 25.4, 1) for v in (tb.x0, tb.y0, tb.x1, tb.y1)], (round(w * 25.4), round(h * 25.4)))
    if esm:
        fig.savefig(OUTDIR / 'esm_png' / f'{name}.png', dpi=300)
    else:
        fig.savefig(OUTDIR / 'figures_eps' / f'{name}.eps')
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=600)
        Image.open(buf).convert('RGB').save(OUTDIR / 'figures_tiff' / f'{name}.tif', compression='tiff_lzw',
                                            dpi=(600, 600))       # RGB, 8 bits per channel (colour art rule)
        fig.savefig(OUTDIR / 'preview_png' / f'{name}.png', dpi=200)
    SAVED[name] = (round(w * 25.4, 1), round(h * 25.4, 1))
    plt.close(fig)
    print(f'saved {name}: {w * 25.4:.0f} x {h * 25.4:.0f} mm')


# ---- Fig. 1: workflow --------------------------------------------------------------------------------------------
a = audit.merge(tasks_final[['task_id', 'modality']], on='task_id', how='left').sort_values('auc').reset_index(drop=True)
assert a.modality.notna().all()
n_tasks, n_flag, n_ct = len(a), int(a.flagged.sum()), sv.task_id.nunique()
assert (n_tasks, n_flag, n_ct) == (26, 4, 19)
steps = [('TCIA public interface', 'series metadata only,\nno image download'),
         ('1  Metadata screen', f'{n_tasks} benchmark tasks,\n{n_flag} flagged'),
         ('2  Tissue-removed probes', f'background only and\nsilhouette, {n_ct} CT tasks'),
         ('3  External transfer', 'independent vs\nsimilarly assembled\ntarget'),
         ('4  Decoding', 'embeddings decode\nscanner settings')]
fig = plt.figure(figsize=(FULL, 30 * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1); ax.axis('off')
for i, (head, body) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((i + 0.04, 0.06), 0.84, 0.88, boxstyle='round,pad=0,rounding_size=0.05',
                                facecolor='#eef4fd' if i else '#f1f0ec', edgecolor=AXIS, lw=0.6))
    ax.text(i + 0.46, 0.76, head, ha='center', va='center', fontsize=FS, fontweight='bold', color=INK)
    ax.text(i + 0.46, 0.38, body, ha='center', va='center', fontsize=FS, color=INK, linespacing=1.25)
    if i < len(steps) - 1:
        ax.annotate('', xy=(i + 1.045, 0.5), xytext=(i + 0.885, 0.5),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8, mutation_scale=7))
save(fig, 'Fig1')

# ---- Fig. 3 (cited second in Results): field groups (risk map) -----------------------------------------------------------------------------
GROUP_ORDER = ['scanner (vendor, model)', 'software version', 'image count', 'slice thickness and kernel', 'body part',
               'description words']
GROUP_LABEL = ['Scanner\n(vendor,\nmodel)', 'Software\nversion', 'Image\ncount', 'Slice\nthickness\nand kernel',
               'Body\npart', 'Description\nwords']
assert sorted(groups_fg.group.unique()) == sorted(GROUP_ORDER)
M = groups_fg.pivot(index='task_id', columns='group', values='alone_auc').loc[a.task_id, GROUP_ORDER]
fig = plt.figure(figsize=(FULL, 150 * MM))
ax = fig.add_axes([0.335, 0.025, 0.54, 0.83])
cax = fig.add_axes([0.895, 0.25, 0.018, 0.45])
vals = M.to_numpy(dtype=float)
im = ax.imshow(np.clip(vals, 0.5, 0.9), cmap=SEQ, vmin=0.5, vmax=0.9, aspect='auto', origin='lower')
for (i, j), v in np.ndenumerate(vals):
    if np.isnan(v):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='white', edgecolor=MUTED, hatch='////', lw=0))
    else:
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FS, color='white' if v >= 0.72 else INK)
ax.set_xticks(range(len(GROUP_ORDER)), GROUP_LABEL)
ax.xaxis.tick_top()
ax.set_yticks(range(len(a)), [short(t) for t in a.task_id])
for lab, r in zip(ax.get_yticklabels(), a.itertuples()):
    lab.set_fontweight('bold' if r.flagged else 'normal')
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(GROUP_ORDER)), minor=True)
ax.set_yticks(np.arange(-0.5, len(a)), minor=True)
ax.grid(which='minor', color='white', lw=1.2)
ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, cax=cax, ticks=[0.5, 0.6, 0.7, 0.8, 0.9])
cb.set_label('AUC of the field group alone', fontsize=FS)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=FS, length=2)
save(fig, 'Fig3')

# ---- Fig. 2 (cited first in Results): metadata screen, forest plot ------------------------------------------------------------------------
fig = plt.figure(figsize=(FULL, 140 * MM))
ax = fig.add_axes([0.37, 0.085, 0.6, 0.905])
for i, r in a.iterrows():
    c = ORANGE if r.flagged else BLUE
    ci_point(ax, r.auc, r.ci_low, r.ci_high, i, c, filled=bool(r.flagged), marker='o' if r.modality == 'CT' else 's',
             size=22)
chance(ax)
ax.set_yticks(range(len(a)), [short(t) for t in a.task_id])
for lab, r in zip(ax.get_yticklabels(), a.itertuples()):
    lab.set_fontweight('bold' if r.flagged else 'normal')
ax.set_ylim(-0.7, len(a) - 0.3)
ax.set_xlim(0.2, 1.0)
ax.set_xlabel('Label predicted from acquisition metadata only: AUC (95% CI)')
axis_style(ax)
ax.legend(handles=[mk('o', ORANGE, True, label='flagged (lower CI ≥ 0.60 and q < 0.05)'),
                   mk('o', BLUE, False, label='not flagged'),
                   mk('o', INK2, False, label='CT task'), mk('s', INK2, False, label='MRI task')],
          loc='lower right', handletextpad=0.3, borderaxespad=0.2)
save(fig, 'Fig2')

# ---- Fig. 4: site or protocol ------------------------------------------------------------------------------------
s = sites.sort_values('overall_auc_same_patients').reset_index(drop=True)
fig = plt.figure(figsize=(FULL, 128 * MM))
ax = fig.add_axes([0.37, 0.2, 0.6, 0.79])
offs = {'overall': 0.27, 'site only': 0.09, 'sites held out': -0.09, 'within site': -0.27}
for i, r in s.iterrows():
    ax.scatter([r.overall_auc_same_patients], [i + offs['overall']], s=20, marker='D', color=INK, zorder=3)
    ci_point(ax, r.site_only_auc, r.site_only_ci_low, r.site_only_ci_high, i + offs['site only'], BLUE, marker='o',
             size=18, lw=1)
    ci_point(ax, r.site_heldout_auc, r.site_heldout_ci_low, r.site_heldout_ci_high, i + offs['sites held out'], ORANGE,
             marker='s', size=16, lw=1)
    ci_point(ax, r.within_site_auc, r.within_site_ci_low, r.within_site_ci_high, i + offs['within site'], AQUA,
             marker='^', size=20, lw=1)
    if i:
        ax.axhline(i - 0.5, color=GRID, lw=0.5)
chance(ax, flag_line=False)
ax.set_yticks(range(len(s)), [short(t) for t in s.task_id])
ax.set_ylim(-0.6, len(s) - 0.4)
ax.set_xlim(0, 1)
ax.set_xlabel('AUC (95% CI)')
axis_style(ax, grid=None)
fig.legend(handles=[mk('D', INK, True, ms=4.5, label='metadata, all patients'),
                    mk('o', BLUE, True, label='contributing site alone'),
                    mk('s', ORANGE, True, ms=4.5, label='metadata, whole sites held out'),
                    mk('^', AQUA, True, label='metadata, pairs within one site')],
           loc='lower left', bbox_to_anchor=(0.37, 0.0), ncol=2, handletextpad=0.3, columnspacing=1.2)
save(fig, 'Fig4')

# ---- Fig. 5: diagnosis without the patient -----------------------------------------------------------------------
p5 = probe05.merge(sil06[['task_id', 'encoder', 'silhouette_auc', 'silhouette_ci_low', 'silhouette_ci_high',
                          'silhouette_flagged']], on=['task_id', 'encoder'], how='left', validate='one_to_one')
assert p5.silhouette_auc.notna().all()
order5 = a[a.flagged].sort_values('auc').task_id.tolist()
fig = plt.figure(figsize=(FULL, 82 * MM))
axes = [fig.add_axes([0.33, 0.27, 0.31, 0.66]), fig.add_axes([0.675, 0.27, 0.31, 0.66])]
for ax, enc in zip(axes, ENCODERS):
    d = p5[p5.encoder == enc].set_index('task_id').loc[order5]
    for i, (t, r) in enumerate(d.iterrows()):
        ax.scatter([r.metadata_auc_same_patients], [i + 0.3], s=20, marker='D', color=INK, zorder=3)
        for k, y in (('image', 0.1), ('background', -0.1), ('silhouette', -0.3)):
            ci_point(ax, r[f'{k}_auc'], r[f'{k}_ci_low'], r[f'{k}_ci_high'], i + y, INPUT_COLOR[k],
                     filled=bool(r[f'{k}_flagged']), marker=INPUT_MARK[k], size=20)
        if i:
            ax.axhline(i - 0.5, color=GRID, lw=0.5)
    chance(ax)
    ax.set_title(f'{"ab"[ENCODERS.index(enc)]}  {ENC[enc]}', fontsize=9)
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(-0.55, len(order5) - 0.45)
    ax.set_xlabel('AUC (95% CI)')
    axis_style(ax, grid=None)
axes[0].set_yticks(range(len(order5)), [short(t) for t in order5])
axes[1].set_yticks(range(len(order5)), [''] * len(order5))
fig.legend(handles=[mk('D', INK, True, ms=4.5, label='metadata only'),
                    mk('o', BLUE, True, label='full image'),
                    mk('s', ORANGE, True, ms=4.5, label='background only (tissue removed)'),
                    mk('^', AQUA, True, label='silhouette (body outline only)'),
                    mk('o', INK2, False, label='open marker: not flagged')],
           loc='lower left', bbox_to_anchor=(0.02, 0.0), ncol=3, handletextpad=0.3, columnspacing=1.2)
save(fig, 'Fig5')

# ---- Fig. 6: validation of the screen ----------------------------------------------------------------------------
FLAG_ORDER = a[a.flagged].sort_values('auc', ascending=False).task_id.tolist()      # 1 = highest metadata AUC
NUM6 = {t: str(i + 1) for i, t in enumerate(FLAG_ORDER)}
KEY6 = {'renal subtype (KIRC vs KIRP)': 'KIRC vs KIRP', 'lung histology (LUAD vs LUSC)': 'LUAD vs LUSC',
        'LIDC-IDRI | primary vs metastatic': 'LIDC-IDRI primary vs metastatic', 'CPTAC-PDA | stage': 'CPTAC-PDA stage'}
NUM_POS = {('ctfm', 'LIDC-IDRI | primary vs metastatic'): (-8, 7)}
fig = plt.figure(figsize=(FULL, 128 * MM))
axes = [fig.add_axes([0.075, 0.37, 0.4, 0.57]), fig.add_axes([0.575, 0.37, 0.4, 0.57])]
for ax, enc in zip(axes, ENCODERS):
    d = sv[sv.encoder == enc]
    for flag in (False, True):
        q = d[d.metadata_flagged.astype(bool) == flag]
        ax.scatter(q.metadata_auc, q.background_auc, s=26, facecolor=ORANGE if flag else 'white',
                   edgecolor=ORANGE if flag else BLUE, linewidth=1.0, zorder=3)
    bf = d[d.background_flagged.astype(bool)]
    ax.scatter(bf.metadata_auc, bf.background_auc, s=95, facecolor='none', edgecolor=INK, linewidth=0.9, zorder=4)
    for r in d[d.metadata_flagged.astype(bool)].itertuples():
        dx, dy = NUM_POS.get((enc, r.task_id), (8, 0))
        ax.annotate(NUM6[r.task_id], (r.metadata_auc, r.background_auc), xytext=(dx, dy), textcoords='offset points',
                    ha='center', va='center', fontsize=FS, fontweight='bold', color=INK)
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0.3, 0.9); ax.set_ylim(0.3, 0.9)
    ax.set_xlabel('Metadata-only AUC')
    ax.set_ylabel('Background-only image AUC')
    rr = robust[(robust['mode'] == 'real') & (robust.encoder == enc)].set_index('check')
    lines = [('All 19 tasks', 'all 19 tasks (notebook 07, recomputed)'),
             ('15 unflagged tasks (post hoc)', 'without the flagged tasks'),
             ('11 patient-disjoint groups (post hoc)', 'patient-disjoint task groups (group means)')]
    txt = '\n'.join(f'{lab}: ρ = {rr.loc[k, "spearman_rho"]:.2f}, P = {rr.loc[k, "perm_p_one_sided"]:.3f}'
                    for lab, k in lines)
    ax.text(0.0, -0.2, 'Spearman ρ, one-sided permutation P\n' + txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=FS, color=INK, linespacing=1.35)
    ax.set_title(f'{"ab"[ENCODERS.index(enc)]}  {ENC[enc]}{" (primary encoder)" if enc == "ctfm" else ""}')
    axis_style(ax, grid='both')
fig.text(0.5, 0.075, 'Flagged tasks: ' + ', '.join(f'{NUM6[t]} {KEY6[t]}' for t in FLAG_ORDER), ha='center',
         va='bottom', fontsize=FS)
fig.legend(handles=[mk('o', ORANGE, True, label='flagged by the metadata screen'),
                    mk('o', BLUE, False, label='not flagged'),
                    Line2D([], [], marker='o', ls='', mfc='none', mec=INK, ms=9, mew=0.9,
                           label='ring: background-only probe flagged')],
           loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, handletextpad=0.3)
save(fig, 'Fig6')

# ---- Fig. 7: external transfer (five pairs and the CPTAC decomposition) ------------------------------------------
PANELS = [(transfer11, 'lung: TCGA -> NSCLC Radiogenomics', 'a  TCGA lung → NSCLC Radiogenomics (independent)'),
          (transfer11, 'lung: TCGA -> NSCLC-Radiomics', 'b  TCGA lung → NSCLC-Radiomics (independent, added)'),
          (transfer11, 'lung: TCGA -> Lung-PET-CT-Dx', 'c  TCGA lung → Lung-PET-CT-Dx (independent, added)'),
          (transfer11, 'lung: TCGA -> CPTAC', 'd  TCGA lung → CPTAC lung (assembled like TCGA)'),
          (transfer, 'kidney: TCGA -> C4KC-KiTS', 'e  TCGA kidney → C4KC-KiTS')]
ROWS7 = [('ctfm', 'image'), ('ctfm', 'background'), ('merlin', 'image'), ('merlin', 'background')]
fig = plt.figure(figsize=(FULL, 205 * MM))
xs, ys, pw, ph = (0.215, 0.705), (0.715, 0.445, 0.175), 0.27, 0.215
axes = [fig.add_axes([xs[i % 2], ys[i // 2], pw, ph]) for i in range(6)]
for ax, (tab, pair, title) in zip(axes[:5], PANELS):
    d = tab[tab.pair == pair].set_index(['encoder', 'input'])
    for i, key in enumerate(ROWS7[::-1]):
        r = d.loc[key]
        ci_point(ax, r.source_internal_auc, r.source_internal_ci_low, r.source_internal_ci_high, i + 0.24, MUTED,
                 marker='D', size=16, lw=1)
        ci_point(ax, r.external_auc, r.external_ci_low, r.external_ci_high, i, INPUT_COLOR[key[1]],
                 filled=r.reading != 'does not transfer', marker=INPUT_MARK[key[1]], size=22)
        ci_point(ax, r.target_internal_auc, r.target_internal_ci_low, r.target_internal_ci_high, i - 0.24, INK,
                 marker='v', filled=False, size=18, lw=1)
        if i:
            ax.axhline(i - 0.5, color=GRID, lw=0.5)
    chance(ax, flag_line=False)
    ax.set_yticks(range(4), [f'{ENC[e]}, {INPUT_NAME[k]}' for e, k in ROWS7[::-1]])
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.55, 3.55)
    axis_style(ax, grid=None)
ax = axes[5]
rows_d = []
for r in sil09[sil09.target == 'CPTAC lung'].itertuples():
    e = transfer[(transfer.pair == 'lung: TCGA -> CPTAC') & (transfer.encoder == r.encoder)
                 & (transfer.input == 'background')].iloc[0]
    rows_d.append((f'{ENC[r.encoder]}, background only', e.external_auc, e.external_ci_low, e.external_ci_high, ORANGE,
                   'background', e.reading != 'does not transfer'))
for r in sil09[sil09.target == 'CPTAC lung'].itertuples():
    rows_d.append((f'{ENC[r.encoder]}, silhouette', r.silhouette_external_auc, r.silhouette_ci_low, r.silhouette_ci_high,
                   AQUA, 'silhouette', bool(r.silhouette_transfers)))
h = hb09[hb09.target == 'CPTAC lung'].iloc[0]
rows_d.append(('DICOM header fields', h.header_external_auc, h.header_ci_low, h.header_ci_high, INK2, 'header',
               bool(h.header_transfers)))
rows_d.append(('body size', h.body_size_external_auc, h.body_size_ci_low, h.body_size_ci_high, INK2, 'body',
               bool(h.body_size_transfers)))
for i, (lab, v, lo, hi, c, kind, filled) in enumerate(rows_d[::-1]):
    ci_point(ax, v, lo, hi, i, c, filled=filled, marker=INPUT_MARK.get(kind, 'P'), size=22)
chance(ax, flag_line=False)
ax.set_yticks(range(len(rows_d)), [r[0] for r in rows_d[::-1]])
ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.6, len(rows_d) - 0.4)
axis_style(ax, grid=None)
titles = [t_ for *_, t_ in PANELS] + ['f  TCGA lung → CPTAC lung, decomposed (post hoc)']
for ax, t_, x in zip(axes, titles, (0.01, 0.5) * 3):
    fig.text(x, ax.get_position().y1 + 0.008, t_, ha='left', va='bottom', fontsize=8, fontweight='bold')
for ax in axes[4:]:
    ax.set_xlabel('AUC (95% CI)')
fig.legend(handles=[mk('D', MUTED, True, ms=4.5, label='source, internal (cross-validated)'),
                    mk('o', BLUE, True, label='external: full image'),
                    mk('s', ORANGE, True, ms=4.5, label='external: background only'),
                    mk('^', AQUA, True, label='external: silhouette'),
                    mk('P', INK2, True, label='external: header fields or body size'),
                    mk('v', INK, False, label='target, internal (cross-validated)'),
                    mk('o', INK2, False, label='open marker: does not transfer')],
           loc='lower left', bbox_to_anchor=(0.01, 0.0), ncol=3, handletextpad=0.3, columnspacing=1.0)
save(fig, 'Fig7')

# ---- the supplement: Figs. S1 and S2 --------------------------------------------------------------------------
TARGETS_C = decoding.target.unique().tolist()
cols = [(e, k) for e in ENCODERS for k in ('image', 'background', 'silhouette')]
D = decoding.set_index(['target', 'encoder', 'input'])
vals = np.array([[D.loc[(t, e, k), 'pooled_auc'] for e, k in cols] for t in TARGETS_C], dtype=float)
H1 = 9 * len(TARGETS_C) + 26
fig = plt.figure(figsize=(FULL, H1 * MM))
ax = fig.add_axes([0.3, 4 / H1, 0.54, 1 - 4 / H1 - 21 / H1])
cax = fig.add_axes([0.86, 0.12, 0.02, 0.5])
im = ax.imshow(np.clip(vals, 0.5, 1.0), cmap=SEQ, vmin=0.5, vmax=1.0, aspect='auto')
for (i, j), v in np.ndenumerate(vals):
    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FS, color='white' if v >= 0.8 else INK)
ax.set_xticks(range(len(cols)), [ENC[e] + '\n' + INPUT_NAME[k].replace(' ', '\n') for e, k in cols])
TARGET_LABEL = {'manufacturer': 'Manufacturer', 'slice spacing (<2 / 2-<4 / >=4 mm)': 'Slice spacing (3 bins)',
                'reconstruction diameter (median split)': 'Reconstruction diameter', 'kVp (120 vs other)': 'kVp (120 vs other)',
                'kernel': 'Kernel', 'contrast agent given (clinical, reported separately)': 'Contrast agent (clinical)'}
assert set(TARGETS_C) == set(TARGET_LABEL)
ax.set_yticks(range(len(TARGETS_C)), [TARGET_LABEL[t] for t in TARGETS_C])
ax.xaxis.tick_top(); ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(TARGETS_C)), minor=True)
ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, cax=cax, ticks=[0.5, 0.75, 1.0])
cb.set_label('Decoding AUC, pooled', fontsize=FS)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=FS, length=2)
save(fig, 'FigS3', esm=True)

PURPLE, OCHRE = '#7a4cc2', '#9a6b00'
DS = [('lung histology (LUAD vs LUSC)', 'TCGA lung', BLUE, 'o'), ('CPTAC lung', 'CPTAC lung', ORANGE, 's'),
      ('NSCLC Radiogenomics', 'NSCLC Radiogenomics', AQUA, '^'), ('NSCLC-Radiomics', 'NSCLC-Radiomics', PURPLE, 'D'),
      ('Lung-PET-CT-Dx', 'Lung-PET-CT-Dx', OCHRE, 'P')]
FIELD_NAME = {'hdr_slice_thickness': 'slice thickness', 'hdr_z_spacing': 'slice spacing',
              'hdr_recon_diameter': 'reconstruction diameter', 'hdr_kvp': 'kVp', 'hdr_tube_current': 'tube current',
              'body_area_cm2': 'body cross-section area'}
fig = plt.figure(figsize=(FULL, 125 * MM))
ax = fig.add_axes([0.25, 0.2, 0.7, 0.77])
fl = list(FIELD_NAME)[::-1]
off = np.linspace(0.32, -0.32, len(DS))
for i, f in enumerate(fl):
    for k, (ds, lab, c, m) in enumerate(DS):
        r = fields11[(fields11.dataset == ds) & (fields11.field == f)].iloc[0]
        if np.isfinite(r.auc_squamous_higher):
            ci_point(ax, r.auc_squamous_higher, r.ci_low, r.ci_high, i + off[k], c, marker=m, size=18, lw=1)
        else:
            ax.text(0.62, i + off[k], 'not computable (one value in all scans)', fontsize=7, color=INK2, va='center')
    if i:
        ax.axhline(i - 0.5, color=GRID, lw=0.5)
chance(ax, flag_line=False)
ax.set_yticks(range(len(fl)), [FIELD_NAME[f] for f in fl])
ax.set_ylim(-0.55, len(fl) - 0.45); ax.set_xlim(0.15, 0.9)
ax.set_xlabel('AUC for squamous cell carcinoma having the higher value (95% CI)')
axis_style(ax, grid=None)
fig.legend(handles=[mk(m, c, True, label=lab) for _, lab, c, m in DS], loc='lower left', bbox_to_anchor=(0.12, 0.0),
           ncol=3, handletextpad=0.3)
save(fig, 'FigS2', esm=True)

import shutil                                        # Fig. S1: the mask quality-control pictures (review check R6)
shutil.copyfile(IN['12'] / 'r6_mask_qc.png', OUTDIR / 'esm_png' / 'FigS1.png')
# the supplement prints Fig. S1 in two parts (rows 1-4 and 5-8) so that the labels stay readable
from PIL import Image
_qc = Image.open(OUTDIR / 'esm_png' / 'FigS1.png')
_h = _qc.size[1] // 2
_qc.crop((0, 0, _qc.size[0], _h)).save(OUTDIR / 'esm_png' / 'FigS1a.png')
_qc.crop((0, _h, _qc.size[0], _qc.size[1])).save(OUTDIR / 'esm_png' / 'FigS1b.png')
SAVED['FigS1'] = 'copied from results/12_review_checks/r6_mask_qc.png'

json.dump({'sizes_mm': SAVED, 'matplotlib': matplotlib.__version__, 'inputs_verified': sorted(man)},
          open(OUTDIR / 'figures_record.json', 'w'), indent=1)
print('matplotlib', matplotlib.__version__)
