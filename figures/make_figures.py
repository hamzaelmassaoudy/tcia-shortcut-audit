"""Figures of the study, 12 Nov 2026 (Fig. 1 with the names of the Methods subsections; Fig. 7 with the two lung cohorts of plan 7j; Fig. 8 for the experiments of plan 7o; supplementary figures numbered by first citation: S1 mask QC, S2 fields by class in five lung datasets, S3 acquisition decoding). notebook 10's figures redrawn to the journal's artwork rules, from the same verified result files.

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
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BAND, BAND_FLAG, TINT = '#f6f5f1', '#fdefe7', '#e9f1fc'      # row bands (plain, flagged) and the tint of small tags
BLUE, ORANGE, AQUA = '#2a78d6', '#d9541e', '#12936a'          # checked together for colour-vision deficiencies on white
INPUT_COLOR = {'image': BLUE, 'background': ORANGE, 'silhouette': AQUA}
INPUT_MARK = {'image': 'o', 'background': 's', 'silhouette': '^'}
INPUT_NAME = {'image': 'full image', 'background': 'background only', 'silhouette': 'silhouette'}
ENCODERS = ('ctfm', 'merlin')
ENC = {'ctfm': 'CT-FM', 'merlin': 'Merlin'}
RAMP = ['#e3eefc', '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
        '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('blue_seq', RAMP)
FS = 8           # every piece of lettering is 8-10 pt at final size
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


def axis_style(ax, grid='x', left=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(left)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def bands(ax, n, flagged=(), x0=0.0, x1=1.0):
    """Alternating row bands of a forest plot; x0 and x1 in axes fractions (may reach beyond the axes)."""
    for i in range(n):
        c = BAND_FLAG if i in flagged else (BAND if i % 2 == 0 else None)
        if c:
            ax.axhspan(i - 0.5, i + 0.5, xmin=x0, xmax=x1, facecolor=c, edgecolor='none', zorder=0, clip_on=False)


def chance(ax, orient='v', flag_line=True):
    line = ax.axvline if orient == 'v' else ax.axhline
    line(0.5, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    if flag_line:
        line(0.6, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)


LIGHT = {BLUE: '#bcd6f5', ORANGE: '#f4c7b4', AQUA: '#b0dccb', INK2: '#cfcec9', MUTED: '#dad9d4', INK: '#c4c4c4'}


def state_of(filled):
    """'solid' (full colour), 'light' (pale fill: weaker evidence) or 'open' (white); True and False mean solid and open."""
    return filled if isinstance(filled, str) else ('solid' if filled else 'open')


def level(ci_above, corrected):
    """Marker state of a transfer: solid if the CI lies above 0.5 and the corrected test is passed, light if only the first."""
    return 'solid' if (ci_above and corrected) else ('light' if ci_above else 'open')


def ci_point(ax, x, lo, hi, y, color, filled=True, marker='o', size=26, lw=1.3):
    st = state_of(filled)
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='round', zorder=2)
    ax.scatter([x], [y], s=size * (1.15 if st == 'solid' else 1.0), marker=marker,
               facecolor={'solid': color, 'light': LIGHT[color], 'open': 'white'}[st],
               edgecolor='white' if st == 'solid' else color, linewidth=0.6 if st == 'solid' else 1.2, zorder=3)


def mk(marker, color, filled=True, ms=5.2, label=''):
    st = state_of(filled)
    return Line2D([], [], marker=marker, ls='', mfc={'solid': color, 'light': LIGHT[color], 'open': 'white'}[st],
                  mec=color, mew=1.1, ms=ms, label=label)


def ref_handles(flag_line=True):
    h = [Line2D([], [], color=INK2, lw=0.8, ls=(0, (4, 2)), label='chance (0.5)')]
    if flag_line:
        h.append(Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 2)), label='flag threshold (0.60)'))
    return h


def dec3(v, d=3):
    """A number printed with d decimals, ties rounded to the even digit (as in the tables of the article)."""
    from decimal import Decimal, ROUND_HALF_EVEN
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_EVEN))


def panel_title(fig, ax, text, dy=0.008):
    p = ax.get_position()
    fig.text(p.x0, p.y1 + dy, text, ha='left', va='bottom', fontsize=9, fontweight='bold', linespacing=1.15)


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
                                            dpi=(600, 600))       # RGB, 8 bits per channel
        fig.savefig(OUTDIR / 'preview_png' / f'{name}.png', dpi=200)
    SAVED[name] = (round(w * 25.4, 1), round(h * 25.4, 1))
    plt.close(fig)
    print(f'saved {name}: {w * 25.4:.0f} x {h * 25.4:.0f} mm')


# ---- Fig. 1: workflow (each card carries the name of its Methods subsection, in the same order) ------------------
from PIL import Image as _PILImage

a = audit.merge(tasks_final[['task_id', 'modality']], on='task_id', how='left').sort_values('auc').reset_index(drop=True)
assert a.modality.notna().all()
n_tasks, n_flag, n_ct = len(a), int(a.flagged.sum()), sv.task_id.nunique()
n_mr = int((a.modality == 'MR').sum())
assert (n_tasks, n_flag, n_ct, n_mr) == (26, 4, 19, 7)
EXAMPLE = 'TCGA-LUAD__TCGA-17-Z013.png'      # one public scan (TCGA-LUAD, CC BY 3.0) as an example of the image inputs


def qc_panels(path):
    """The three image panels of a saved quality-control picture (960 x 336 px; panel titles above row 34)."""
    im = _PILImage.open(path).convert('RGB')
    assert im.size == (960, 336), (path, im.size)
    arr = np.asarray(im)[34:]
    used = (arr.min(axis=2) < 250).any(axis=0).astype(int)          # columns that hold image content
    edges = np.flatnonzero(np.diff(np.r_[0, used, 0]))
    runs = [(s_, e_) for s_, e_ in zip(edges[::2], edges[1::2]) if e_ - s_ > 200]
    assert len(runs) == 3, runs
    out = []
    for s_, e_ in runs:
        sub = arr[:, s_:e_]
        rows = np.flatnonzero((sub.min(axis=2) < 250).any(axis=1))
        out.append(sub[rows[0]:rows[-1] + 1])
    return out


TH2 = qc_panels(IN['06'] / 'cache' / 'qc' / EXAMPLE)                 # full image with mask outline, background, silhouette
TH4 = qc_panels(RES / '12_image_experiments' / 'cache' / 'qc' / EXAMPLE)   # lung mask, tumour mask, harmonized geometry
FLOW = [dict(kind='data', head='Data: Collections, Tasks and Labels',
             body=['Series metadata from the public interface of The Cancer', 'Imaging Archive (TCIA), without any image download.'],
             visual='stats', arrow=f'{n_tasks} tasks'),
        dict(kind='1', head='Step 1: Metadata Screen',
             body=['Logistic regression on acquisition metadata alone;', 'flag rule, site proxy, open tool and positive control.'],
             visual='dots', arrow=f'{n_flag} flagged and {n_ct - n_flag} unflagged CT tasks'),
        dict(kind='2', head='Step 2: Tissue-Removed Image Probes',
             body=['Two frozen CT foundation models with linear probes', f'on three inputs; compared with the screen ({n_ct} CT tasks).'],
             visual='thumbs2', arrow='probes of the flagged lung and kidney tasks'),
        dict(kind='3', head='Step 3: External Transfer',
             body=['Probes trained on TCGA, applied unchanged to', 'external datasets.'],
             visual='chips3', arrow='lung histology'),
        dict(kind='4', head='Step 4: Experiments on the Source of the Signal',
             body=['Lungs-only and tumour-only inputs, body habitus,', 'harmonized geometry and a 3D network from scratch.'],
             visual='thumbs4', arrow='from Step 2: embeddings', rail=False),
        dict(kind='5', head='Step 5: Acquisition Decoding',
             body=['Scanner settings decoded from the embeddings', 'without diagnostic labels (1249 scans, 13 collections).'],
             visual='chips5', arrow=None)]
SIDE = [dict(head='Targeted Literature Check', x0=15.0, x1=82.0,
             body=['How published studies on the flagged', 'tasks handled acquisition: 895 records', 'screened, 13 studies included.']),
        dict(head='Evaluation Metrics and Statistical Analysis', x0=86.0, x1=172.5,
             body=['In every step: AUC with 95% CI and permutation P.', 'Flagged: lower CI bound ≥ 0.60 and q < 0.05.',
                   'Transfers: external CI above 0.5 and corrected P < 0.05.'])]
W1, RAIL, CX0, CX1, VX0 = 174.0, 7.0, 15.0, 172.5, 109.0
PAD, HEAD, LINE, GAP, THUMB = 2.6, 4.7, 3.7, 6.2, 15.5
card_h = lambda c: 2 * PAD + max(HEAD + LINE * len(c['body']), (THUMB + 4.6) if c['visual'].startswith('thumbs') else 0)
heights = [card_h(c) for c in FLOW]
side_h = 2 * PAD + HEAD + LINE * 3
H1F = 1.0 + sum(heights) + GAP * (len(FLOW) - 1) + 4.0 + side_h + 1.0
fig = plt.figure(figsize=(FULL, H1F * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W1); ax.set_ylim(H1F, 0); ax.axis('off')
_r = fig.canvas.get_renderer()
PX = fig.dpi * MM                                                     # pixels per millimetre


def text_w(t):
    return t.get_window_extent(_r).width / PX


def card(x0, y, x1, h, accent, dashed=False):
    box = ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle='round,pad=0,rounding_size=1.4', facecolor='white',
                                      edgecolor=AXIS, lw=0.7, linestyle=(0, (4, 2)) if dashed else '-', zorder=2))
    strip = ax.add_patch(plt.Rectangle((x0, y), 1.3, h, facecolor=accent, edgecolor='none', zorder=3))
    strip.set_clip_path(box)
    return box


def chip_row(x, y, labels, face=TINT):
    for lab in labels:
        t = ax.text(x + 1.6, y + 2.3, lab, ha='left', va='center', fontsize=FS, color=INK, zorder=5)
        w = text_w(t) + 3.2
        ax.add_patch(FancyBboxPatch((x, y), w, 4.6, boxstyle='round,pad=0,rounding_size=1.2', facecolor=face,
                                    edgecolor='none', zorder=4))
        x += w + 1.6
    return x


boxes, texts, y, centers = [], [], 1.0, []
for c, h in zip(FLOW, heights):
    step = c['kind'] != 'data'
    boxes.append(card(CX0, y, CX1, h, BLUE if step else INK2))
    k = len(boxes) - 1
    texts.append((k, VX0, ax.text(CX0 + 4.2, y + PAD, c['head'], ha='left', va='top', fontsize=9.5, fontweight='bold',
                                  color=INK, zorder=5)))
    for j, line in enumerate(c['body']):
        texts.append((k, VX0, ax.text(CX0 + 4.2, y + PAD + HEAD + LINE * j, line, ha='left', va='top', fontsize=FS,
                                      color=INK2, zorder=5)))
    cy = y + PAD + 2.4
    centers.append(cy)
    ax.add_patch(plt.Circle((RAIL, cy), 3.1, facecolor=BLUE if step else INK2, edgecolor='white', lw=1.0, zorder=6))
    if step:
        ax.text(RAIL, cy + 0.05, c['kind'], ha='center', va='center', fontsize=9.5, fontweight='bold', color='white', zorder=7)
    else:
        for dy in (-1.1, 0.0, 1.1):
            ax.plot([RAIL - 1.5, RAIL + 1.5], [cy + dy, cy + dy], color='white', lw=1.1, solid_capstyle='round', zorder=7)
    vy = y + PAD                                                      # the visual zone of the card
    if c['visual'] == 'stats':
        for i, (num, lab) in enumerate((('33', 'collections'), ('63 468', 'series'), (str(n_tasks), 'tasks'),
                                        ('2775', 'patients'))):
            xc = VX0 + 7.5 + i * 15.6
            texts.append((k, CX1, ax.text(xc, vy + 0.6, num, ha='center', va='top', fontsize=10.5, fontweight='bold',
                                          color=INK, zorder=5)))
            texts.append((k, CX1, ax.text(xc, vy + 6.4, lab, ha='center', va='top', fontsize=FS, color=INK2, zorder=5)))
    elif c['visual'] == 'dots':
        for i in range(n_tasks):
            flagged = i < n_flag
            ax.add_patch(plt.Circle((VX0 + 2.0 + (i % 13) * 4.4, vy + 2.2 + (i // 13) * 4.0), 1.35,
                                    facecolor=ORANGE if flagged else 'white', edgecolor=ORANGE if flagged else BLUE,
                                    lw=0.9, zorder=5))
        texts.append((k, CX1, ax.text(VX0 + 0.6, vy + 9.2, f'{n_flag} of {n_tasks} tasks flagged (filled)', ha='left',
                                      va='top', fontsize=FS, color=INK2, zorder=5)))
    elif c['visual'].startswith('thumbs'):
        pics = TH2 if c['visual'] == 'thumbs2' else TH4
        names = (('full image', 'background only', 'silhouette') if c['visual'] == 'thumbs2'
                 else ('lung mask', 'tumour mask', 'harmonized'))
        for i, (pic, nm) in enumerate(zip(pics, names)):
            x = VX0 + 2.0 + i * (THUMB + 5.0)
            frame = ax.add_patch(FancyBboxPatch((x, vy), THUMB, THUMB, boxstyle='round,pad=0,rounding_size=0.9',
                                                facecolor='black', edgecolor=AXIS, lw=0.6, zorder=4))
            im = ax.imshow(pic, extent=(x, x + THUMB, vy + THUMB, vy), aspect='auto', zorder=5, interpolation='lanczos')
            im.set_clip_path(frame)
            texts.append((k, CX1, ax.text(x + THUMB / 2, vy + THUMB + 1.0, nm, ha='center', va='top', fontsize=FS,
                                          color=INK2, zorder=5)))
    elif c['visual'] == 'chips3':
        chip_row(VX0 + 0.6, vy + 0.6, ['3 independent lung cohorts', '1 kidney cohort'])
        chip_row(VX0 + 0.6, vy + 6.4, ['1 lung benchmark assembled like TCGA'], face=BAND_FLAG)
    elif c['visual'] == 'chips5':
        chip_row(VX0 + 0.6, vy + 0.6, ['manufacturer', 'kernel', 'kVp'])
        chip_row(VX0 + 0.6, vy + 6.4, ['slice spacing', 'reconstruction diameter'])
    y += h
    if c['arrow']:
        if c.get('rail', True):
            ax.annotate('', xy=(RAIL, y + GAP + PAD - 1.4), xytext=(RAIL, cy + 3.4),
                        arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0), zorder=1)
        ax.text(CX0 + 4.2, y + GAP / 2, c['arrow'], ha='left', va='center', fontsize=FS, color=INK2, style='italic')
        y += GAP
# Step 5 does not follow from Step 4: it takes the embeddings of Step 2 (side arrow from badge 2 to badge 5)
ax.plot([RAIL - 3.1, 1.8, 1.8], [centers[2], centers[2], centers[5]], color=MUTED, lw=1.0, solid_capstyle='round', zorder=1)
ax.annotate('', xy=(RAIL - 3.2, centers[5]), xytext=(1.8, centers[5]),
            arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0), zorder=1)
y += 4.0
for s in SIDE:
    boxes.append(card(s['x0'], y, s['x1'], side_h, MUTED, dashed=True))
    k = len(boxes) - 1
    texts.append((k, s['x1'], ax.text(s['x0'] + 4.2, y + PAD, s['head'], ha='left', va='top', fontsize=9.5,
                                      fontweight='bold', color=INK, zorder=5)))
    for j, line in enumerate(s['body']):
        texts.append((k, s['x1'], ax.text(s['x0'] + 4.2, y + PAD + HEAD + LINE * j, line, ha='left', va='top',
                                          fontsize=FS, color=INK2, zorder=5)))
ax.set_xlim(0, W1); ax.set_ylim(H1F, 0)                              # imshow must not change the page
fig.canvas.draw()                      # every text must lie inside its card (1 mm margin) and left of its limit
_r = fig.canvas.get_renderer()
for k, right_mm, t in texts:
    tb, bb = t.get_window_extent(_r), boxes[k].get_window_extent(_r)
    m = fig.dpi * MM
    assert tb.x0 >= bb.x0 + m and tb.x1 <= min(bb.x1 - m, right_mm * m) and tb.y0 >= bb.y0 + m * 0.8 and tb.y1 <= bb.y1 - m * 0.8, \
        f'Fig. 1: text {t.get_text()!r} does not fit inside its card'
save(fig, 'Fig1')

# ---- Fig. 2 (cited first in Results): metadata screen, forest plot with the values at the right -----------------
n = len(a)
fig = plt.figure(figsize=(FULL, 140 * MM))
AX2 = [0.335, 0.115, 0.385, 0.835]
ax = fig.add_axes(AX2)
XL, XR = (0.004 - AX2[0]) / AX2[2], (0.996 - AX2[0]) / AX2[2]         # bands run across the whole page width
flag_rows = set(a.index[a.flagged.astype(bool)])
bands(ax, n, flag_rows, XL, XR)
tr = ax.get_yaxis_transform()
for i, r in a.iterrows():
    c = ORANGE if r.flagged else BLUE
    ci_point(ax, r.auc, r.ci_low, r.ci_high, i, c, filled=bool(r.flagged), marker='o' if r.modality == 'CT' else 's')
    w = 'bold' if r.flagged else 'normal'
    ax.text(1.035, i, f'{dec3(r.auc)} ({dec3(r.ci_low)}–{dec3(r.ci_high)})', transform=tr, ha='left', va='center',
            fontsize=FS, fontweight=w)
    ax.text(XR - 0.03, i, dec3(r.perm_q_bh, 4 if r.perm_q_bh < 0.01 else 3), transform=tr, ha='right',
            va='center', fontsize=FS, fontweight=w)
chance(ax)
ax.set_yticks(range(n), [short(t) for t in a.task_id])
for lab, r in zip(ax.get_yticklabels(), a.itertuples()):
    lab.set_fontweight('bold' if r.flagged else 'normal')
for x, t, ha in ((-0.025, 'Task', 'right'), (0.5, 'Metadata-only AUC', 'center'), (1.035, 'AUC (95% CI)', 'left'),
                 (XR - 0.03, 'q', 'right')):
    ax.text(x, n - 0.38, t, transform=tr, ha=ha, va='bottom', fontsize=FS, fontweight='bold', color=INK)
ax.plot([XL, XR], [n - 0.5, n - 0.5], transform=tr, color=INK2, lw=0.7, clip_on=False)
ax.set_ylim(-0.5, n - 0.5)
ax.set_xlim(0.2, 1.0)
ax.set_xlabel('AUC (95% CI)')
axis_style(ax)
fig.legend(handles=[mk('o', ORANGE, True, label='flagged'), mk('o', BLUE, False, label='not flagged'),
                    mk('o', INK2, False, label='CT task'), mk('s', INK2, False, label='MRI task')] + ref_handles(),
           loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=6, handletextpad=0.4, columnspacing=1.4, handlelength=1.6)
save(fig, 'Fig2')

# ---- Fig. 3 (cited second in Results): field groups ----------------------------------------------------------------
GROUP_ORDER = ['scanner (vendor, model)', 'software version', 'image count', 'slice thickness and kernel', 'body part',
               'description words']
GROUP_LABEL = ['Scanner\n(vendor,\nmodel)', 'Software\nversion', 'Image\ncount', 'Slice\nthickness\nand kernel',
               'Body\npart', 'Description\nwords']
assert sorted(groups_fg.group.unique()) == sorted(GROUP_ORDER)
M = groups_fg.pivot(index='task_id', columns='group', values='alone_auc').loc[a.task_id, GROUP_ORDER]
fig = plt.figure(figsize=(FULL, 138 * MM))
ax = fig.add_axes([0.335, 0.012, 0.54, 0.895])
cax = fig.add_axes([0.905, 0.30, 0.016, 0.34])
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
ax.grid(which='minor', color='white', lw=1.4)
ax.tick_params(which='minor', length=0)
ax.plot([-0.62, len(GROUP_ORDER) - 0.5], [n - n_flag - 0.5] * 2, color=INK2, lw=0.8, clip_on=False)   # flagged tasks above
cb = fig.colorbar(im, cax=cax, ticks=[0.5, 0.6, 0.7, 0.8, 0.9])
cb.set_label('AUC of the field group alone', fontsize=FS)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=FS, length=2)
save(fig, 'Fig3')

# ---- Fig. 4: site or protocol ------------------------------------------------------------------------------------
s = sites.sort_values('overall_auc_same_patients').reset_index(drop=True)
fig = plt.figure(figsize=(FULL, 122 * MM))
AX4 = [0.335, 0.085, 0.635, 0.80]
ax = fig.add_axes(AX4)
bands(ax, len(s), (), (0.004 - AX4[0]) / AX4[2], (0.996 - AX4[0]) / AX4[2])
offs = {'overall': 0.3, 'site only': 0.1, 'sites held out': -0.1, 'within site': -0.3}
for i, r in s.iterrows():
    ax.scatter([r.overall_auc_same_patients], [i + offs['overall']], s=24, marker='D', color=INK, edgecolor='white',
               linewidth=0.5, zorder=3)
    ci_point(ax, r.site_only_auc, r.site_only_ci_low, r.site_only_ci_high, i + offs['site only'], BLUE, marker='o',
             size=22, lw=1.2)
    ci_point(ax, r.site_heldout_auc, r.site_heldout_ci_low, r.site_heldout_ci_high, i + offs['sites held out'], ORANGE,
             marker='s', size=20, lw=1.2)
    ci_point(ax, r.within_site_auc, r.within_site_ci_low, r.within_site_ci_high, i + offs['within site'], AQUA,
             marker='^', size=24, lw=1.2)
chance(ax, flag_line=False)
ax.set_yticks(range(len(s)), [short(t) for t in s.task_id])
ax.set_ylim(-0.5, len(s) - 0.5)
ax.set_xlim(0, 1)
ax.set_xlabel('AUC (95% CI)')
axis_style(ax)
fig.legend(handles=[mk('D', INK, True, ms=4.6, label='metadata, all patients'),
                    mk('o', BLUE, True, label='contributing site alone'),
                    mk('s', ORANGE, True, ms=4.8, label='metadata, whole sites held out'),
                    mk('^', AQUA, True, ms=5.6, label='metadata, pairs within one site')] + ref_handles(False),
           loc='upper left', bbox_to_anchor=(0.02, 0.995), ncol=3, handletextpad=0.4, columnspacing=1.6, handlelength=1.6)
save(fig, 'Fig4')

# ---- Fig. 5: diagnosis with segmented tissue removed ----------------------------------------------------------------
p5 = probe05.merge(sil06[['task_id', 'encoder', 'silhouette_auc', 'silhouette_ci_low', 'silhouette_ci_high',
                          'silhouette_flagged']], on=['task_id', 'encoder'], how='left', validate='one_to_one')
assert p5.silhouette_auc.notna().all()
order5 = a[a.flagged].sort_values('auc').task_id.tolist()
fig = plt.figure(figsize=(FULL, 84 * MM))
axes = [fig.add_axes([0.335, 0.135, 0.305, 0.635]), fig.add_axes([0.685, 0.135, 0.305, 0.635])]
for ax, enc in zip(axes, ENCODERS):
    d = p5[p5.encoder == enc].set_index('task_id').loc[order5]
    p = ax.get_position()
    bands(ax, len(order5), (), ((0.004 - p.x0) / p.width) if enc == 'ctfm' else 0.0,
          ((0.685 - p.x0) / p.width) if enc == 'ctfm' else (0.996 - p.x0) / p.width)
    for i, (t, r) in enumerate(d.iterrows()):
        ax.scatter([r.metadata_auc_same_patients], [i + 0.3], s=24, marker='D', color=INK, edgecolor='white',
                   linewidth=0.5, zorder=3)
        for k, yo in (('image', 0.1), ('background', -0.1), ('silhouette', -0.3)):
            ci_point(ax, r[f'{k}_auc'], r[f'{k}_ci_low'], r[f'{k}_ci_high'], i + yo, INPUT_COLOR[k],
                     filled=bool(r[f'{k}_flagged']), marker=INPUT_MARK[k], size=24)
    chance(ax)
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(-0.5, len(order5) - 0.5)
    ax.set_xlabel('AUC (95% CI)')
    axis_style(ax)
    panel_title(fig, ax, f'{"ab"[ENCODERS.index(enc)]}  {ENC[enc]}')
axes[0].set_yticks(range(len(order5)), [short(t) for t in order5])
axes[1].set_yticks(range(len(order5)), [''] * len(order5))
fig.legend(handles=[mk('D', INK, True, ms=4.6, label='metadata only'),
                    mk('o', BLUE, True, label='full image'),
                    mk('s', ORANGE, True, ms=4.8, label='background only (tissue removed)'),
                    mk('^', AQUA, True, ms=5.6, label='silhouette (body shape only)'),
                    mk('o', INK2, False, label='open marker: not flagged')] + ref_handles(),
           loc='upper left', bbox_to_anchor=(0.02, 0.995), ncol=4, handletextpad=0.4, columnspacing=1.4, handlelength=1.6)
save(fig, 'Fig5')

# ---- Fig. 6: validation of the screen ----------------------------------------------------------------------------
FLAG_ORDER = a[a.flagged].sort_values('auc', ascending=False).task_id.tolist()      # 1 = highest metadata AUC
NUM6 = {t: str(i + 1) for i, t in enumerate(FLAG_ORDER)}
KEY6 = {'renal subtype (KIRC vs KIRP)': 'KIRC vs KIRP', 'lung histology (LUAD vs LUSC)': 'LUAD vs LUSC',
        'LIDC-IDRI | primary vs metastatic': 'LIDC-IDRI primary vs metastatic', 'CPTAC-PDA | stage': 'CPTAC-PDA stage'}
fig = plt.figure(figsize=(FULL, 124 * MM))
axes = [fig.add_axes([0.075, 0.385, 0.40, 0.56]), fig.add_axes([0.585, 0.385, 0.40, 0.56])]
for ax, enc in zip(axes, ENCODERS):
    d = sv[sv.encoder == enc]
    q = d[~d.metadata_flagged.astype(bool)]
    ax.scatter(q.metadata_auc, q.background_auc, s=26, facecolor='white', edgecolor=BLUE, linewidth=1.2, zorder=3)
    q = d[d.metadata_flagged.astype(bool)]
    ax.scatter(q.metadata_auc, q.background_auc, s=120, facecolor=ORANGE, edgecolor='white', linewidth=0.7, zorder=4)
    bf = d[d.background_flagged.astype(bool)]
    ax.scatter(bf.metadata_auc, bf.background_auc, s=230, facecolor='none', edgecolor=INK, linewidth=1.0, zorder=5)
    for r in q.itertuples():
        ax.text(r.metadata_auc, r.background_auc - 0.001, NUM6[r.task_id], ha='center', va='center', fontsize=FS,
                fontweight='bold', color='white', zorder=6)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(0.3, 0.9); ax.set_ylim(0.3, 0.9)
    ax.set_xlabel('Metadata-only AUC')
    ax.set_ylabel('Background-only image AUC')
    rr = robust[(robust['mode'] == 'real') & (robust.encoder == enc)].set_index('check')
    lines = [('All 19 tasks', 'all 19 tasks (notebook 07, recomputed)'),
             ('15 unflagged tasks (post hoc)', 'without the flagged tasks'),
             ('11 patient-disjoint groups (post hoc)', 'patient-disjoint task groups (group means)')]
    ax.text(0.0, -0.185, 'Spearman correlation', transform=ax.transAxes, ha='left', va='top', fontsize=FS, fontweight='bold')
    ax.text(0.80, -0.185, 'ρ', transform=ax.transAxes, ha='right', va='top', fontsize=FS, fontweight='bold')
    ax.text(1.0, -0.185, 'P', transform=ax.transAxes, ha='right', va='top', fontsize=FS, fontweight='bold')
    ax.plot([0, 1], [-0.235, -0.235], transform=ax.transAxes, color=AXIS, lw=0.6, clip_on=False)
    for j, (lab, k) in enumerate(lines):
        yy = -0.255 - 0.065 * j
        ax.text(0.0, yy, lab, transform=ax.transAxes, ha='left', va='top', fontsize=FS, color=INK)
        ax.text(0.80, yy, f'{rr.loc[k, "spearman_rho"]:.2f}'.replace('-', '−'), transform=ax.transAxes, ha='right',
                va='top', fontsize=FS)
        ax.text(1.0, yy, f'{rr.loc[k, "perm_p_one_sided"]:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=FS)
    panel_title(fig, ax, f'{"ab"[ENCODERS.index(enc)]}  {ENC[enc]}{" (primary encoder)" if enc == "ctfm" else ""}')
    axis_style(ax, grid='both', left=True)
fig.text(0.5, 0.062, 'Flagged tasks:  ' + ';  '.join(f'{NUM6[t]}  {KEY6[t]}' for t in FLAG_ORDER), ha='center',
         va='bottom', fontsize=FS)
fig.legend(handles=[mk('o', ORANGE, True, ms=7, label='flagged by the metadata screen (numbered)'),
                    mk('o', BLUE, False, label='not flagged'),
                    Line2D([], [], marker='o', ls='', mfc='none', mec=INK, ms=10, mew=1.0,
                           label='ring: background-only probe flagged')],
           loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, handletextpad=0.4, columnspacing=1.6)
save(fig, 'Fig6')

# ---- Fig. 7: external transfer (five pairs and the CPTAC decomposition) ------------------------------------------
PANELS = [(transfer11, 'lung: TCGA -> NSCLC Radiogenomics', 'a  TCGA lung → NSCLC Radiogenomics (independent)'),
          (transfer11, 'lung: TCGA -> NSCLC-Radiomics', 'b  TCGA lung → NSCLC-Radiomics (independent, added)'),
          (transfer11, 'lung: TCGA -> Lung-PET-CT-Dx', 'c  TCGA lung → Lung-PET-CT-Dx (independent, added)'),
          (transfer11, 'lung: TCGA -> CPTAC', 'd  TCGA lung → CPTAC lung (assembled like TCGA)'),
          (transfer, 'kidney: TCGA -> C4KC-KiTS', 'e  TCGA kidney → C4KC-KiTS')]
ROWS7 = [('ctfm', 'image'), ('ctfm', 'background'), ('merlin', 'image'), ('merlin', 'background')]
strict7 = pd.read_csv(IN['12'] / 'r7_external_strict.csv').set_index(['pair', 'encoder', 'input'])
assert len(strict7) == 20


def level7(pair, key, r):
    """One rule for every transfer: external CI above 0.5 and Holm-corrected target-label P below 0.05."""
    above = bool(r.external_ci_low > 0.5)
    assert above == (r.reading != 'does not transfer'), (pair, key)
    return level(above, bool(strict7.loc[(pair, key[0], key[1]), 'survives_holm_0.05']))


fig = plt.figure(figsize=(FULL, 176 * MM))
xs, ys, pw, ph = (0.215, 0.705), (0.675, 0.385, 0.075), 0.27, 0.21
axes = [fig.add_axes([xs[i % 2], ys[i // 2], pw, ph]) for i in range(6)]
for ax, (tab, pair, title) in zip(axes[:5], PANELS):
    d = tab[tab.pair == pair].set_index(['encoder', 'input'])
    bands(ax, 4, (), -xs[0] / pw + 0.004 / pw, 1.0)
    for i, key in enumerate(ROWS7[::-1]):
        r = d.loc[key]
        ci_point(ax, r.source_internal_auc, r.source_internal_ci_low, r.source_internal_ci_high, i + 0.26, MUTED,
                 marker='D', size=15, lw=1.0)
        ci_point(ax, r.external_auc, r.external_ci_low, r.external_ci_high, i, INPUT_COLOR[key[1]],
                 filled=level7(pair, key, r), marker=INPUT_MARK[key[1]], size=28, lw=1.5)
        ci_point(ax, r.target_internal_auc, r.target_internal_ci_low, r.target_internal_ci_high, i - 0.26, INK,
                 marker='v', filled=False, size=18, lw=1.0)
    chance(ax, flag_line=False)
    ax.set_yticks(range(4), [f'{ENC[e]}, {INPUT_NAME[k]}' for e, k in ROWS7[::-1]])
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.5, 3.5)
    axis_style(ax)
ax = axes[5]
rows_d = []
for r in sil09[sil09.target == 'CPTAC lung'].itertuples():
    e = transfer[(transfer.pair == 'lung: TCGA -> CPTAC') & (transfer.encoder == r.encoder)
                 & (transfer.input == 'background')].iloc[0]
    rows_d.append((f'{ENC[r.encoder]}, background only', e.external_auc, e.external_ci_low, e.external_ci_high, ORANGE,
                   'background', level7('lung: TCGA -> CPTAC', (r.encoder, 'background'), e)))
for r in sil09[sil09.target == 'CPTAC lung'].itertuples():
    rows_d.append((f'{ENC[r.encoder]}, silhouette', r.silhouette_external_auc, r.silhouette_ci_low, r.silhouette_ci_high,
                   AQUA, 'silhouette', 'light' if bool(r.silhouette_transfers) else 'open'))   # not in the corrected test
h = hb09[hb09.target == 'CPTAC lung'].iloc[0]
rows_d.append(('DICOM header fields', h.header_external_auc, h.header_ci_low, h.header_ci_high, INK2, 'header',
               'light' if bool(h.header_transfers) else 'open'))
rows_d.append(('body size', h.body_size_external_auc, h.body_size_ci_low, h.body_size_ci_high, INK2, 'body',
               'light' if bool(h.body_size_transfers) else 'open'))
bands(ax, len(rows_d), (), -xs[0] / pw + 0.004 / pw, 1.0)
for i, (lab, v, lo, hi, c, kind, filled) in enumerate(rows_d[::-1]):
    ci_point(ax, v, lo, hi, i, c, filled=filled, marker=INPUT_MARK.get(kind, 'P'), size=28, lw=1.5)
chance(ax, flag_line=False)
ax.set_yticks(range(len(rows_d)), [r[0] for r in rows_d[::-1]])
ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.5, len(rows_d) - 0.5)
axis_style(ax)
titles = [t_ for *_, t_ in PANELS] + ['f  TCGA lung → CPTAC lung, decomposed (post hoc)']
for ax, t_, x in zip(axes, titles, (0.012, 0.502) * 3):
    fig.text(x, ax.get_position().y1 + 0.007, t_, ha='left', va='bottom', fontsize=9, fontweight='bold')
for ax in axes[4:]:
    ax.set_xlabel('AUC (95% CI)')
fig.legend(handles=[mk('D', MUTED, True, ms=4.2, label='source, cross-validated'),
                    mk('v', INK, False, label='target, cross-validated'),
                    mk('o', BLUE, True, label='external: full image'),
                    mk('s', ORANGE, True, ms=4.8, label='external: background only'),
                    mk('^', AQUA, True, ms=5.6, label='external: silhouette'),
                    mk('P', INK2, True, ms=5.6, label='external: header fields, body size'),
                    mk('o', INK2, True, label='filled: transfers'),
                    mk('o', INK2, 'light', label='light: CI above 0.5 only'),
                    mk('o', INK2, False, label='open: CI includes 0.5')] + ref_handles(False),
           loc='upper left', bbox_to_anchor=(0.012, 0.998), ncol=4, handletextpad=0.4, columnspacing=1.5, handlelength=1.6)
save(fig, 'Fig7')

# ---- Fig. 8: source of the tissue-free signal in lung histology (Step 4; plan 7o, notebooks 12 and 13) ---------
G = RES / '12_image_experiments' / 'real'
C13 = RES / '13_cnn' / 'real'
for _folder in (G, C13):
    _m = verify(_folder)
    assert _m['in_colab'] and _m.get('labels', _m.get('mode')) == 'real'
LUNG = 'lung histology (LUAD vs LUSC)'
g1i = pd.read_csv(G / 'g1_internal.csv'); g1i = g1i[g1i.dataset == LUNG].set_index(['input', 'encoder'])
g1t = pd.read_csv(G / 'g1_transfer.csv').set_index(['input', 'encoder', 'target'])
g2i = pd.read_csv(G / 'g2_internal.csv').set_index(['model', 'encoder'])
g2t = pd.read_csv(G / 'g2_transfer.csv').set_index(['input', 'encoder', 'target'])
g3i = pd.read_csv(G / 'g3_internal.csv').set_index(['input', 'encoder'])
g3t = pd.read_csv(G / 'g3_transfer.csv').set_index(['input', 'encoder', 'target'])
c13i = pd.read_csv(C13 / 'cnn_internal.csv').set_index('input')
c13t = pd.read_csv(C13 / 'cnn_transfer.csv').set_index(['input', 'target'])
TARGETS8 = [('CPTAC lung', 'lung: TCGA -> CPTAC', 'b  → CPTAC\nlung'),
            ('NSCLC Radiogenomics', 'lung: TCGA -> NSCLC Radiogenomics', 'c  → NSCLC\nRadiogenomics'),
            ('NSCLC-Radiomics', 'lung: TCGA -> NSCLC-Radiomics', 'd  → NSCLC-\nRadiomics'),
            ('Lung-PET-CT-Dx', 'lung: TCGA -> Lung-PET-CT-Dx', 'e  → Lung-PET-\nCT-Dx')]
# (row label, colour, kind, key): the colour names the kind of input (full image, background, region, none)
ROWS8 = [('Full image (Steps 2–3)', BLUE, 'native', 'image'),
         ('Background only (Steps 2–3)', ORANGE, 'native', 'background'),
         ('Lungs only', AQUA, 'g1', 'lungs'),
         ('Tumour only', AQUA, 'g1', 'tumour'),
         ('Habitus model, seven measures', INK2, 'habitus', None),
         ('Background only,\nhabitus removed', ORANGE, 'resid', 'background, habitus removed'),
         ('Harmonized geometry,\nfull image', BLUE, 'g3', 'harm_full'),
         ('Harmonized geometry,\nbackground only', ORANGE, 'g3', 'harm_background'),
         ('3D network, full image', BLUE, 'cnn', 'full'),
         ('3D network, background only', ORANGE, 'cnn', 'background')]
_src = transfer11[transfer11.pair == 'lung: TCGA -> CPTAC'].set_index(['encoder', 'input'])


def internal8(ci_low):
    """Marker state inside TCGA: solid if the lower CI bound is at least 0.60, light if it is above 0.5, otherwise open."""
    return 'solid' if ci_low >= 0.60 else ('light' if ci_low > 0.5 else 'open')


def cells8(kind, key, target, pair):
    """[(marker, auc, low, high, filled)] of one row in one panel; target None is TCGA by cross-validation."""
    out = []
    if kind in ('native', 'g1', 'resid', 'g3'):
        for enc in ENCODERS:
            if target is None:
                if kind == 'native':
                    r = _src.loc[(enc, key)]; v = (r.source_internal_auc, r.source_internal_ci_low, r.source_internal_ci_high)
                else:
                    r = {'g1': g1i, 'resid': g2i, 'g3': g3i}[kind].loc[(key, enc)]; v = (r.auc, r.ci_low, r.ci_high)
                out.append(('o' if enc == 'ctfm' else 's', *v, internal8(v[1])))
            elif kind == 'native':
                r = transfer11[transfer11.pair == pair].set_index(['encoder', 'input']).loc[(enc, key)]
                out.append(('o' if enc == 'ctfm' else 's', r.external_auc, r.external_ci_low, r.external_ci_high,
                            level7(pair, (enc, key), r)))
            else:
                r = {'g1': g1t, 'resid': g2t, 'g3': g3t}[kind].loc[(key, enc, target)]
                out.append(('o' if enc == 'ctfm' else 's', r.external_auc, r.ci_low, r.ci_high,
                            level(bool(r.transfers_ci), bool(r.transfers))))
    elif kind == 'habitus':
        if target is None:
            r = g2i.loc[('habitus', '-')]; out.append(('D', r.auc, r.ci_low, r.ci_high, internal8(r.ci_low)))
        else:
            r = g2t.loc[('habitus model', '-', target)]
            out.append(('D', r.external_auc, r.ci_low, r.ci_high, level(bool(r.transfers_ci), bool(r.transfers))))
    else:
        if target is None:
            r = c13i.loc[key]; out.append(('D', r.auc, r.ci_low, r.ci_high, internal8(r.ci_low)))
        else:
            r = c13t.loc[(key, target)]
            out.append(('D', r.external_auc, r.ci_low, r.ci_high, level(bool(r.transfers_ci), bool(r.transfers))))
    return out


fig = plt.figure(figsize=(FULL, 124 * MM))
x0, pw, gap, yb, ph8 = 0.245, 0.130, 0.021, 0.085, 0.725
axes = [fig.add_axes([x0 + i * (pw + gap), yb, pw, ph8]) for i in range(5)]
PANELS8 = [(None, None, 'a  TCGA lung\n(internal)')] + TARGETS8
n8 = len(ROWS8)
for k, (ax, (target, pair, title)) in enumerate(zip(axes, PANELS8)):
    bands(ax, n8, (), ((0.004 - x0) / pw) if k == 0 else -gap / pw, 1.0 if k < 4 else (0.996 - ax.get_position().x0) / pw)
    for i, (lab, col, kind, key) in enumerate(ROWS8):
        yc = n8 - 1 - i
        cs = cells8(kind, key, target, pair)
        offs8 = [0.17, -0.17] if len(cs) == 2 else [0.0]
        for (mkr, v, lo, hi, filled), dy in zip(cs, offs8):
            ci_point(ax, v, lo, hi, yc + dy, col, filled=filled, marker=mkr, size=18 if mkr == 'D' else 21, lw=1.2)
    chance(ax, flag_line=False)
    ax.set_xlim(0.25, 0.95); ax.set_ylim(-0.5, n8 - 0.5)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_yticks(range(n8), [r[0] for r in ROWS8[::-1]] if ax is axes[0] else [''] * n8)
    axis_style(ax)
    ax.set_xlabel('AUC (95% CI)')
    fig.text(ax.get_position().x0, ax.get_position().y1 + 0.012, title, ha='left', va='bottom', fontsize=9,
             fontweight='bold', linespacing=1.15)
fig.legend(handles=[mk('o', INK2, True, label='CT-FM'), mk('s', INK2, True, ms=4.8, label='Merlin'),
                    mk('D', INK2, True, ms=4.4, label='no encoder (habitus model, 3D network)'),
                    mk('o', INK2, True, label='filled: lower CI bound ≥ 0.60 (a); transfers (b–e)'),
                    mk('o', INK2, 'light', label='light: CI above 0.5 only'),
                    mk('o', INK2, False, label='open: CI includes 0.5')] + ref_handles(False),
           loc='upper left', bbox_to_anchor=(0.012, 0.998), ncol=3, handletextpad=0.4, columnspacing=1.6, handlelength=1.6)
save(fig, 'Fig8')

# ---- supplementary figures S2 and S3 ------------------------------------------------------------------------------
TARGETS_C = decoding.target.unique().tolist()
cols = [(e, k) for e in ENCODERS for k in ('image', 'background', 'silhouette')]
D = decoding.set_index(['target', 'encoder', 'input'])
vals = np.array([[D.loc[(t, e, k), 'pooled_auc'] for e, k in cols] for t in TARGETS_C], dtype=float)
H1 = 9 * len(TARGETS_C) + 26
fig = plt.figure(figsize=(FULL, H1 * MM))
ax = fig.add_axes([0.3, 4 / H1, 0.54, 1 - 4 / H1 - 21 / H1])
cax = fig.add_axes([0.87, 0.14, 0.018, 0.46])
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
ax.grid(which='minor', color='white', lw=1.4); ax.tick_params(which='minor', length=0)
ax.plot([2.5, 2.5], [-0.5, len(TARGETS_C) - 0.5], color='white', lw=4)             # the two encoders, set apart
cb = fig.colorbar(im, cax=cax, ticks=[0.5, 0.75, 1.0])
cb.set_label('Decoding AUC, pooled', fontsize=FS)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=FS, length=2)
save(fig, 'FigS3', esm=True)

# one panel per dataset (five datasets cannot be told apart by colour alone)
DS = [('lung histology (LUAD vs LUSC)', 'a  TCGA lung'), ('CPTAC lung', 'b  CPTAC lung'),
      ('NSCLC Radiogenomics', 'c  NSCLC\nRadiogenomics'), ('NSCLC-Radiomics', 'd  NSCLC-\nRadiomics'),
      ('Lung-PET-CT-Dx', 'e  Lung-PET-\nCT-Dx')]
FIELD_NAME = {'hdr_slice_thickness': 'Slice thickness', 'hdr_z_spacing': 'Slice spacing',
              'hdr_recon_diameter': 'Reconstruction diameter', 'hdr_kvp': 'kVp', 'hdr_tube_current': 'Tube current',
              'body_area_cm2': 'Body cross-section area'}
fig = plt.figure(figsize=(FULL, 78 * MM))
x0, pw, gap = 0.215, 0.137, 0.02
axes = [fig.add_axes([x0 + i * (pw + gap), 0.15, pw, 0.67]) for i in range(5)]
fl = list(FIELD_NAME)[::-1]
for k, (ax, (ds, title)) in enumerate(zip(axes, DS)):
    bands(ax, len(fl), (), ((0.004 - x0) / pw) if k == 0 else -gap / pw, 1.0 if k < 4 else (0.996 - ax.get_position().x0) / pw)
    for i, f in enumerate(fl):
        r = fields11[(fields11.dataset == ds) & (fields11.field == f)].iloc[0]
        if np.isfinite(r.auc_squamous_higher):
            ci_point(ax, r.auc_squamous_higher, r.ci_low, r.ci_high, i, BLUE,
                     size=22, lw=1.2)
        else:
            ax.text(0.525, i, 'one value', fontsize=FS, color=INK2, va='center', ha='left')
    chance(ax, flag_line=False)
    ax.set_yticks(range(len(fl)), [FIELD_NAME[f] for f in fl] if k == 0 else [''] * len(fl))
    ax.set_ylim(-0.5, len(fl) - 0.5); ax.set_xlim(0.15, 0.9)
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.set_xlabel('AUC (95% CI)')
    axis_style(ax)
    fig.text(ax.get_position().x0, ax.get_position().y1 + 0.02, title, ha='left', va='bottom', fontsize=9,
             fontweight='bold', linespacing=1.15)
save(fig, 'FigS2', esm=True)

import shutil                                        # Fig. S1: the mask quality-control pictures (review check R6)
from PIL import Image, ImageDraw, ImageFont
# The QC pictures saved by notebook 06 print the scan name squeezed into the panel titles. The images themselves (from
# row 34 down, identical in all 8 pictures) are kept unchanged; above them, the scan name and the three panel titles
# (same words) are redrawn in a clean header. Order as in results/12_review_checks/r6_mask_qc.png (sorted by file name).
_pics = sorted((IN['12'].parent / '06_image_posthoc' / 'cache' / 'qc').glob('*.png'))
assert len(_pics) == 8, len(_pics)
_TOP, _CENTRES = 34, (163, 479, 795)
_TITLES = ('original + body outline', 'background only (notebook 05)', 'silhouette (this notebook)')


def _font(names, size):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


_fb = _font(('arialbd.ttf', 'Arial Bold.ttf', 'DejaVuSans-Bold.ttf'), 15)
_fr = _font(('arial.ttf', 'Arial.ttf', 'DejaVuSans.ttf'), 13)
_rows = []
for _p in _pics:
    _im = Image.open(_p).convert('RGB')
    assert _im.size == (960, 336), (_p.name, _im.size)
    _body = _im.crop((0, _TOP, _im.width, _im.height))
    _row = Image.new('RGB', (_im.width, 52 + _body.height), 'white')
    _d = ImageDraw.Draw(_row)
    _d.text((_im.width // 2, 13), _p.stem.replace('__', ': '), fill='black', font=_fb, anchor='mm')
    for _x, _t in zip(_CENTRES, _TITLES):
        _d.text((_x, 38), _t, fill='black', font=_fr, anchor='mm')
    _row.paste(_body, (0, 52))
    _rows.append(_row)
_qc = Image.new('RGB', (max(r.width for r in _rows), sum(r.height for r in _rows)), 'white')
_y = 0
for _r in _rows:
    _qc.paste(_r, (0, _y))
    _y += _r.height
_qc.save(OUTDIR / 'esm_png' / 'FigS1.png')
# Online Resource 1 prints Fig. S1 in two parts (rows 1-4 and 5-8) so that the labels stay readable
_h = 4 * _rows[0].height
_qc.crop((0, 0, _qc.size[0], _h)).save(OUTDIR / 'esm_png' / 'FigS1a.png')
_qc.crop((0, _h, _qc.size[0], _qc.size[1])).save(OUTDIR / 'esm_png' / 'FigS1b.png')
SAVED['FigS1'] = 'rebuilt from results/06_image_posthoc/cache/qc (as results/12_review_checks/r6_mask_qc.png, with clear scan labels)'

json.dump({'sizes_mm': SAVED, 'matplotlib': matplotlib.__version__, 'inputs_verified': sorted(man)},
          open(OUTDIR / 'figures_record.json', 'w'), indent=1)
print('matplotlib', matplotlib.__version__)
