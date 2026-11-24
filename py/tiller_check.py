"""Measure where each limb of a bow bends, from tiller photos.

Subcommands
  grid IMG [--step 100]            write IMG_grid.png: a copy with labelled pixel gridlines, so the
                                   assistant can read off panel boxes and anchor points by eye.
  crop IMG --box x0,y0,x1,y1 --out PANEL.png
  measure --drawn PANEL.png --drawn-pts LT;LH;RH;RT [--rest PANEL.png --rest-pts ...]
          [--front PANEL.png] [--out DIR]
        Points are x,y pixels in that panel: LT left tip, LH left limb's handle end (where the fade
        meets the limb), RH right handle end, RT right tip. The tool finds the lowest-cost path along
        pale wood between each pair, so the points only need to be close (within ~2 limb widths).

What it reports: each limb split into 10 equal stations by length (1 = handle end, 10 = tip) and
the percent of that limb's bend that happens in each station; inner/mid/outer thirds; left vs right
total bend; and, with --rest, bend measured from the unbraced shape so string follow and reflex do
not count. Overlay PNGs show the trace colour-coded (blue = little bend, red = most) for checking.

Requirements: camera square to the bow; whole bow in frame. Without --grid the bow must be paler
than the background. With --grid (a gridded backdrop behind the bow, any colour) the tool models the
backdrop colour, erases the thin grid lines, finds the grid pitch, and with --grid-in (size of one
square in inches) reports lengths and tip drop in inches plus a camera-squareness check.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter

N_BINS = 10
HINGE_DEG = 8       # polynomial degree for the hinge profile: keeps a kink, drops pixel noise
HINGE_EXCESS = 16.0 # [tool] 10-station view; calibrated on synthetic kinks and the sample bows
HINGE_EXCESS_20 = 9.0  # [tool] 20 half-station view (sharper, catches outer-limb kinks)
GAP_MAX = 0.15     # a station whose trace is more than 15% off visible wood is unsure


def load(p):
    return np.asarray(Image.open(p).convert('RGB'))


def paleness(rgb):
    """0..1 score: bow wood is bright and low-saturation; brown walls, pine posts score low."""
    hsv = rgb2hsv(rgb)
    v, s = hsv[..., 2], hsv[..., 1]
    return np.clip((v - s - 0.15) / 0.45, 0, 1) * (v > 0.55)


GRID = {'on': False, 'line_px': 7}


def backdrop_score(rgb):
    """0..1 score for 'bow, not board': CIELAB distance from BOTH the board colour and the grid-line
    colour (whichever is nearer). Lines are erased by colour, not by thickness, so thin limb tips
    survive. A 3 px opening cleans the anti-aliased line edges."""
    from scipy import ndimage
    lab = rgb2lab(rgb)
    flat = lab.reshape(-1, 3)
    bg = np.median(flat, 0)
    dbg = np.linalg.norm(lab - bg, axis=2)
    far = flat[dbg.ravel() > 20]
    if len(far) > 0.01 * len(flat):                # grid lines are the commonest non-board colour
        q = np.round(far / 8)
        vals, cnt = np.unique(q, axis=0, return_counts=True)
        line = vals[np.argmax(cnt)] * 8
        d = np.minimum(dbg, np.linalg.norm(lab - line, axis=2))
    else:
        d = dbg
    k = GRID['line_px']
    if k > 1:
        d = ndimage.grey_opening(d, size=(k, k))
    return np.clip((d - 8) / 22, 0, 1)             # dE 8 = board noise, dE 30+ = clearly not board


_CACHE = {}


def wood(rgb):
    key = (rgb.shape, hash(rgb[::37, ::41].tobytes()), GRID['on'], GRID['line_px'])
    if key not in _CACHE:
        _CACHE[key] = backdrop_score(rgb) if GRID['on'] else paleness(rgb)
    return _CACHE[key]


def grid_pitch(rgb):
    """Grid square size in px, found from the period of the line pattern (autocorrelation of the
    high-pass brightness, summed along rows/columns). Returned per direction and per half of the
    panel; left and right halves differing means the camera was angled (perspective)."""
    from scipy import ndimage
    L = rgb.mean(2)
    hp = np.abs(L - ndimage.gaussian_filter(L, 6))
    hp[wood(rgb) > 0.3] = 0                        # ignore the bow itself
    def period(sig):
        """Grid period from the autocorrelation: the strongest peak, then stepped down to its
        fundamental while a peak near half the lag is nearly as strong (lines repeat at every
        multiple of the square size, so the first strong peak is the square)."""
        sig = sig - sig.mean()
        ac = np.correlate(sig, sig, 'full')[len(sig) - 1:]
        ac /= ac[0] + 1e-9
        lo, hi = 8, len(ac) // 3
        pk = [i for i in range(lo, hi) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.15]
        if not pk:
            return None
        best = max(pk, key=lambda i: ac[i])
        while True:
            half = [i for i in pk if abs(i - best / 2) <= max(2, best * 0.06) and ac[i] > 0.6 * ac[best]]
            if not half:
                break
            best = max(half, key=lambda i: ac[i])
        i = best
        a0, b0, c0 = ac[i - 1], ac[i], ac[i + 1]
        return i + 0.5 * (a0 - c0) / (a0 - 2 * b0 + c0 + 1e-12)
    H, W = L.shape
    res = {'x_px': period(hp.mean(0)), 'y_px': period(hp.mean(1)),
           'x_left_px': period(hp[:, :W // 2].mean(0)), 'x_right_px': period(hp[:, W // 2:].mean(0))}
    return {k: (round(float(v), 1) if v else None) for k, v in res.items()}


def rgb2hsv(rgb):
    """RGB (uint8 or 0..1 float) to HSV in 0..1 (numpy only, so the phone needs no scikit-image)."""
    x = rgb.astype(float) / (255.0 if rgb.dtype == np.uint8 else 1.0)
    mx = x.max(-1); mn = x.min(-1); d = mx - mn
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    dd = np.where(d > 0, d, 1)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) / 6
    h = np.where(d > 0, h, 0)
    return np.stack([h, s, mx], -1)


def rgb2lab(rgb):
    """sRGB to CIELAB (D65), numpy only."""
    x = rgb.astype(float) / (255.0 if rgb.dtype == np.uint8 else 1.0)
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    xyz = x @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    return np.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _route(cost, a, b, margin=40):
    """Lowest-cost 8-connected path from a to b ((row, col)) over `cost`, geometric step lengths.
    Runs scipy's Dijkstra on the box around the two points (plus margin), so no scikit-image."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    H, W = cost.shape
    r0, r1 = max(0, min(a[0], b[0]) - margin), min(H, max(a[0], b[0]) + margin + 1)
    c0, c1 = max(0, min(a[1], b[1]) - margin), min(W, max(a[1], b[1]) + margin + 1)
    sub = cost[r0:r1, c0:c1]; h, w = sub.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, wts = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        sa = idx[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
        sb = idx[max(0, dr):h + min(0, dr) if dr else h, max(0, dc):w + min(0, dc) if dc else w]
        ca = sub[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
        cb = sub[max(0, dr):h + min(0, dr) if dr else h, max(0, dc):w + min(0, dc) if dc else w]
        wt = (ca + cb) / 2 * (np.sqrt(2) if dr and dc else 1.0)
        rows.append(sa.ravel()); cols.append(sb.ravel()); wts.append(wt.ravel())
    rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    g = coo_matrix((np.concatenate([wts, wts]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(h * w, h * w)).tocsr()
    src = idx[a[0] - r0, a[1] - c0]; dst = idx[b[0] - r0, b[1] - c0]
    _, pred = dijkstra(g, indices=src, return_predecessors=True)
    path = [dst]
    while path[-1] != src and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    path = np.array(path[::-1])
    return np.stack([path // w + r0, path % w + c0], 1)


_TRACE = {}


def trace(rgb, a, b):
    """Lowest-cost path along the CENTRE of the pale limb from point a to b (x,y). Returns (N,2) x,y.
    Cost falls with paleness and with distance from the limb's edge, so the path rides the centreline.
    Routed at half resolution (4x fewer nodes, same shape once smoothed) and cached per image+ends."""
    from scipy import ndimage
    key = (rgb.shape, hash(rgb[::37, ::41].tobytes()), GRID['on'], tuple(a), tuple(b))
    if key in _TRACE:
        return _TRACE[key]
    pale = wood(rgb)
    dt = ndimage.distance_transform_edt(pale > 0.35)
    cost = 1.0 / ((pale ** 2 + 0.01) * (1 + dt) ** 2)
    c2 = cost[: cost.shape[0] // 2 * 2, : cost.shape[1] // 2 * 2]
    c2 = c2.reshape(c2.shape[0] // 2, 2, c2.shape[1] // 2, 2).mean((1, 3))
    A = (min(a[1] // 2, c2.shape[0] - 1), min(a[0] // 2, c2.shape[1] - 1))
    B = (min(b[1] // 2, c2.shape[0] - 1), min(b[0] // 2, c2.shape[1] - 1))
    rc = _route(c2, A, B, margin=max(40, c2.shape[0] // 3)) * 2 + 0.5   # room for the arc to bulge
    xy = rc[:, ::-1].astype(float)
    xy[0] = a; xy[-1] = b
    _TRACE[key] = xy
    return xy



def bezier_fit(xy, deg=5, iters=4):
    """Robust least-squares Bezier (Bernstein) fit of the traced points, parameterised by arc length.
    Points far from the curve (trace wandering onto a grid line, the string, a knot shadow) are
    down-weighted each pass (Tukey biweight), so one detour cannot bend the limb."""
    from math import comb
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]); t = s / s[-1]
    B = np.stack([comb(deg, k) * t ** k * (1 - t) ** (deg - k) for k in range(deg + 1)], 1)
    w = np.ones(len(t))
    for _ in range(iters):
        W = w[:, None]
        P, *_ = np.linalg.lstsq(B * W, xy * W, rcond=None)
        res = np.linalg.norm(B @ P - xy, axis=1)
        c = 4.685 * max(np.median(res) / 0.6745, 1.0)
        w = np.where(res < c, (1 - (res / c) ** 2) ** 2, 0.0)
    return P


def bezier_eval(P, t):
    from math import comb
    n = len(P) - 1
    B = np.stack([comb(n, k) * t ** k * (1 - t) ** (n - k) for k in range(n + 1)], 1)
    dP = n * np.diff(P, axis=0)
    dB = np.stack([comb(n - 1, k) * t ** k * (1 - t) ** (n - 1 - k) for k in range(n)], 1)
    return B @ P, dB @ dP


def bend_profile(xy, n=240, deg='default'):
    """deg='default': robust Bezier fit (smooth limb shape). deg=None: raw trace, savgol only, used
    to look for hinge spikes the smooth fit would hide."""
    if deg == 'default':
        P = bezier_fit(xy, BEZ_DEG)
        tt = np.linspace(0, 1, 2000)
        pts, d = bezier_eval(P, tt)
        sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        L = sl[-1]; f = np.linspace(0, 1, n)
        tq = np.interp(f * L, sl, tt)
        pts, d = bezier_eval(P, tq)
        ang = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
        return f, ang - ang[0], L, pts
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    L = s[-1]
    t = np.linspace(0, L, n)
    x = np.interp(t, s, xy[:, 0]); y = np.interp(t, s, xy[:, 1])
    win = max(9, (n // 12) | 1)
    x = savgol_filter(x, win, 3); y = savgol_filter(y, win, 3)
    ang = np.degrees(np.unwrap(np.arctan2(np.gradient(y), np.gradient(x))))
    f = t / L
    if deg:
        ang = np.polyval(np.polyfit(f, ang, deg), f)
    return f, ang - ang[0], L, np.stack([x, y], 1)


BEZ_DEG = 6   # degree-6 Bezier: enough freedom for an elliptical or circular limb and a stiff/soft
              # zone ~1/5 limb long; too little to follow pixel jitter or a short detour
POLY_DEG = 4


def stations(frac, ang, rest=None):
    e = np.linspace(0, 1, N_BINS + 1)
    d = np.diff(np.interp(e, frac, ang))
    if rest is not None:
        d = d - np.diff(np.interp(e, rest[0], rest[1]))
    return np.abs(d)


def parse_pts(s):
    pts = [tuple(int(float(v)) for v in p.split(',')) for p in s.split(';')]
    if len(pts) != 4:
        sys.exit('need 4 points: LT;LH;RH;RT')
    return pts


TRIM_HANDLE, TRIM_TIP = 0.02, 0.05   # drop fade hook and nock/string hook; the nock does not bend


HANDLE_FRAC = 0.08   # handle + fade, each side of centre, as a share of the tip-to-tip span (about
                     # 5 in on a 66 in bow); LH/RH land just past the fades. Tune per bow if needed.


def _walk(sc, p, d, step, stop=0.12, max_miss=15, cone=35):
    """Follow the wood ridge from p in direction d; gaps up to max_miss steps are jumped (glare,
    a knot, a grid line). Returns the last point that was on wood."""
    H, W = sc.shape; p = np.array(p, float); d = np.array(d, float); d /= np.linalg.norm(d)
    last_good = p.copy(); miss = 0
    angs = np.radians(np.arange(-cone, cone + 1, 5))
    for _ in range(4000):
        best, bv = None, -1.0
        for ang in angs:
            c, sn = np.cos(ang), np.sin(ang)
            nd = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
            q = p + nd * step
            x, y = int(round(q[0])), int(round(q[1]))
            if not (0 <= x < W and 0 <= y < H):
                continue
            v = sc[y, x] - 0.002 * abs(np.degrees(ang))           # prefer going straight
            if v > bv:
                bv, best = v, (q, nd)
        if best is None:
            break
        q, nd = best
        if sc[int(round(q[1])), int(round(q[0]))] < stop:
            miss += 1
            if miss > max_miss:
                break
        else:
            miss = 0; last_good = q.copy()
        d = 0.6 * d + 0.4 * nd; d /= np.linalg.norm(d); p = q
    return last_good


def auto_anchors(rgb, handle_frac=HANDLE_FRAC):
    """Find LT, LH, RH, RT without hand-picked points. The photo is framed with the bow across the
    middle (the app's guide line), so the handle is the row in the centre band with the most wood.
    From there the ridge of wood is walked left and right to the tips; a lowest-cost trace tip to
    tip follows the limb, and the handle ends are trace points handle_frac of the span either side
    of the centre. Raises ValueError with a plain retake reason when the result is not believable."""
    from scipy import ndimage
    H, W = rgb.shape[:2]
    w = wood(rgb)
    sc = ndimage.gaussian_filter(w, 1.5)
    band = slice(int(W * 0.4), int(W * 0.6))
    frac = ndimage.uniform_filter1d((w[:, band] > 0.3).mean(1), 9)
    if frac.max() < 0.3:
        raise ValueError('no bow in the middle of the photo: line the handle up with the guide')
    seed = (W // 2, int(np.argmax(frac)))
    step = max(4, W // 300)
    lt = _walk(sc, seed, (-1, 0), step); rt = _walk(sc, seed, (1, 0), step)
    lt = (int(lt[0]), int(lt[1])); rt = (int(rt[0]), int(rt[1]))
    span = rt[0] - lt[0]
    if span < 0.5 * W:
        raise ValueError('bow too small in the photo: stand closer, whole bow in the frame')
    left_reach, right_reach = seed[0] - lt[0], rt[0] - seed[0]
    if min(left_reach, right_reach) < 0.75 * max(left_reach, right_reach):
        raise ValueError('could not follow one limb to its tip: more light on the bow, or a darker board')
    if lt[0] < W * 0.01 or rt[0] > W * 0.99:
        raise ValueError('a bow tip is cut off at the edge of the photo')
    path = trace(rgb, lt, rt)
    cx = (lt[0] + rt[0]) / 2
    def at_x(x):
        j = int(np.argmin(np.abs(path[:, 0] - x)))
        return (int(path[j, 0]), int(path[j, 1]))
    lh, rh = at_x(cx - handle_frac * span), at_x(cx + handle_frac * span)
    return [lt, lh, rh, rt], {'seed': list(seed), 'span_px': int(span)}


def snap(rgb, p, r=25):
    """Move an anchor onto the limb centreline WITHOUT sliding it along the limb: take the nearest
    pixel that is solidly wood, then the deepest-inside pixel within 4 px of it."""
    from scipy import ndimage
    w = wood(rgb); x, y = p
    y0, x0 = max(0, y - r), max(0, x - r)
    win = w[y0:y + r + 1, x0:x + r + 1] > 0.5
    if not win.any():
        return (x, y)
    yy, xx = np.nonzero(win)
    i = np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2)
    ny, nx = yy[i] + y0, xx[i] + x0
    dt = ndimage.distance_transform_edt(w > 0.35)
    a0, b0 = max(0, ny - 4), max(0, nx - 4)
    sub = dt[a0:ny + 5, b0:nx + 5]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return (int(b0 + ix), int(a0 + iy))


def _trim(xy):
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    L = s[-1]
    return xy[(s >= TRIM_HANDLE * L) & (s <= (1 - TRIM_TIP) * L)]


def limbs(rgb, pts):
    lt, lh, rh, rt = (snap(rgb, p) for p in pts)
    return _trim(trace(rgb, lh, lt)), _trim(trace(rgb, rh, rt))   # each ordered handle end -> tip


def overlay(rgb, paths, bends, out):
    img = Image.fromarray(rgb).convert('RGB'); d = ImageDraw.Draw(img)
    w = max(4, rgb.shape[1] // 250)
    for xy, b in zip(paths, bends):
        L = len(xy); mx = max(float(b.max()), 1e-6)
        for i in range(N_BINS):
            seg = xy[int(i * L / N_BINS): int((i + 1) * L / N_BINS) + 1]
            r = float(b[i]) / mx
            d.line([tuple(p) for p in seg], fill=(int(255 * r), 40, int(255 * (1 - r))), width=w)
            m = seg[len(seg) // 2]
            d.text((m[0] + 2 * w, m[1] - 4 * w), str(i + 1), fill=(255, 255, 0))
    img.save(out)


def width_profile(rgb, handle_frac=None):
    """Front profile: limb width per station (pixels), stations from the fade to the tip like the bend
    stations (the handle, handle_frac of the span each side of centre, is left out)."""
    hf = HANDLE_FRAC if handle_frac is None else handle_frac
    m = wood(rgb) > 0.4
    cols = np.nonzero(m.sum(0) > 2)[0]
    if len(cols) < 20:
        return None
    x0, x1 = cols.min(), cols.max(); cx = (x0 + x1) / 2; span = x1 - x0
    wid = m.sum(0)
    res = {}
    for side, a, b in (('left', cx - hf * span, x0), ('right', cx + hf * span, x1)):
        e = np.linspace(a, b, N_BINS + 1)
        res[side] = [int(np.median(wid[int(min(p, q)):int(max(p, q)) + 1])) for p, q in zip(e[:-1], e[1:])]
    return res


JITTER = [(0, 0), (12, -10), (-12, 10)]    # re-run with anchors nudged; results must agree


def _profiles(rgb, pts, rest, deg):
    out = []
    for p, rp in zip(limbs(rgb, pts), rest):
        frac, ang, length, sm = bend_profile(p, deg=deg)
        b = stations(frac, ang, rp)
        out.append((b, length, sm, p))
    return out


def rest_follow(rr, pts, ntn, ppi=None):
    """Unbraced tip drop below the line through the two handle ends, all taken from the traced
    centrelines (handle end = start of each trimmed limb trace, tip = its end, 95% out), so a snapped
    anchor on the thick fade cannot tilt the reference line. Positive = string follow / set;
    negative = reflex."""
    L, R = limbs(rr, pts)
    lh, lt, rh, rt = L[0], L[-1], R[0], R[-1]
    u = (rh - lh) / np.linalg.norm(rh - lh); nrm = np.array([-u[1], u[0]])
    if nrm[1] < 0:
        nrm = -nrm                                  # normal pointing down the image
    span = bend_profile(L)[2] + bend_profile(R)[2] + np.linalg.norm(rh - lh)
    res = {}
    for side, tip, base in (('left', lt, lh), ('right', rt, rh)):
        drop = float(np.dot(tip - base, nrm))
        res[side + '_tip_drop_px'] = round(drop, 1)
        if ppi:
            res[side + '_tip_drop_in'] = round(drop / ppi, 2)
        elif ntn:
            res[side + '_tip_drop_in'] = round(float(drop * ntn / span), 2)
    k = 'tip_drop_in' if 'left_tip_drop_in' in res else 'tip_drop_px'
    res['mean_' + k] = round((res['left_' + k] + res['right_' + k]) / 2, 2)
    res['note'] = ('use the MEAN as the set/follow figure; the left/right split depends on a short '
                   'handle line and swings by ~+/-0.3 in, so do not read limb-to-limb set from it')
    return res


def grid_scale(rgb, grid_in):
    g = grid_pitch(rgb)
    out = {'pitch': g}
    if g['x_px'] and g['y_px']:
        out['x_over_y'] = round(g['x_px'] / g['y_px'], 3)
    if g['x_left_px'] and g['x_right_px']:
        out['left_over_right'] = round(g['x_left_px'] / g['x_right_px'], 3)
    vals = [v for v in (g['x_px'], g['y_px']) if v]
    if grid_in and vals:
        out['px_per_in'] = round(float(np.mean(vals)) / grid_in, 2)
    out['note'] = ('x_over_y and left_over_right should be within ~0.03 of 1.00 for a square-on camera; '
                   'the bow hangs in front of the board, so inches read slightly high (closer = bigger)')
    return out



# ---------------------------------------------------------------- scrape map
# Target bend per station as a fraction of the reference mid-limb bend (stations 4-7). Station 10 (nock
# end) is never judged. Elliptical = Santana's default for tapered limbs [DS]: inners bend a little
# less, outers at least as much as mids. Circular = pyramid / strong width taper.
TARGETS = {'elliptical': [0.75, 0.85, 0.9, 1, 1, 1, 1, 1.0, 1.0],
           'circular':   [0.85, 0.95, 1, 1, 1, 1, 1, 1, 0.95]}


GHOST_TIP = {'elliptical': 0.6, 'circular': 0.6}   # station 10 (nock end): stiff by design


def target_curve(xy, shape='elliptical', n=240):
    """The limb's TARGET shape for the ghost overlay: same start point, start direction, length and
    total bend as the measured limb, but the bend shared out per station by TARGETS[shape] (Santana's
    elliptical default: inners a little less, outers at least as much as mids). Returns (n,2) x,y."""
    seg = np.hypot(*np.diff(xy, axis=0).T); s = np.concatenate([[0], np.cumsum(seg)]); L = s[-1]
    t = np.linspace(0, L, n)
    x = np.interp(t, s, xy[:, 0]); y = np.interp(t, s, xy[:, 1])
    ang = np.unwrap(np.arctan2(np.diff(y), np.diff(x)))
    total = ang[-1] - ang[0]
    w = np.array(TARGETS[shape] + [GHOST_TIP[shape]], float)
    per = total * w / w.sum()                                    # bend per station
    f = (np.arange(n - 1) + 0.5) / (n - 1)
    st = np.minimum((f * N_BINS).astype(int), N_BINS - 1)
    dtheta = per[st] / (n - 1) * N_BINS                          # spread each station's bend evenly
    heading = ang[0] + np.concatenate([[0], np.cumsum(dtheta)[:-1]])
    step = L / (n - 1)
    pts = np.zeros((n, 2)); pts[0] = (x[0], y[0])
    pts[1:] = pts[0] + np.cumsum(np.stack([np.cos(heading), np.sin(heading)], 1) * step, 0)
    return pts


# ---------------------------------------------------------------- strain mode (beam inverse)
# At full draw a limb is a beam loaded only by the string: moment M(s) = F * h(s), h = distance to the
# string line. With curvature change k (drawn minus unbraced) and width w from the front photo:
#   EI/F = h/k,  thickness t ~ (EI/w)^(1/3),  belly strain e = k t / 2.
# Equal strain is the tiller goal ("fair share"; "tiller equalizes strain, not radius"). Validated in
# design/spike_fea/ (FINDINGS.md): ~4% median, ~14% worst-10% station error at 1 px trace noise.
STRAIN_LOW, STRAIN_VERY_LOW, STRAIN_HIGH = 0.85, 0.70, 1.15   # [tool] bands around even strain


def string_dir(rgb, tip, hook=None, toward_x=None, r0=25, r1=140):
    """Unit vector along the string from a tip: toward the draw hook when it is known, else the darkest
    straight ray from the tip aimed down and toward the bow's centre."""
    tip = np.asarray(tip, float)
    if hook is not None:
        d = np.asarray(hook, float) - tip; return d / np.linalg.norm(d)
    g = rgb.mean(2); best, bv = np.array([0.0, 1.0]), 1e9
    sgn = 1 if (toward_x or rgb.shape[1] / 2) > tip[0] else -1
    for a in np.radians(np.arange(5, 86, 1)):
        d = np.array([sgn * np.cos(a), np.sin(a)])
        pts = tip + np.outer(np.arange(r0, r1), d)
        xs, ys = pts[:, 0].astype(int), pts[:, 1].astype(int)
        ok = (xs >= 0) & (xs < rgb.shape[1]) & (ys >= 0) & (ys < rgb.shape[0])
        if ok.sum() < (r1 - r0) * 0.8:
            continue
        v = np.median(g[ys[ok], xs[ok]])
        if v < bv:
            bv, best = v, d
    return best


def strain_profile(xy_drawn, xy_rest, tip, sdir, width):
    """Per station (fade -> tip): curvature change (rad/px), moment arm (px), relative EI, relative
    thickness and relative strain (both scaled to mid-limb stations 4-7)."""
    fd, ad, Ld, sm = bend_profile(xy_drawn)
    e = np.linspace(0, 1, N_BINS + 1)
    th = np.interp(e, fd, ad)
    if xy_rest is not None:
        fr, ar, _, _ = bend_profile(xy_rest); th = th - np.interp(e, fr, ar)
    k = np.abs(np.radians(np.diff(th))) / (Ld / N_BINS)
    sl = np.linspace(0, Ld, len(sm))
    mid = np.stack([np.interp((np.arange(N_BINS) + 0.5) / N_BINS * Ld, sl, sm[:, c]) for c in (0, 1)], 1)
    nrm = np.array([-sdir[1], sdir[0]])
    h = np.abs((mid - np.asarray(tip, float)) @ nrm)
    EI = h / np.maximum(k, 1e-12)
    w = np.maximum(np.asarray(width, float), 1.0)
    t = (EI / w) ** (1 / 3)
    strain = k * t / 2
    r = lambda v: v / v[3:7].mean()
    return {'k_per_px': k, 'h_px': h, 'EI_rel': r(EI), 't_rel': r(t), 'strain_rel': r(strain), 't_raw': t, 'strain_raw': strain}


def strain_plan(rep, sd_key='strain'):
    """Scrape calls from strain, same format as scrape_plan: 'more' under STRAIN_VERY_LOW, 'light' under
    STRAIN_LOW, 'leave' over STRAIN_HIGH or on a hinge; stations 1 (fade) and 10 (tip) never called;
    unsure stations (runs disagree, or the trace left the wood) get no call."""
    plan = {}
    for sd in ('left', 'right'):
        st = rep[sd_key][sd]; calls = []
        hinge = set(rep[sd]['hinge_candidates_station']); gap = rep[sd].get('station_gap', [0] * N_BINS)
        for i in range(N_BINS):
            v, spread = st['strain_rel'][i], st['strain_spread'][i]
            if (i + 1) in hinge:
                c = 'leave'
            elif i in (0, N_BINS - 1) or gap[i] > GAP_MAX or spread > 0.12:
                c = 'ok'
            elif v < STRAIN_VERY_LOW:
                c = 'more'
            elif v < STRAIN_LOW:
                c = 'light'
            elif v > STRAIN_HIGH:
                c = 'leave'
            else:
                c = 'ok'
            calls.append(c)
        plan[sd] = {'station_call': calls,
                    'thickness_cut_pct': [round(float(100 * (1 - np.sqrt(min(1.0, v)))), 0) if calls[i] in ('more', 'light') else 0
                                          for i, v in enumerate(st['strain_rel'])]}
    plan['basis'] = 'strain: equal belly strain along each limb (beam model from the photos)'
    plan['shape'] = 'equal strain'
    return plan


def strain_block(rgb, rr, base_pts, rest_pts, widths, hook=None, ppi=None, caliper=None):
    """Strain per station for both limbs, averaged over the jitter runs. caliper = (side, inches from the
    string groove, thickness mm) turns relative strain into percent when the grid scale is known."""
    out = {}
    cx = (base_pts[0][0] + base_pts[3][0]) / 2
    for si, (side, tip_i) in enumerate((('left', 0), ('right', 3))):
        runs = []
        for dx, dy in JITTER:
            pts = [(x + dx * (1 if i % 2 else -1), y + dy) for i, (x, y) in enumerate(base_pts)]
            rp = [(x + dx * (1 if i % 2 else -1), y + dy // 3) for i, (x, y) in enumerate(rest_pts)] if rest_pts else None
            xyd = limbs(rgb, pts)[si]; xyr = limbs(rr, rp)[si] if rr is not None else None
            tip = np.asarray(base_pts[tip_i], float)
            sdir = string_dir(rgb, tip, hook=hook, toward_x=cx)
            runs.append(strain_profile(xyd, xyr, tip, sdir, widths[side]))
        S = np.array([r['strain_rel'] for r in runs]); T = np.array([r['t_rel'] for r in runs])
        blk = {'strain_rel': [round(float(v), 2) for v in S.mean(0)],
               'strain_spread': [round(float(v), 2) for v in S.std(0)],
               't_rel': [round(float(v), 2) for v in T.mean(0)],
               'width_px': list(widths[side])}
        if caliper and ppi and caliper[0] == side:
            # absolute: k per inch from the grid, thickness scaled so the calipered station matches
            r0 = runs[0]; L_in = bend_profile(limbs(rgb, base_pts)[si])[2] / ppi
            st = int(min(N_BINS - 1, max(0, round((1 - caliper[1] / L_in) * N_BINS - 0.5))))
            t_in = r0['t_raw'] * (caliper[2] / 25.4) / r0['t_raw'][st]
            eps = np.mean([r['k_per_px'] for r in runs], 0) * ppi * t_in / 2
            blk['strain_pct'] = [round(float(v * 100), 2) for v in eps]
            blk['thickness_in'] = [round(float(v), 3) for v in t_in]
        out[side] = blk
    return out


def scrape_plan(rep, shape, balance_ok, ppi):
    """Per station: 'more' / 'light' scrape, 'ok', or 'leave' (bending most or hinge: never scrape).
    Wood only comes off, so the reference is the BEST-bending limb's mids when the photo is square
    (equalises the limbs: the stiffer limb gets more marks); otherwise each limb is judged alone."""
    tgt = TARGETS[shape]
    deg = {sd: np.array(rep[sd]['station_pct']) * rep[sd]['total_bend_deg'] / 100 for sd in ('left', 'right')}
    mids = {sd: deg[sd][3:7].mean() for sd in deg}
    plan = {}
    for sd in ('left', 'right'):
        R = max(mids.values()) if balance_ok else mids[sd]
        d = deg[sd]; hinge = set(rep[sd]['hinge_candidates_station'])
        top = int(np.argmax(d[:9]))
        st, short_l = [], []
        for i in range(9):
            short = (tgt[i] * R - d[i]) / R
            short_l.append(round(float(100 * short)))
            shaky = rep[sd]['station_spread_pct'][i] > 1.5 or rep[sd].get('station_gap', [0] * N_BINS)[i] > GAP_MAX
            if (i + 1) in hinge:
                c = 'leave'
            elif shaky:
                c = 'ok'                             # runs disagree here: make no call
            elif i <= 1 and short > 0.10:
                c = 'ok'                             # inner 20% next to the fade: low bend is normal on a
                                                     # stiff handle, the reading leans on where the handle
                                                     # ends were placed, and scraping here risks a fade
                                                     # hinge. Santana: leave the inners, work mids and outers
            elif short > 0.25:
                c = 'more'
            elif short > 0.10:
                c = 'light'
            elif i == top or d[i] > 1.15 * tgt[i] * R:
                c = 'leave'
            else:
                c = 'ok'
            st.append(c)
        st.append('ok')                              # station 10: nock end, not judged
        L = rep[sd]['length_px']
        spans = []
        for i in range(N_BINS):
            a0, a1 = i / N_BINS, (i + 1) / N_BINS
            spans.append((a0 * L / ppi, a1 * L / ppi) if ppi else (a0 * 100, a1 * 100))
        plan[sd] = {'station_call': st, 'shortfall_pct_of_mid': short_l + [None], 'spans': spans}
    plan['units'] = 'in from the fade' if ppi else '% of the way from fade to nock'
    plan['basis'] = ('both limbs against the better-bending limb (equalises them)' if balance_ok
                     else 'each limb on its own shape only: camera not square enough to compare limbs; do a flip check')
    plan['shape'] = shape
    return plan


def plan_text(plan):
    out = [f"SCRAPE PLAN ({plan['shape']} target; {plan['basis']})"]
    name = {'more': 'SCRAPE MORE', 'light': 'scrape light', 'leave': 'LEAVE ALONE'}
    for sd in ('left', 'right'):
        out.append(f"{sd.upper()} limb (belly side):")
        calls = plan[sd]['station_call']; i = 0; any_ = False
        while i < N_BINS:
            j = i
            while j + 1 < N_BINS and calls[j + 1] == calls[i]:
                j += 1
            if calls[i] != 'ok':
                any_ = True
                a, b = plan[sd]['spans'][i][0], plan[sd]['spans'][j][1]
                stn = f"station {i + 1}" if i == j else f"stations {i + 1}-{j + 1}"
                out.append(f"  {name[calls[i]]}: {stn}, {a:.0f}-{b:.0f} {plan['units']}")
            i = j + 1
        if not any_:
            out.append('  nothing: this limb matches the target; leave it')
    return '\n'.join(out)


def draw_plan(rgb, paths, plan, out):
    """Coloured bands on the BELLY side of each drawn limb: dark green = scrape more, light green =
    scrape light, red X = leave alone; station numbers on the back side. Santana marks his bows the
    same way: scribbles where to work, X where to avoid [DS]."""
    from PIL import ImageFont
    base = Image.fromarray(rgb).convert('RGBA')
    lay = Image.new('RGBA', base.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    W = rgb.shape[1]; w = max(8, W // 110); off = int(1.6 * w); fs = max(14, W // 70)
    try:
        font = ImageFont.truetype('arialbd.ttf', fs)
    except OSError:
        font = ImageFont.load_default()
    col = {'more': (0, 140, 40, 235), 'light': (120, 230, 120, 215)}
    for sd, xy in zip(('left', 'right'), paths):
        calls = plan[sd]['station_call']; n = len(xy); mid_chord = (xy[0] + xy[-1]) / 2
        for i in range(N_BINS):
            seg = xy[int(i * n / N_BINS): int((i + 1) * n / N_BINS) + 1]
            m = seg[len(seg) // 2]
            tg = seg[-1] - seg[0]; nr = np.array([-tg[1], tg[0]]) / (np.linalg.norm(tg) + 1e-9)
            if np.dot(nr, mid_chord - m) < 0:      # belly = concave side = toward the chord
                nr = -nr
            c = calls[i]
            if c in col:
                d.line([tuple(p + nr * off) for p in seg], fill=col[c], width=w)
            elif c == 'leave':
                k = w * 1.3; q = m + nr * off
                d.line([tuple(q + (-k, -k)), tuple(q + (k, k))], fill=(220, 0, 0, 255), width=max(3, w // 2))
                d.line([tuple(q + (-k, k)), tuple(q + (k, -k))], fill=(220, 0, 0, 255), width=max(3, w // 2))
            t = m - nr * (off + w)
            d.text((t[0] - fs // 3, t[1] - fs // 2), str(i + 1), fill=(255, 255, 0, 255), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0, 255))
    lx, ly = int(W * 0.02), int(rgb.shape[0] * 0.03)
    d.rectangle([lx - 8, ly - 8, lx + fs * 13, ly + int(fs * 4.4)], fill=(0, 0, 0, 170))
    for k, (lab, c) in enumerate((('SCRAPE MORE (belly)', col['more']), ('scrape light', col['light']))):
        y = ly + int(k * fs * 1.4) + fs // 2
        d.line([(lx, y), (lx + fs * 2, y)], fill=c, width=w)
        d.text((lx + int(fs * 2.5), y - fs // 2), lab, fill=(255, 255, 255, 255), font=font)
    y = ly + int(2 * fs * 1.4) + fs // 2
    d.text((lx + int(fs * 0.5), y - fs // 2), 'X', fill=(220, 0, 0, 255), font=font)
    d.text((lx + int(fs * 2.5), y - fs // 2), 'LEAVE ALONE', fill=(255, 255, 255, 255), font=font)
    Image.alpha_composite(base, lay).convert('RGB').save(out)


def measure(a):
    os.makedirs(a.out, exist_ok=True)
    GRID['on'] = bool(a.grid); GRID['line_px'] = a.line_px
    rgb = load(a.drawn)
    rr = load(a.rest) if a.rest else None
    auto = {}
    if not a.drawn_pts or a.drawn_pts == 'auto':
        base_pts, auto['drawn'] = auto_anchors(rgb)
    else:
        base_pts = parse_pts(a.drawn_pts)
    rest_pts = None
    if rr is not None:
        if not a.rest_pts or a.rest_pts == 'auto':
            rest_pts, auto['rest'] = auto_anchors(rr)
        else:
            rest_pts = parse_pts(a.rest_pts)
    rep = {'stations': '10 equal lengths per limb; 1 = handle end (fade), 10 = tip',
           'bend_basis': 'drawn minus unbraced' if a.rest else 'drawn shape only (no unbraced photo)',
           'runs': len(JITTER)}
    if auto:
        rep['auto_anchors'] = {k: v for k, v in auto.items()}
        rep['anchors'] = {'drawn': base_pts, **({'rest': rest_pts} if rest_pts else {})}
    ppi = None
    if a.grid:
        rep['grid'] = grid_scale(rgb, a.grid_in)
        ppi = rep['grid'].get('px_per_in')
    smooth = {'left': [], 'right': []}; raw = {'left': [], 'right': []}; totals = {'left': [], 'right': []}
    hinge20 = {'left': [], 'right': []}
    keep = None
    for k, (dx, dy) in enumerate(JITTER):
        pts = [(x + dx * (1 if i % 2 else -1), y + dy) for i, (x, y) in enumerate(base_pts)]
        rest = [None, None]
        if rr is not None:
            rp = [(x + dx * (1 if i % 2 else -1), y + dy // 3) for i, (x, y) in enumerate(rest_pts)]
            rest = [bend_profile(p)[:2] for p in limbs(rr, rp)]
            if k == 0:
                rl = limbs(rr, rp)
                overlay(rr, [bend_profile(p)[3] for p in rl], [np.ones(N_BINS)] * 2, os.path.join(a.out, 'rest_overlay.png'))
        for side, (b, length, sm, p) in zip(('left', 'right'), _profiles(rgb, pts, rest, 'default')):
            smooth[side].append(100 * b / b.sum()); totals[side].append(float(b.sum()))
            if k == 0:
                keep = keep or {}; keep[side] = (b, length, sm, p)
        rest_raw = [bend_profile(p, deg=HINGE_DEG)[:2] for p in limbs(rr, rest_pts)] if rr is not None else [None, None]
        for side, (b, *_rest) in zip(('left', 'right'), _profiles(rgb, pts, rest_raw, HINGE_DEG)):
            raw[side].append(100 * b / b.sum())
        for side, p in zip(('left', 'right'), limbs(rgb, pts)):
            f, ang, _, _ = bend_profile(p, deg=HINGE_DEG)
            d20 = np.abs(np.diff(np.interp(np.linspace(0, 1, 21), f, ang)))
            hinge20[side].append(100 * d20 / max(d20.sum(), 1e-9))
    wd = wood(rgb)
    for side in ('left', 'right'):
        S = np.array(smooth[side]); Rw = np.array(raw[side]); m = S.mean(0)
        b, length, sm, p = keep[side]
        # share of each station's trace that is NOT on visible wood (glare, a gap, clutter): a
        # station the trace had to jump is unsure, so it can raise no hinge and gets no scrape call
        pi = np.clip(np.round(p).astype(int), 0, [rgb.shape[1] - 1, rgb.shape[0] - 1])
        off = wd[pi[:, 1], pi[:, 0]] < 0.2
        gap = [float(off[int(i * len(p) / N_BINS): int((i + 1) * len(p) / N_BINS)].mean()) for i in range(N_BINS)]
        # hinge = a bump of EXTRA bend over what the neighbouring limb predicts, in every run, from
        # two views of the same degree-8 profile: 10 stations (window vs stations 2 away, flag at
        # HINGE_EXCESS) or 20 half-stations (window vs 3 away, flag at HINGE_EXCESS_20). Calibration
        # (tests/hinge_calibration.py): 0 false alarms on 40 clean limbs and the sample bows; 10 deg
        # kinks caught ~80%, 15 deg ~98%. A backstop: the bowyer still looks along the limb.
        spikes = []
        prof20 = hinge20[side]
        for i in range(2, N_BINS - 2):
            if any(gap[j] > GAP_MAX for j in range(i - 1, i + 2)):
                continue
            ex = [r[i - 1] + r[i] + r[i + 1] - 3 * (r[i - 2] + r[i + 2]) / 2 for r in Rw]
            ex20 = [max(r[k - 1] + r[k] + r[k + 1] - 3 * (r[k - 3] + r[k + 3]) / 2
                        for k in range(max(3, 2 * i), min(17, 2 * i + 2))) for r in prof20]
            if min(ex) >= HINGE_EXCESS or min(ex20) >= HINGE_EXCESS_20:
                spikes.append(i + 1)
        mids = m[3:7].mean()
        rep[side] = {'length_px': round(length),
                     **({'length_in_traced': round(length / ppi, 1)} if ppi else {}),
                     # bend per station relative to the mid-limb (stations 4-7). Elliptical tiller wants
                     # outer >= ~1.0; inner well above 1.0 = inner limbs doing too much (most common fault)
                     'outer_vs_mid': round(float(m[7:9].mean() / mids), 2),
                     'inner_vs_mid': round(float(m[0:3].mean() / mids), 2),
                     'total_bend_deg': round(float(np.mean(totals[side])), 1),
                     'station_pct': [round(float(v), 1) for v in m],
                     'station_spread_pct': [round(float(v), 1) for v in S.std(0)],
                     'thirds_pct_inner_mid_outer': [round(float(m[:3].sum()), 1), round(float(m[3:7].sum()), 1), round(float(m[7:].sum()), 1)],
                     'hinge_candidates_station': spikes,
                     'station_gap': [round(g, 2) for g in gap],
                     'raw_station_pct': [[round(float(v), 1) for v in r] for r in Rw],
                     'tip_xy': [round(float(v)) for v in p[-1]]}
    paths = [keep['left'][2], keep['right'][2]]
    bends = [np.array(rep['left']['station_pct']), np.array(rep['right']['station_pct'])]
    rep['balance'] = {'left_over_right_bend': round(rep['left']['total_bend_deg'] / max(rep['right']['total_bend_deg'], 1e-6), 2),
                      'left_over_right_length': round(rep['left']['length_px'] / max(rep['right']['length_px'], 1), 2),
                      'note': 'length ratio far from 1.0 means the camera was not square or the bow is off-centre; treat balance with care'}
    if rr is not None:
        rppi = None
        if a.grid:
            rep['grid_unbraced'] = grid_scale(rr, a.grid_in)
            rppi = rep['grid_unbraced'].get('px_per_in')
        rep['unbraced'] = rest_follow(rr, rest_pts, a.ntn, rppi)
    if a.front:
        fr = load(a.front)
        rep['front_width_px'] = width_profile(fr)
        if a.grid and a.grid_in:
            fp = grid_scale(fr, a.grid_in).get('px_per_in')
            if fp and rep['front_width_px']:
                rep['front_width_in'] = {k: [round(v / fp, 2) for v in w] for k, w in rep['front_width_px'].items()}
    overlay(rgb, paths, bends, os.path.join(a.out, 'drawn_overlay.png'))
    bal_ok = 0.95 <= rep['balance']['left_over_right_length'] <= 1.05
    if rep.get('grid'):
        g = rep['grid']
        bal_ok = bal_ok and all(0.97 <= g.get(k, 1) <= 1.03 for k in ('x_over_y', 'left_over_right'))
    rep['scrape_plan_rules'] = scrape_plan(rep, a.shape, bal_ok, ppi)
    rep['scrape_plan'] = rep['scrape_plan_rules']
    if rep.get('front_width_px') and rr is not None:
        hook = None
        if ppi and getattr(a, 'draw_in', None):
            lh, rh = np.asarray(base_pts[1], float), np.asarray(base_pts[2], float)
            c = (lh + rh) / 2; u = (rh - lh) / np.linalg.norm(rh - lh); nrm = np.array([-u[1], u[0]])
            if np.dot(nrm, (np.asarray(base_pts[0], float) + np.asarray(base_pts[3], float)) / 2 - c) < 0:
                nrm = -nrm                                     # toward the tips: the string side
            hook = c + nrm * a.draw_in * ppi
        cal = None
        if getattr(a, 'caliper', None):
            sd, inch, mm = a.caliper.split(','); cal = (sd, float(inch), float(mm))
        rep['strain'] = strain_block(rgb, rr, base_pts, rest_pts, rep['front_width_px'], hook=hook, ppi=ppi, caliper=cal)
        rep['scrape_plan_strain'] = strain_plan(rep)
        rep['scrape_plan_strain']['units'] = rep['scrape_plan_rules']['units']
        for sd in ('left', 'right'):
            rep['scrape_plan_strain'][sd]['spans'] = rep['scrape_plan_rules'][sd]['spans']
            rep['scrape_plan_strain'][sd]['shortfall_pct_of_mid'] = [None] * N_BINS
        if getattr(a, 'mode', 'rules') == 'strain':
            rep['scrape_plan'] = rep['scrape_plan_strain']
    elif getattr(a, 'mode', 'rules') == 'strain':
        raise ValueError('strain mode needs the unbraced photo and a front photo')
    draw_plan(rgb, paths, rep['scrape_plan'], os.path.join(a.out, 'scrape_map.png'))
    json.dump(rep, open(os.path.join(a.out, 'report.json'), 'w'), indent=1)
    lines = [f"basis: {rep['bend_basis']}"]
    if rep.get('grid'):
        lines.append(f"grid (drawn panel): {rep['grid']}")
    for side in ('left', 'right'):
        r = rep[side]
        lines.append(f"{side.upper()}: total {r['total_bend_deg']} deg, length {r['length_px']} px{(' = ' + str(r['length_in_traced']) + ' in traced') if 'length_in_traced' in r else ''}, thirds inner/mid/outer % {r['thirds_pct_inner_mid_outer']}, outer_vs_mid {r['outer_vs_mid']}, inner_vs_mid {r['inner_vs_mid']}")
        lines.append('   stations % ' + ' '.join(f"{i + 1}:{v}" for i, v in enumerate(r['station_pct'])))
        lines.append('   spread +/- ' + ' '.join(f"{v}" for v in r['station_spread_pct']) + f"   hinge candidates: {r['hinge_candidates_station'] or 'none'}")
    lines.append(f"balance: left/right total bend {rep['balance']['left_over_right_bend']}, left/right limb length {rep['balance']['left_over_right_length']}")
    if rep.get('unbraced'):
        lines.append(f"unbraced tip drop (+ = follow/set, - = reflex): {rep['unbraced']}")
    if rep.get('front_width_in'):
        lines.append(f"front width in (handle->tip): {rep['front_width_in']}")
    if rep.get('front_width_px'):
        lines.append(f"front width px (handle->tip): {rep['front_width_px']}")
    if rep.get('strain'):
        for sd in ('left', 'right'):
            b = rep['strain'][sd]
            lines.append(f"STRAIN {sd.upper()} (vs mid-limb): " + ' '.join(f"{i + 1}:{v}" for i, v in enumerate(b['strain_rel'])))
            lines.append(f"   thickness vs mid: " + ' '.join(f"{v}" for v in b['t_rel']) + (f"   strain %: {b['strain_pct']}" if 'strain_pct' in b else ''))
        lines.append(plan_text(rep['scrape_plan_strain']))
    lines.append(plan_text(rep['scrape_plan']) if rep['scrape_plan'] is rep['scrape_plan_rules'] else '(plan above drives the result: strain mode)')
    txt = '\n'.join(lines)
    open(os.path.join(a.out, 'report.txt'), 'w').write(txt + '\n')
    print(txt)


def grid(a):
    im = Image.open(a.img).convert('RGB'); d = ImageDraw.Draw(im)
    W, H = im.size; st = a.step
    for x in range(0, W, st):
        d.line([(x, 0), (x, H)], fill=(0, 255, 255) if x % (st * 5) else (255, 0, 255), width=2)
        d.text((x + 3, 3), str(x), fill=(255, 255, 0))
    for y in range(0, H, st):
        d.line([(0, y), (W, y)], fill=(0, 255, 255) if y % (st * 5) else (255, 0, 255), width=2)
        d.text((3, y + 3), str(y), fill=(255, 255, 0))
    out = os.path.splitext(a.img)[0] + '_grid.png'
    scale = min(1.0, 1400 / W)
    im.resize((int(W * scale), int(H * scale))).save(out)
    print(out, 'scale', round(scale, 3), '(labels are original pixels)')


def crop(a):
    x0, y0, x1, y1 = (int(v) for v in a.box.split(','))
    Image.open(a.img).convert('RGB').crop((x0, y0, x1, y1)).save(a.out); print(a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    g = sp.add_parser('grid'); g.add_argument('img'); g.add_argument('--step', type=int, default=100)
    c = sp.add_parser('crop'); c.add_argument('img'); c.add_argument('--box', required=True); c.add_argument('--out', required=True)
    m = sp.add_parser('measure')
    m.add_argument('--drawn', required=True); m.add_argument('--drawn-pts', default='auto')
    m.add_argument('--rest'); m.add_argument('--rest-pts', default='auto'); m.add_argument('--front'); m.add_argument('--out', default='tiller_out')
    m.add_argument('--ntn', type=float, help='nock-to-nock length in inches, to convert tip drop to inches')
    m.add_argument('--shape', choices=['elliptical', 'circular'], default='elliptical', help='target: elliptical (Santana default, parallel-then-taper limbs) or circular (pyramid / strong width taper)')
    m.add_argument('--mode', choices=['rules', 'strain'], default='rules', help='which scrape plan drives the result; strain needs --rest and --front')
    m.add_argument('--draw-in', type=float, help='draw length on the tree (in), aims the string at the draw hook for strain mode')
    m.add_argument('--caliper', help='side,inches_from_tip,thickness_mm (e.g. left,8,11.5): strain in percent (needs --grid-in)')
    m.add_argument('--grid', action='store_true', help='a gridded backdrop is behind the bow')
    m.add_argument('--grid-in', type=float, help='size of one grid square in inches (gives px per inch)')
    m.add_argument('--line-px', type=int, default=3, help='opening size (px) that cleans anti-aliased line edges')
    a = ap.parse_args()
    {'grid': grid, 'crop': crop, 'measure': measure}[a.cmd](a)
