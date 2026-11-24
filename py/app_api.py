"""The phone app's single entry point: two photos in, the result screen's data out.

analyze(drawn_bytes, rest_bytes, grid_in, target_wt=None, today_draw=None, shape='elliptical')
returns a JSON-able dict:
  ok        False with `error` (one plain retake line) when the photo cannot be trusted
  stop      True when a hinge is suspected (the app shows STOP and no scrape steps)
  steps     [{limb, first, pencil: [near, far], boundaries: [..], zones: [{call, near, far, strokes}]}]
            inches from the string groove at the tip; first = the stiffer limb, worked first
  balance   {stiff: 'left'|'right'|None, pct}
  set_in    mean unbraced tip drop in inches (string follow + set)
  photos    {drawn, ghost, rest, stop}: JPEG data URLs, marked per design/IMAGE_SPEC.md
  report    the full tiller_check report, for history and debugging
Runs the same code under CPython and Pyodide (no files outside a temp folder, no network).
"""
import base64, io, json, os, tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tiller_check as tc

MAX_W = 1600                              # phones shoot 4000 px; 1600 keeps measuring fast enough
GREEN = {'more': (0, 140, 40), 'light': (120, 220, 120)}
YEL, BLACK, WHITE, GHOST = (255, 210, 0), (0, 0, 0), (255, 255, 255), (240, 214, 90)
STROKES = {'normal': {'more': 20, 'light': 10}, 'stiff': {'more': 25, 'light': 12}}
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'BigShouldersText-wght.ttf')


def _font(px):
    try:
        f = ImageFont.truetype(FONT_PATH, int(px))
        try:
            f.set_variation_by_name('ExtraBold')
        except Exception:
            pass
        return f
    except OSError:
        pass
    for f in ('arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, int(px))
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=int(px))
    except TypeError:
        return ImageFont.load_default()


def _open(b):
    im = Image.open(io.BytesIO(b))
    try:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(im)         # phone photos carry their rotation in EXIF
    except Exception:
        pass
    im = im.convert('RGB')
    if im.width < im.height:
        raise ValueError('hold the phone sideways and retake')
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    return im


def _url(im, q=82):
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=q)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


class _Limbs:
    """Geometry helpers for one photo: limb paths, inch positions from the string groove."""

    def __init__(self, rgb, pts, rep, ppi):
        self.rgb, self.rep, self.ppi = rgb, rep, ppi
        self.paths = dict(zip(('left', 'right'), (tc.bend_profile(p)[3] for p in tc.limbs(rgb, pts))))

    def tip_offset(self, sd):
        U = self.rep[sd]['length_px'] / (1 - tc.TRIM_HANDLE - tc.TRIM_TIP)
        return tc.TRIM_TIP * U + 1.0 * self.ppi

    def length_in(self, sd):
        xy = self.paths[sd]; L = np.hypot(*np.diff(xy, axis=0).T).sum()
        return (L + self.tip_offset(sd)) / self.ppi

    def at_inch(self, sd, inch):
        xy = self.paths[sd]; seg = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]); L = seg[-1]
        s = L + self.tip_offset(sd) - inch * self.ppi
        if s >= L:
            t = xy[-1] - xy[-6]; t /= np.linalg.norm(t); return xy[-1] + t * (s - L), t
        s = max(s, 0); i = min(np.searchsorted(seg, s), len(xy) - 2)
        t = xy[i + 1] - xy[max(i - 1, 0)]; t /= np.linalg.norm(t)
        return np.array([np.interp(s, seg, xy[:, 0]), np.interp(s, seg, xy[:, 1])]), t

    def inward(self, sd, p, t):
        xy = self.paths[sd]; mc = (xy[0] + xy[-1]) / 2
        nr = np.array([-t[1], t[0]])
        return nr if np.dot(nr, mc - p) >= 0 else -nr       # toward the string: the side to scrape


def _zones(rep, sd, length_in):
    """Scrape runs (call, near, far) in whole inches from the string groove, nearest first. Station
    i covers (i/10..(i+1)/10) of the limb from the fade; the tip end is length_in from the groove."""
    calls = rep['scrape_plan'][sd]['station_call']; out = []; i = 0
    while i < 10:
        j = i
        while j + 1 < 10 and calls[j + 1] == calls[i]:
            j += 1
        if calls[i] in GREEN:
            near = length_in * (1 - (j + 1) / 10) + tc.TRIM_TIP * length_in
            far = length_in * (1 - i / 10) + tc.TRIM_TIP * length_in
            out.append((calls[i], int(round(near)), int(round(far))))
        i = j + 1
    return sorted(out, key=lambda z: z[1])


def _draw(im, geo, plan_zones, order, stiff, ghost=False, marks=True):
    """The marked photo: green bands on the string side, tape numbers every 6 in from the tip,
    yellow 0 in dot, numbered circles, START HERE under the first limb, optional dashed target."""
    im = im.copy(); d = ImageDraw.Draw(im); W = im.width
    w = max(6, W // 90); off = int(1.5 * w)
    for sd in ('left', 'right'):
        if marks:
            for c, a, b in plan_zones[sd]:
                p, t = geo.at_inch(sd, (a + b) / 2); n = geo.inward(sd, p, t)
                seg = [geo.at_inch(sd, x / 4)[0] for x in range(4 * a, 4 * b + 1)]
                d.line([tuple(q + n * off) for q in seg], fill=GREEN[c], width=w)
        for inch in range(0, int(geo.length_in(sd)) + 1, 6):
            p, t = geo.at_inch(sd, inch); n = geo.inward(sd, p, t)
            d.line([tuple(p - n * w * 0.6), tuple(p - n * w * 1.6)], fill=WHITE, width=max(2, w // 3))
            d.text(tuple(p - n * w * 3.0), str(inch), font=_font(1.9 * w), fill=WHITE, anchor='mm', stroke_width=max(2, w // 3), stroke_fill=BLACK)
        if marks and sd in order and plan_zones[sd]:
            z = plan_zones[sd]; p, t = geo.at_inch(sd, (z[0][1] + z[-1][2]) / 2)
            c = p + geo.inward(sd, p, t) * (off + 3.4 * w); r = 2.4 * w
            d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=YEL, outline=BLACK, width=max(2, w // 3))
            d.text(tuple(c), str(order.index(sd) + 1), font=_font(r * 1.3), fill=BLACK, anchor='mm')
            if sd == stiff:
                d.text((c[0], c[1] + 4.2 * w), 'START HERE', font=_font(2.6 * w), fill=YEL, anchor='mm', stroke_width=max(2, w // 2), stroke_fill=BLACK)
        p0, _ = geo.at_inch(sd, 0)
        d.ellipse([p0[0] - w, p0[1] - w, p0[0] + w, p0[1] + w], fill=YEL, outline=BLACK, width=max(2, w // 4))
        if ghost:
            g = tc.target_curve(geo.paths[sd])
            for k in range(0, len(g) - 4, 8):
                d.line([tuple(g[k]), tuple(g[k + 4])], fill=GHOST, width=max(3, w // 2))
    ys = [p[1] for xy in geo.paths.values() for p in xy]
    top = max(0, int(min(ys)) - 6 * w); bot = min(im.height, int(max(ys)) + 8 * w)
    return im.crop((0, top, W, bot))


def analyze(drawn_bytes, rest_bytes, grid_in, target_wt=None, today_draw=None, shape='elliptical'):
    tc._CACHE.clear(); tc._TRACE.clear()
    try:
        drawn, rest = _open(drawn_bytes), _open(rest_bytes)
    except ValueError as e:
        return {'ok': False, 'error': str(e).capitalize() + '.'}
    with tempfile.TemporaryDirectory() as tmp:
        dp, rp = os.path.join(tmp, 'drawn.png'), os.path.join(tmp, 'rest.png')
        drawn.save(dp); rest.save(rp)
        a = SimpleNamespace(drawn=dp, drawn_pts='auto', rest=rp, rest_pts='auto', front=None, out=tmp,
                            ntn=None, grid=True, grid_in=float(grid_in), line_px=3, shape=shape)
        try:
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                tc.measure(a)
        except ValueError as e:
            return {'ok': False, 'error': str(e)[0].upper() + str(e)[1:] + '.'}
        rep = json.load(open(os.path.join(tmp, 'report.json')))
    g = rep.get('grid', {})
    ppi = g.get('px_per_in')
    if not ppi:
        return {'ok': False, 'error': 'Grid board not found. Whole board behind the bow, in focus, and retake.'}
    if abs(g.get('x_over_y', 1) - 1) > 0.05 or abs(g.get('left_over_right', 1) - 1) > 0.05:
        return {'ok': False, 'error': 'Camera tilted. Stand square to the grid and retake.'}

    geo = _Limbs(np.asarray(drawn), [tuple(p) for p in rep['anchors']['drawn']], rep, ppi)
    lengths = {sd: geo.length_in(sd) for sd in ('left', 'right')}
    plan_zones = {sd: _zones(rep, sd, lengths[sd]) for sd in ('left', 'right')}
    lr = rep['balance']['left_over_right_bend']
    pct = round(abs(1 - lr) * 100)
    balance_ok = rep['scrape_plan']['basis'].startswith('both limbs')
    stiff = ('left' if lr < 1 else 'right') if (pct > 4 and balance_ok) else None
    has = [sd for sd in ('left', 'right') if plan_zones[sd]]
    order = sorted(has, key=lambda sd: 0 if sd == stiff else 1)
    hinge = any(rep[sd]['hinge_candidates_station'] for sd in ('left', 'right'))

    steps = []
    for sd in order:
        s = STROKES['stiff' if sd == stiff else 'normal']
        z = plan_zones[sd]
        steps.append({'limb': sd, 'first': sd == stiff,
                      'pencil': [z[0][1], z[-1][2]],
                      'boundaries': sorted({v for _, a, b in z for v in (a, b)} - {z[0][1], z[-1][2]}),
                      'zones': [{'call': c, 'near': a, 'far': b, 'strokes': s[c]} for c, a, b in z]})

    rest_geo = _Limbs(np.asarray(rest), [tuple(p) for p in rep['anchors']['rest']], rep,
                      rep.get('grid_unbraced', {}).get('px_per_in') or ppi)
    rest_zones = {sd: _zones(rep, sd, rest_geo.length_in(sd)) for sd in ('left', 'right')}
    photos = {'drawn': _url(_draw(drawn, geo, plan_zones, order, stiff)),
              'ghost': _url(_draw(drawn, geo, plan_zones, order, stiff, ghost=True)),
              'rest': _url(_draw(rest, rest_geo, rest_zones, order, None)),
              'stop': _url(_draw(drawn, geo, plan_zones, order, stiff, marks=False))}
    return {'ok': True, 'stop': hinge, 'steps': [] if hinge else steps,
            'balance': {'stiff': stiff, 'pct': pct},
            'set_in': rep.get('unbraced', {}).get('mean_tip_drop_in'),
            'today_draw': today_draw, 'target_wt': target_wt,
            'photos': photos, 'report': rep}


if __name__ == '__main__':
    import sys
    d, r, gi = sys.argv[1], sys.argv[2], float(sys.argv[3])
    res = analyze(open(d, 'rb').read(), open(r, 'rb').read(), gi)
    out = sys.argv[4] if len(sys.argv) > 4 else 'app_api_out'
    os.makedirs(out, exist_ok=True)
    for k, v in res.pop('photos', {}).items():
        open(os.path.join(out, k + '.jpg'), 'wb').write(base64.b64decode(v.split(',', 1)[1]))
    res.pop('report', None)
    print(json.dumps(res, indent=1))
