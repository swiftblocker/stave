// Measuring worker: runs tools/tiller_check.py + tools/app_api.py in Pyodide, off the main thread.
// Messages in:  {type: 'warm'}  load Python and packages ahead of time
//               {type: 'analyze', id, drawn: ArrayBuffer, rest: ArrayBuffer, gridIn, targetWt, todayDraw, shape}
// Messages out: {type: 'ready'} | {type: 'result', id, data} | {type: 'error', id, message}
const PYODIDE = new URL('vendor/pyodide/', self.location.href).href;   // self-hosted Pyodide 0.28.3
let ready = null;

function boot() {
  if (ready) return ready;
  ready = (async () => {
    importScripts(PYODIDE + 'pyodide.js');
    const py = await loadPyodide({ indexURL: PYODIDE });
    await py.loadPackage(['numpy', 'scipy', 'pillow']);
    for (const f of ['tiller_check.py', 'app_api.py']) {
      const src = await (await fetch('py/' + f)).text();
      py.FS.writeFile('/home/pyodide/' + f, src);
    }
    py.FS.mkdirTree('/home/pyodide/fonts');
    py.FS.writeFile('/home/pyodide/fonts/BigShouldersText-wght.ttf', new Uint8Array(await (await fetch('py/fonts/BigShouldersText-wght.ttf')).arrayBuffer()));
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide'); import app_api, json");
    self.postMessage({ type: 'ready' });
    return py;
  })();
  return ready;
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'warm') { boot().catch(err => self.postMessage({ type: 'error', message: String(err) })); return; }
  if (m.type !== 'analyze') return;
  try {
    const py = await boot();
    py.globals.set('drawn_b', new Uint8Array(m.drawn));
    py.globals.set('rest_b', new Uint8Array(m.rest));
    py.globals.set('args', py.toPy({ grid_in: m.gridIn, target_wt: m.targetWt, today_draw: m.todayDraw, shape: m.shape || 'elliptical' }));
    const out = py.runPython(`
res = app_api.analyze(bytes(drawn_b.to_py()), bytes(rest_b.to_py()), args['grid_in'],
                      target_wt=args['target_wt'], today_draw=args['today_draw'], shape=args['shape'])
res.pop('report', None)
json.dumps(res)
`);
    self.postMessage({ type: 'result', id: m.id, data: JSON.parse(out) });
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
