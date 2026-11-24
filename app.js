// Stave: screen routing, forms, set check, photo flow, live measuring (worker.js + Pyodide) and the
// result screen built from the measured data. Demo URL flags (no measuring): ?stop=1 hinge stop,
// ?bad=1 retake, ?empty=1 no bows, ?set=1 set warning, ?measuring=1 measuring, ?filled=1 inputs.
(() => {
  const q = new URLSearchParams(location.search);
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const screens = $$('[data-screen]'), result = $('#result'), again = $('#again');
  const num = v => { const n = parseFloat(String(v).replace(',', '.')); return Number.isFinite(n) ? n : NaN; };
  const round1 = n => Math.round(n * 10) / 10;
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  let measureTimer = 0;
  // the two result photos are created here so the page never ships an image with no source
  [['#shot-drawn', 'marked', 'The bow drawn on the tree, with the scrape zones marked'], ['#shot-rest', '', 'The unstrung bow with the same scrape zones marked']]
    .forEach(([sel, cls, alt]) => { const im = el('img', cls || null); im.alt = alt; im.width = 1200; im.height = 336; $(sel).append(im); });


  // ---- storage: IndexedDB on the phone only. bows {id, name, wt, dr, grid, stop, last, summary, thumb}
  //      checks {id, bow, at, scaleWt, draw, setBefore, setAfter, data}
  const DB = new Promise((resolve, reject) => {
    if (!window.indexedDB) { resolve(null); return; }
    const r = indexedDB.open('stave', 1);
    r.onupgradeneeded = () => { const db = r.result; db.createObjectStore('bows', { keyPath: 'id' }); db.createObjectStore('checks', { keyPath: 'id' }); };
    r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error);
  });
  const tx = async (store, mode, fn) => {
    const db = await DB; if (!db) return null;
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, mode), st = t.objectStore(store); let out;
      const req = fn(st); if (req) req.onsuccess = () => { out = req.result; };
      t.oncomplete = () => resolve(out); t.onerror = () => reject(t.error);
    });
  };
  const put = (store, v) => tx(store, 'readwrite', st => st.put(v));
  const get = (store, id) => tx(store, 'readonly', st => st.get(id));
  const all = (store) => tx(store, 'readonly', st => st.getAll());
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  const persist = () => navigator.storage && navigator.storage.persist && navigator.storage.persist().catch(() => {});

  // ---- routing: unknown hash falls back to the bow list; focus the new screen's heading
  const resetSteps = () => {
    $$('.step').forEach(s => s.classList.remove('is-done'));
    $$('.done').forEach(b => b.setAttribute('aria-pressed', 'false'));
    if (!result.classList.contains('is-stop')) again.hidden = true;
  };
  const show = () => {
    let id = (location.hash || '#bows').slice(1);
    if (!screens.some(s => s.id === id)) { id = 'bows'; history.replaceState(null, '', '#bows'); }
    if (id !== 'photo2') { clearTimeout(measureTimer); $('.measuring').hidden = true; $('#photo2 .shoot').hidden = false; }
    if (id === 'result') resetSteps();
    if (id === 'check' || id === 'photo1') warm();
    if (id === 'bows') listBows();
    screens.forEach(s => s.classList.toggle('is-on', s.id === id));
    window.scrollTo(0, 0);
    const focusEl = id === 'result' && result.classList.contains('is-stop') ? $('#stopmsg') : $(`#${id} h1`);
    if (focusEl) { focusEl.setAttribute('tabindex', '-1'); focusEl.focus({ preventScroll: true }); }
  };
  addEventListener('hashchange', show);

  // ---- forms: the forward button stays disabled until every field holds a sane value
  const valid = inp => {
    if (!inp.value.trim()) return false;
    if (!inp.dataset.num) return true;
    const [lo, hi] = inp.dataset.num.split('-').map(Number), n = num(inp.value);
    return n >= lo && n <= hi;
  };
  $$('[data-form]').forEach(form => {
    const go = form.querySelector('.go'), fields = [...form.querySelectorAll('input[required]')];
    const check = () => go.setAttribute('aria-disabled', fields.every(valid) ? 'false' : 'true');
    form.addEventListener('input', check);
    form.addEventListener('submit', e => e.preventDefault());
    go.addEventListener('click', e => { if (go.getAttribute('aria-disabled') === 'true') e.preventDefault(); });
    form._check = check;
  });

  // ---- the bow and today's numbers (persisted per bow in the storage step)
  let bow = { id: null, name: 'Hickory #1', wt: 50, dr: 28, grid: 2, stop: false, last: null };
  const useBow = (b) => {
    bow = b; localStorage.setItem('stave.current', b.id || '');
    $$('#check h1, #result h1').forEach(h => { h.textContent = bow.name; }); caps();
  };
  $('#new .go').addEventListener('click', async (e) => {
    if ($('#new .go').getAttribute('aria-disabled') === 'true') return;
    e.preventDefault();
    const b = { id: uid(), name: $('#f-name').value.trim(), wt: num($('#f-wt').value), dr: num($('#f-dr').value),
                grid: num($('#f-grid').value), stop: false, last: null, summary: '', thumb: '' };
    await put('bows', b); persist(); useBow(b);
    $$('#new input').forEach(i => { i.value = ''; }); $('#new [data-form]')._check();
    location.hash = '#check';
  });
  // delete a bow with every saved check (photos too)
  const deleteBow = async (id) => {
    const checks = ((await all('checks')) || []).filter(c => c.bow === id);
    await tx('checks', 'readwrite', st => { checks.forEach(c => st.delete(c.id)); });
    await tx('bows', 'readwrite', st => st.delete(id));
    if (bow.id === id) { localStorage.removeItem('stave.current'); bow = { id: null, name: '', wt: 50, dr: 28, grid: 2, stop: false, last: null }; }
  };

  // swipe left on a bow row: the row slides aside and shows Delete; one row open at a time
  const SWIPE = 96;
  const closeRows = (except) => $$('.bows li.open').forEach(li => { if (li !== except) li.classList.remove('open'); });
  const swipeable = (li, row) => {
    let x0 = 0, y0 = 0, dx = 0, mode = null, id = null;
    row.addEventListener('pointerdown', e => {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      x0 = e.clientX; y0 = e.clientY; dx = 0; mode = null; id = e.pointerId;
    });
    row.addEventListener('pointermove', e => {
      if (e.pointerId !== id) return;
      const mx = e.clientX - x0, my = e.clientY - y0;
      if (!mode) { if (Math.abs(mx) > 8 && Math.abs(mx) > Math.abs(my)) { mode = 'swipe'; row.setPointerCapture(id); li.classList.add('dragging'); closeRows(li); } else if (Math.abs(my) > 8) mode = 'scroll'; }
      if (mode !== 'swipe') return;
      const base = li.classList.contains('open') ? -SWIPE : 0;
      dx = Math.max(-SWIPE * 1.25, Math.min(0, base + mx));
      row.style.transform = `translateX(${dx}px)`;
    });
    const end = e => {
      if (e.pointerId !== id) return;
      id = null;
      if (mode === 'swipe') {
        li.classList.remove('dragging'); row.style.transform = '';
        li.classList.toggle('open', dx < -SWIPE / 2);
        row.dataset.swiped = '1'; setTimeout(() => { delete row.dataset.swiped; }, 0);
      }
    };
    row.addEventListener('pointerup', end); row.addEventListener('pointercancel', end);
  };
  document.addEventListener('pointerdown', e => { if (!e.target.closest('.bows li.open')) closeRows(); });

  const listBows = async () => {
    const bows = ((await all('bows')) || []).sort((a, b) => (b.updated || 0) - (a.updated || 0));
    if (q.has('empty') || q.has('demo') && !bows.length) return;
    const ul = $('.bows'); ul.replaceChildren();
    ul.hidden = !bows.length; $('.empty').hidden = !!bows.length;
    for (const b of bows) {
      const a = el('a', 'bow-row'); a.href = '#'; a.dataset.id = b.id;
      const img = el('img'); img.alt = ''; img.width = 96; img.height = 56; img.src = b.thumb || 'icons/icon-192.png';
      const txt = el('span'); txt.append(el('b', null, b.name),
        b.stop ? el('span', 'stopped', 'Stopped: may be hinging') : el('span', null, b.summary || 'No checks yet'));
      a.append(img, txt);
      const li = el('li');
      const del = el('button', 'swipe-del', 'Delete'); del.type = 'button'; del.setAttribute('aria-label', `Delete ${b.name}`);
      del.addEventListener('click', async () => { await deleteBow(b.id); listBows(); });
      a.addEventListener('click', async (ev) => {
        ev.preventDefault();
        if (a.dataset.swiped) return;                       // the end of a swipe is not a tap
        if (li.classList.contains('open')) { li.classList.remove('open'); return; }
        useBow(b);
        const last = b.last && await get('checks', b.last);
        if (last) { render(last.data); location.hash = '#result'; } else location.hash = '#check';
      });
      li.append(a, del); swipeable(li, a); ul.append(li);
    }
  };
  const today = () => num($('#c-dr').value) || 20;
  const caps = () => {
    $$('.cap-dr').forEach(e => { e.textContent = String(round1(today())); });
    $$('.cap-wt').forEach(e => { e.textContent = String(round1(bow.wt)); });
    $('#t-wt').textContent = String(round1(bow.wt)); $('#t-dr').textContent = String(round1(bow.dr));
  };
  $('#c-dr').addEventListener('input', caps);

  // ---- set check: weight at the same draw before and after 20 pulls; a drop of 1 lb or more is set
  const warn = $('#setwarn');
  const setCheck = () => {
    const b = num($('#s-before').value), a = num($('#s-after').value);
    const drop = round1(b - a);
    if (!(b >= 1 && b <= 150 && a >= 1 && a <= 150) || drop < 1) { warn.classList.remove('on'); return; }
    const next = round1(bow.wt - drop);
    warn.querySelector('.v-drop').textContent = `${drop} lb`;
    warn.querySelectorAll('.v-new').forEach(e => { e.textContent = `${next} lb`; });
    warn.classList.add('on'); warn.dataset.next = String(next);
  };
  $('#s-before').addEventListener('change', setCheck); $('#s-after').addEventListener('change', setCheck);
  $('#use-new').addEventListener('click', () => {
    bow.wt = num(warn.dataset.next); caps();
    $('#s-before').value = ''; $('#s-after').value = ''; warn.classList.remove('on');
    $('#check [data-form]')._check();
  });

  // ---- steps: Done strikes the step through with a pencil line; New photos appears when all are done
  const pencil = '<svg class="strike" viewBox="0 0 300 14" preserveAspectRatio="none" aria-hidden="true"><path d="M2 8.6 C60 7.4 140 6.2 210 5.6 C250 5.2 280 4.4 297 3.2 L298 5.4 C281 6.9 252 8.1 211 8.8 C141 9.6 61 10.4 3 10.6 Z"/></svg>';
  const wireSteps = () => {
    const steps = $$('.step');
    steps.forEach(step => {
      if (!step.querySelector('.strike')) step.insertAdjacentHTML('beforeend', pencil);
      const btn = step.querySelector('.done');
      btn.onclick = () => {
        const on = btn.getAttribute('aria-pressed') !== 'true';
        btn.setAttribute('aria-pressed', String(on)); step.classList.toggle('is-done', on);
        if (!result.classList.contains('is-stop')) again.hidden = !steps.every(s => s.classList.contains('is-done'));
      };
    });
  };
  wireSteps();

  // ---- result screen from measured data
  const LIMB = { left: 'Left limb, on your left as you face the tree.', right: 'Right limb.' };
  const stepEl = (n, first, title) => {
    const li = el('li', 'step' + (first ? ' first' : ''));
    li.append(el('span', 'n', String(n)), el('h2', null, title));
    const b = el('button', 'done', 'Done'); b.type = 'button'; b.setAttribute('aria-pressed', 'false'); b.setAttribute('aria-label', `Step ${n} done`);
    li._done = b; return li;
  };
  const render = (d) => {
    const stop = !!d.stop;
    result.classList.toggle('is-stop', stop);
    const img = $('img.marked');
    img.src = stop ? d.photos.stop : d.photos.drawn; img.dataset.plain = img.src; img.dataset.ghost = d.photos.ghost;
    img.alt = 'The bow drawn on the tree' + (stop ? '' : ', with the scrape zones marked');
    const rest = $$('.shots img')[1]; rest.src = d.photos.rest;
    $('#ghost').setAttribute('aria-pressed', 'false');
    const ol = $('.steps'); ol.replaceChildren();
    let n = 1;
    for (const s of d.steps || []) {
      const li = stepEl(n++, s.first, `${LIMB[s.limb]} Pencil a line at each number.`);
      const marks = el('div', 'marks'); marks.append(String(s.pencil[0]));
      const rule = el('span', 'rule'); const span = s.pencil[1] - s.pencil[0] || 1; let at = 0;
      s.zones.forEach((z, k) => {
        const i = el('i', z.call === 'light' ? 'l' : ''); i.style.flex = String(z.far - z.near); rule.append(i);
        if (k > 0) { const em = el('em', null, String(z.near)); em.style.left = `${(at / span) * 100}%`; rule.append(em); }
        at += z.far - z.near;
      });
      marks.append(rule, String(s.pencil[1])); marks.append(el('small', null, 'in'));
      const zones = el('div', 'zones');
      s.zones.forEach(z => { const sp = el('span', z.call === 'light' ? 'l' : ''); sp.append(el('b', null, `${z.near} to ${z.far} in`), el('em', null, `${z.strokes} scraper strokes`)); zones.append(sp); });
      li.append(marks, zones, li._done); ol.append(li);
    }
    const even = !stop && !(d.steps || []).length;
    const pullTo = even ? round1(today() + 2) : round1(today());
    const pull = stepEl(n, false, ''); pull.classList.add('pull');
    pull.querySelector('h2').append(`Pull 20 times to ${pullTo} in. Stop if the scale passes ${round1(bow.wt)} lb.`);
    pull.append(pull._done); ol.append(pull);
    $('.say.do').replaceChildren(...(even ? ['Tiller looks even.', el('br'), 'Pull 2 in further.'] : ['Scrape green.', el('br'), 'Then 20 pulls.']));
    $('.side').hidden = even;
    again.textContent = stop ? 'Let the bow down' : 'New photos'; again.href = stop ? '#bows' : '#check'; again.hidden = !stop;
    wireSteps();
  };

  $('#clearstop').addEventListener('click', async () => {
    bow.stop = false; bow.updated = Date.now(); if (bow.id) await put('bows', bow);
    result.classList.remove('is-stop'); location.hash = '#check';
  });

  // ---- ghost: the target bend, dashed over the drawn photo; off by default
  const ghost = $('#ghost'), marked = $('img.marked');
  marked.dataset.plain = marked.getAttribute('src') || ''; marked.dataset.ghost = '';
  ghost.addEventListener('click', () => {
    const on = ghost.getAttribute('aria-pressed') !== 'true';
    ghost.setAttribute('aria-pressed', String(on));
    marked.src = on ? marked.dataset.ghost : marked.dataset.plain;
    shots.scrollTo({ left: 0 });
  });

  // ---- photos: swipe dots; save the visible marked image (share sheet where files can be shared)
  const shots = $('.shots'), dots = $$('.dots i');
  shots.addEventListener('scroll', () => {
    const i = Math.round(shots.scrollLeft / shots.clientWidth);
    dots.forEach((d, k) => d.classList.toggle('on', k === i));
  }, { passive: true });
  $('#save').addEventListener('click', async () => {
    const img = shots.querySelectorAll('img')[Math.round(shots.scrollLeft / shots.clientWidth)] || marked;
    const blob = await (await fetch(img.src)).blob();
    const file = new File([blob], 'stave-check.jpg', { type: blob.type || 'image/jpeg' });
    if (navigator.canShare && navigator.canShare({ files: [file] })) { try { await navigator.share({ files: [file] }); } catch (_) { /* cancelled */ } return; }
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = file.name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  });

  // ---- measuring worker
  let worker = null, jobs = 0;
  const warm = () => {
    if (worker || !window.Worker) return;
    worker = new Worker('worker.js');
    worker.postMessage({ type: 'warm' });
  };
  const measure = (drawn, rest) => new Promise((resolve, reject) => {
    warm(); const id = ++jobs;
    const on = (e) => {
      if (e.data.id !== id) return;
      worker.removeEventListener('message', on);
      e.data.type === 'result' ? resolve(e.data.data) : reject(new Error(e.data.message));
    };
    worker.addEventListener('message', on);
    worker.postMessage({ type: 'analyze', id, drawn, rest, gridIn: bow.grid, targetWt: bow.wt, todayDraw: today() }, [drawn, rest]);
  });

  // ---- photo inputs: photo 1 is kept; photo 2 measures both, then shows the result
  const meas = $('.measuring'), bad = $('#bad');
  let photo1 = null;
  $$('.shoot input').forEach(inp => inp.addEventListener('change', async () => {
    if (!inp.files.length) return;
    const file = inp.files[0], next = inp.dataset.next;
    inp.value = '';                                  // the same file can be chosen again
    if (!('measure' in inp.dataset)) { photo1 = file; location.hash = next; return; }
    if (!photo1) { location.hash = '#photo1'; return; }
    bad.classList.remove('on');
    meas.hidden = false; inp.closest('.shoot').hidden = true;
    try {
      const d = await measure(await photo1.arrayBuffer(), await file.arrayBuffer());
      if (location.hash !== '#photo2') return;       // he left while it measured
      if (!d.ok) { bad.textContent = d.error; bad.classList.add('on'); return; }
      render(d);
      await saveCheck(d);
      location.hash = next;
    } catch (err) {
      bad.textContent = 'Could not measure. Retake both photos.'; bad.classList.add('on');
      console.error(err);
    } finally {
      meas.hidden = true; inp.closest('.shoot').hidden = false;
    }
  }));

  const thumbOf = (url) => new Promise(res => {
    const im = new Image(); im.onload = () => {
      const c = document.createElement('canvas'); c.width = 192; c.height = 112;
      const k = Math.max(192 / im.width, 112 / im.height), w = im.width * k, h = im.height * k;
      c.getContext('2d').drawImage(im, (192 - w) / 2, (112 - h) / 2, w, h); res(c.toDataURL('image/jpeg', 0.7));
    }; im.onerror = () => res(''); im.src = url;
  });

  const saveCheck = async (d) => {
    if (!bow.id) return;
    const c = { id: uid(), bow: bow.id, at: Date.now(), scaleWt: num($('#c-wt').value), draw: today(),
                setBefore: num($('#s-before').value), setAfter: num($('#s-after').value), data: d };
    await put('checks', c);
    bow.last = c.id; bow.stop = !!d.stop; bow.updated = c.at; bow.thumb = await thumbOf(d.photos.drawn);
    bow.summary = `Last check: ${round1(c.draw)} in at ${round1(c.scaleWt || 0)} lb`;
    await put('bows', bow);
  };

  // ---- startup: reopen the current bow (a reload or a phone that dropped the app mid-scrape)
  (async () => {
    const id = localStorage.getItem('stave.current');
    const b = id && await get('bows', id);
    if (!b) { if (['#check', '#photo1', '#photo2', '#result'].includes(location.hash) && !q.toString()) location.hash = '#bows'; return; }
    useBow(b);
    if (location.hash === '#result' && b.last) { const c = await get('checks', b.last); if (c) render(c.data); }
  })();

  // ---- demo states
  if (q.has('bad')) bad.classList.add('on');
  if (q.has('filled') || q.has('set')) {
    const v = { 'f-name': 'Hickory #1', 'f-wt': '50', 'f-dr': '28', 'f-grid': '2', 'c-wt': '50', 'c-dr': '20', 's-before': '50', 's-after': q.has('set') ? '48' : '50' };
    Object.entries(v).forEach(([k, val]) => { $('#' + k).value = val; });
    $$('[data-form]').forEach(f => f._check()); setCheck();
  }
  if (q.has('stop')) {
    result.classList.add('is-stop');
    marked.alt = 'The bow drawn on the tree';
    again.textContent = 'Let the bow down'; again.href = '#bows'; again.hidden = false;
  }
  if ('serviceWorker' in navigator && location.protocol !== 'file:') navigator.serviceWorker.register('sw.js').catch(() => {});
  caps(); show();
  if (q.has('measuring')) { meas.hidden = false; $('#photo2 .shoot').hidden = true; }
  // test hook: window.staveMeasure(urlDrawn, urlRest) runs the real measuring on two image URLs
  window.staveMeasure = async (u1, u2) => {
    const [a, b] = await Promise.all([u1, u2].map(async u => (await fetch(u)).arrayBuffer()));
    const t0 = performance.now(); const d = await measure(a, b);
    if (d.ok) { render(d); await saveCheck(d); location.hash = '#result'; }
    return { ms: Math.round(performance.now() - t0), ok: d.ok, error: d.error, stop: d.stop, steps: d.steps, balance: d.balance, set_in: d.set_in };
  };
})();
