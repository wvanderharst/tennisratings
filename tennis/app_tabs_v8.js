(function () {
  // ---------------- shared maths (same chain as the Python model) ----------------
  function sig(x) { if (x >= 0) { var z = Math.exp(-x); return 1 / (1 + z); } var z2 = Math.exp(x); return z2 / (1 + z2); }
  function logit(p) { return Math.log(p / (1 - p)); }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function comb(n, k) { var r = 1; for (var i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }
  function matchFromSet(q, bo) {
    var need = bo === 3 ? 2 : 3, t = 0;
    for (var l = 0; l < need; l++) t += comb(need + l - 1, l) * Math.pow(q, need) * Math.pow(1 - q, l);
    return t;
  }
  var spCache = {};
  function setProb(q) {
    if (spCache[q] !== undefined) return spCache[q];
    var memo = {};
    function rec(a, b) {
      var k = a * 100 + b; if (memo[k] !== undefined) return memo[k];
      var v;
      if (a >= 6 && a - b >= 2) v = 1; else if (b >= 6 && b - a >= 2) v = 0; else if (a === 6 && b === 6) v = q;
      else v = q * rec(a + 1, b) + (1 - q) * rec(a, b + 1);
      return (memo[k] = v);
    }
    return (spCache[q] = rec(0, 0));
  }
  function r4(x) { return Math.round(x * 10000) / 10000; }
  function invMatch(p, bo) { var lo = 0.001, hi = 0.999; for (var i = 0; i < 50; i++) { var m = (lo + hi) / 2; if (matchFromSet(m, bo) < p) lo = m; else hi = m; } return (lo + hi) / 2; }
  function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function fmtDate(iso) { var d = new Date(iso + "T00:00:00Z"); return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }); }
  var SURF = ["Hard", "Clay", "Grass"];
  // serve/return split model (women): point probs on serve -> hold -> sets with a point-level tiebreak -> match
  function hold(p) { var q = 1 - p; return Math.pow(p, 4) * (1 + 4 * q + 10 * q * q) + 20 * Math.pow(p, 3) * Math.pow(q, 3) * p * p / (1 - 2 * p * q); }
  var srCache = {};
  function srSet(pa, pb) {
    var key = pa + "_" + pb; if (srCache[key] !== undefined) return srCache[key];
    var ha = hold(pa), hb = hold(pb), mt = {}, mg = {};
    function tb(x, y) {
      if (x >= 7 && x - y >= 2) return 1; if (y >= 7 && y - x >= 2) return 0;
      if (x === y && x >= 6) { var w2 = pa * (1 - pb), l2 = (1 - pa) * pb; return w2 + l2 > 0 ? w2 / (w2 + l2) : 0.5; }
      var k = x * 100 + y; if (mt[k] !== undefined) return mt[k];
      var s = x + y, aServes = (s % 4 === 0) || (s % 4 === 3), p = aServes ? pa : 1 - pb;
      return (mt[k] = p * tb(x + 1, y) + (1 - p) * tb(x, y + 1));
    }
    function g(x, y) {
      if (x >= 6 && x - y >= 2) return 1; if (y >= 6 && y - x >= 2) return 0;
      if (x === 7 || y === 7) return x > y ? 1 : 0;
      if (x === 6 && y === 6) return tb(0, 0);
      var k = x * 100 + y; if (mg[k] !== undefined) return mg[k];
      var p = (x + y) % 2 === 0 ? ha : 1 - hb;
      return (mg[k] = p * g(x + 1, y) + (1 - p) * g(x, y + 1));
    }
    return (srCache[key] = g(0, 0));
  }
  function r3(x) { return Math.round(x * 1000) / 1000; }
  function srMatchLogit(pa, pb, bo) {
    pa = clamp(pa, 0.02, 0.98); pb = clamp(pb, 0.02, 0.98);
    var s = 0.5 * (srSet(r3(pa), r3(pb)) + 1 - srSet(r3(pb), r3(pa)));
    return logit(clamp(matchFromSet(clamp(s, 1e-6, 1 - 1e-6), bo), 1e-6, 1 - 1e-6));
  }

  // ---------------- in-match engine (line-by-line port of inplay_lib.py) ----------------
  // Player 1 wins a point on serve with sigmoid(l1 + u), player 2 with sigmoid(l2 - u); u = form on the day, on a grid.
  var IM_TAU = { atp: 0.15, wta: 0.2 };            // spread of match-day form, fitted on charted matches (training years, point-by-point evidence)
  var UG = (function () { var a = new Float64Array(161); for (var k = 0; k < 161; k++) a[k] = Math.round((-2.4 + 0.03 * k) * 1e4) / 1e4; return a; })();
  var NU = UG.length;
  function vconst(x) { var a = new Float64Array(NU); a.fill(x); return a; }
  function IM(l1, l2, bo, fmt) {
    this.need = (bo >> 1) + 1; this.fmt = fmt;
    var p1 = new Float64Array(NU), p2 = new Float64Array(NU), h1 = new Float64Array(NU), h2 = new Float64Array(NU);
    for (var k = 0; k < NU; k++) { p1[k] = sig(l1 + UG[k]); p2[k] = sig(l2 - UG[k]); h1[k] = hold(p1[k]); h2[k] = hold(p2[k]); }
    this.p = [p1, p2]; this.h = [h1, h2]; this.V = {}; this.T = {}; this.G = [{}, {}]; this.F = {};
  }
  IM.prototype.final = function (s1, s2) { return s1 + s2 === 2 * this.need - 2; };
  IM.prototype.lim = function (s1, s2) { return this.final(s1, s2) && this.fmt === "tb12" ? 12 : 6; };
  IM.prototype.adv = function (s1, s2) { return this.final(s1, s2) && this.fmt === "adv"; };
  IM.prototype.tbto = function (s1, s2) { return this.final(s1, s2) && this.fmt === "tb10" ? 10 : 7; };
  function setWon(a, b) { return a >= 6 && a - b >= 2 ? 1 : (b >= 6 && b - a >= 2 ? -1 : 0); }
  function tbSrv(f, n) { return ((n + 1) >> 1) % 2 === 0 ? f : 1 - f; }
  IM.prototype.game = function (srv, i, j) {           // P(server wins game) from server points i, receiver points j
    var key = i * 100 + j, G = this.G[srv];
    if (G[key]) return G[key];
    var p = this.p[srv], v = new Float64Array(NU), k;
    if (i >= 4 && i - j >= 2) v.fill(1);
    else if (j >= 4 && j - i >= 2) v.fill(0);
    else if (i >= 3 && j >= 3) {
      for (k = 0; k < NU; k++) { var q = 1 - p[k], D = p[k] * p[k] / (p[k] * p[k] + q * q); v[k] = i === j ? D : (i > j ? p[k] + q * D : p[k] * D); }
    } else {
      var a = this.game(srv, i + 1, j), b = this.game(srv, i, j + 1);
      for (k = 0; k < NU; k++) v[k] = p[k] * a[k] + (1 - p[k]) * b[k];
    }
    return (G[key] = v);
  };
  IM.prototype.tb = function (x, y, f, to) {           // P(player 1 wins the tiebreak); f = first server of the tiebreak
    var key = x + "_" + y + "_" + f + "_" + to;
    if (this.T[key]) return this.T[key];
    var v = new Float64Array(NU), k, p1 = this.p[0], p2 = this.p[1];
    if (x >= to && x - y >= 2) v.fill(1);
    else if (y >= to && y - x >= 2) v.fill(0);
    else if (x === y && x >= to - 1) { for (k = 0; k < NU; k++) { var w = p1[k] * (1 - p2[k]), l = (1 - p1[k]) * p2[k]; v[k] = w / (w + l); } }
    else {
      var s = tbSrv(f, x + y), A = this.tb(x + 1, y, f, to), B = this.tb(x, y + 1, f, to);
      for (k = 0; k < NU; k++) { var pw = s === 0 ? p1[k] : 1 - p2[k]; v[k] = pw * A[k] + (1 - pw) * B[k]; }
    }
    return (this.T[key] = v);
  };
  IM.prototype.afterSet = function (s1, s2, srv) {
    if (s1 === this.need) return vconst(1); if (s2 === this.need) return vconst(0);
    return this.val(s1, s2, 0, 0, srv);
  };
  IM.prototype.afterGame = function (s1, s2, a, b, srv) {
    var w = setWon(a, b);
    if (w === 1) return this.afterSet(s1 + 1, s2, srv); if (w === -1) return this.afterSet(s1, s2 + 1, srv);
    return this.val(s1, s2, a, b, srv);
  };
  function mix(g, A, B) { var v = new Float64Array(NU); for (var k = 0; k < NU; k++) v[k] = g[k] * A[k] + (1 - g[k]) * B[k]; return v; }
  IM.prototype.val = function (s1, s2, a, b, srv) {   // P(player 1 wins match) at the start of a game
    var key = s1 + "_" + s2 + "_" + a + "_" + b + "_" + srv;
    if (this.V[key]) return this.V[key];
    var L = this.lim(s1, s2), v, k, h1 = this.h[0], h2 = this.h[1];
    if (a === b && a === L && !this.adv(s1, s2)) {
      v = mix(this.tb(0, 0, srv, this.tbto(s1, s2)), this.afterSet(s1 + 1, s2, 1 - srv), this.afterSet(s1, s2 + 1, 1 - srv));
    } else if (this.adv(s1, s2) && a === b && a >= 5) {
      v = new Float64Array(NU); for (k = 0; k < NU; k++) { var w = h1[k] * (1 - h2[k]), l = (1 - h1[k]) * h2[k]; v[k] = w / (w + l); }
    } else {
      var g1 = new Float64Array(NU); for (k = 0; k < NU; k++) g1[k] = srv === 0 ? h1[k] : 1 - h2[k];
      v = mix(g1, this.afterGame(s1, s2, a + 1, b, 1 - srv), this.afterGame(s1, s2, a, b + 1, 1 - srv));
    }
    return (this.V[key] = v);
  };
  IM.prototype.pointValue = function (s1, s2, a, b, srv, i, j, tbf) {   // before a point; i, j = player 1 / 2 points; tbf = TB first server or null
    if (tbf !== null) return mix(this.tb(i, j, tbf, this.tbto(s1, s2)), this.afterSet(s1 + 1, s2, 1 - tbf), this.afterSet(s1, s2 + 1, 1 - tbf));
    var gs = srv === 0 ? this.game(0, i, j) : this.game(1, j, i), g1 = new Float64Array(NU);
    for (var k = 0; k < NU; k++) g1[k] = srv === 0 ? gs[k] : 1 - gs[k];
    return mix(g1, this.afterGame(s1, s2, a + 1, b, 1 - srv), this.afterGame(s1, s2, a, b + 1, 1 - srv));
  };
  // ---- evidence: log-likelihood over u of the visible score ----
  IM.prototype.setForward = function (f0, s1, s2, upto) {
    var F = { "0_0": vconst(1) }, L = this.lim(s1, s2), adv = this.adv(s1, s2), h1 = this.h[0], h2 = this.h[1];
    for (var n = 0; n < upto; n++) for (var x = 0; x <= n; x++) {
      var y = n - x, cur = F[x + "_" + y]; if (!cur) continue;
      if (setWon(x, y) !== 0) continue; if (x === y && x === L && !adv) continue;
      var srv = n % 2 === 0 ? f0 : 1 - f0, ka = (x + 1) + "_" + y, kb = x + "_" + (y + 1);
      if (!F[ka]) F[ka] = vconst(0); if (!F[kb]) F[kb] = vconst(0);
      for (var k = 0; k < NU; k++) { var g1 = srv === 0 ? h1[k] : 1 - h2[k]; F[ka][k] += cur[k] * g1; F[kb][k] += cur[k] * (1 - g1); }
    }
    return F;
  };
  function vlog(v) { var o = new Float64Array(NU); for (var k = 0; k < NU; k++) o[k] = Math.log(Math.max(v[k], 1e-300)); return o; }
  IM.prototype.llSet = function (a, b, f0, s1, s2) {    // completed set a-b
    var L = this.lim(s1, s2), F, k, o = new Float64Array(NU);
    if (Math.max(a, b) === L + 1 && Math.min(a, b) === L && !this.adv(s1, s2)) {
      F = this.setForward(f0, s1, s2, 2 * L); var T = this.tb(0, 0, f0, this.tbto(s1, s2)), base = F[L + "_" + L];
      for (k = 0; k < NU; k++) o[k] = (base ? base[k] : 0) * (a > b ? T[k] : 1 - T[k]);
      return vlog(o);
    }
    F = this.setForward(f0, s1, s2, a + b);
    var x = a > b ? a - 1 : a, y = a > b ? b : b - 1, srv = (x + y) % 2 === 0 ? f0 : 1 - f0, prev = F[x + "_" + y];
    for (k = 0; k < NU; k++) { var g1 = srv === 0 ? this.h[0][k] : 1 - this.h[1][k]; o[k] = (prev ? prev[k] : 0) * (a > b ? g1 : 1 - g1); }
    return vlog(o);
  };
  IM.prototype.llPartial = function (a, b, f0, s1, s2) { var F = this.setForward(f0, s1, s2, a + b); return vlog(F[a + "_" + b] || vconst(0)); };
  IM.prototype.llGamePts = function (srv, i, j) {       // displayed points (deuce = 3-3, AD = 4-3)
    var si = srv === 0 ? i : j, sj = srv === 0 ? j : i;
    if (si >= 3 && sj >= 3) { if (si === sj) { si = 3; sj = 3; } else if (si > sj) { si = 4; sj = 3; } else { si = 3; sj = 4; } }
    var p = this.p[srv], o = new Float64Array(NU);
    for (var k = 0; k < NU; k++) o[k] = si * Math.log(p[k]) + sj * Math.log(1 - p[k]);
    return o;
  };
  IM.prototype.llTbPts = function (x, y, f) {
    var F = { "0_0": vconst(1) };
    for (var n = 0; n < x + y; n++) for (var a = 0; a <= n; a++) {
      var b = n - a, cur = F[a + "_" + b]; if (!cur) continue;
      var s = tbSrv(f, n), ka = (a + 1) + "_" + b, kb = a + "_" + (b + 1);
      if (!F[ka]) F[ka] = vconst(0); if (!F[kb]) F[kb] = vconst(0);
      for (var k = 0; k < NU; k++) { var pw = s === 0 ? this.p[0][k] : 1 - this.p[1][k]; F[ka][k] += cur[k] * pw; F[kb][k] += cur[k] * (1 - pw); }
    }
    return vlog(F[x + "_" + y] || vconst(0));
  };

  // ---- score state: {sets: [[a, b], ...], g: [x, y], pt: [i, j], srv: 0|1}; scores player 1 first ----
  function lvRules(bo, fmt, nsets) { var need = (bo >> 1) + 1, fin = nsets === 2 * need - 2; return { need: need, fin: fin, L: fin && fmt === "tb12" ? 12 : 6, adv: fin && fmt === "adv", to: fin && fmt === "tb10" ? 10 : 7 }; }
  function lvInTb(st, bo, fmt) { var r = lvRules(bo, fmt, st.sets.length); return st.g[0] === r.L && st.g[1] === r.L && !r.adv; }
  function lvWon(st) { var w = [0, 0]; st.sets.forEach(function (s) { w[s[0] > s[1] ? 0 : 1]++; }); return w; }
  function lvFirstServers(st, bo, fmt) {       // first server of the match, of each completed set, of the current set, and of a tiebreak in progress
    var K = 0; st.sets.forEach(function (s) { K += s[0] + s[1]; }); var Kc = K + st.g[0] + st.g[1], tbf = null, fm;
    if (lvInTb(st, bo, fmt)) { var n = st.pt[0] + st.pt[1]; tbf = ((n + 1) >> 1) % 2 === 0 ? st.srv : 1 - st.srv; fm = Kc % 2 === 0 ? tbf : 1 - tbf; }
    else fm = Kc % 2 === 0 ? st.srv : 1 - st.srv;
    var fs = [], G = 0; st.sets.forEach(function (s) { fs.push(G % 2 === 0 ? fm : 1 - fm); G += s[0] + s[1]; });
    return { fm: fm, fs: fs, fc: G % 2 === 0 ? fm : 1 - fm, tbf: tbf };
  }
  function lvAdvance(st, w, bo, fmt) {          // add one point won by w (0/1); returns a new state
    var s = { sets: st.sets.map(function (x) { return x.slice(); }), g: st.g.slice(), pt: st.pt.slice(), srv: st.srv,
              base: st.base ? { sets: st.base.sets.map(function (x) { return x.slice(); }), g: st.base.g.slice() } : { sets: [], g: [0, 0] },
              trk: st.trk ? st.trk.slice() : [0, 0, 0, 0], unt: !!st.unt };
    if (!s.unt) s.trk[2 * s.srv + (w === s.srv ? 0 : 1)]++;     // tracked point: [p1 serve won, lost, p2 serve won, lost]
    var gBefore = s.g[0] + s.g[1], setsBefore = s.sets.length;
    var r = lvRules(bo, fmt, s.sets.length), done = false, nsrv;
    if (lvInTb(s, bo, fmt)) {
      var n = s.pt[0] + s.pt[1], tbf = ((n + 1) >> 1) % 2 === 0 ? s.srv : 1 - s.srv;
      s.pt[w]++;
      if (s.pt[w] >= r.to && s.pt[w] - s.pt[1 - w] >= 2) { s.g[w]++; done = true; nsrv = 1 - tbf; }
      else s.srv = tbSrv(tbf, n + 1);
    } else {
      s.pt[w]++;
      if (s.pt[w] >= 4 && s.pt[w] - s.pt[1 - w] >= 2) {
        s.g[w]++; s.pt = [0, 0]; s.srv = 1 - s.srv;
        if (setWon(s.g[0], s.g[1]) !== 0) { done = true; nsrv = s.srv; }
      } else if (s.pt[0] >= 3 && s.pt[1] >= 3) { var m = Math.min(s.pt[0], s.pt[1]) - 3; s.pt[0] -= m; s.pt[1] -= m; }
    }
    if (done) { s.sets.push(s.g.slice()); s.g = [0, 0]; s.pt = [0, 0]; s.srv = nsrv; }
    if (s.unt && (s.sets.length !== setsBefore || s.g[0] + s.g[1] !== gBefore)) {   // the typed game is over: track from the next game on
      s.unt = false; s.base = { sets: s.sets.map(function (x) { return x.slice(); }), g: s.g.slice() };
    }
    return s;
  }
  var PTS = ["0", "15", "30", "40", "AD"];
  function lvText(st, bo, fmt) {
    var parts = st.sets.map(function (s) { return s[0] + "-" + s[1]; }), won = lvWon(st), need = (bo >> 1) + 1;
    if (won[0] === need || won[1] === need) return parts.join(" ");
    parts.push(st.g[0] + "-" + st.g[1]);
    if (st.pt[0] || st.pt[1]) {
      if (lvInTb(st, bo, fmt)) parts.push(st.pt[0] + "-" + st.pt[1]);
      else { var a = st.pt[0], b = st.pt[1]; parts.push(a >= 3 && b >= 3 ? (a === b ? "40-40" : (a > b ? "AD-40" : "40-AD")) : PTS[a] + "-" + PTS[b]); }
    }
    return parts.join(" ");
  }
  function lvParse(txt, bo, fmt, srv) {         // -> {st} or {err}
    var toks = txt.replace(/\(\d+\)/g, "").trim().split(/[\s,;]+/).filter(Boolean), st = { sets: [], g: [0, 0], pt: [0, 0], srv: srv }, need = (bo >> 1) + 1;
    var stage = 0;                               // 0: sets, 1: after current-set games, 2: after points
    for (var t = 0; t < toks.length; t++) {
      var m = toks[t].toUpperCase().match(/^(\d+|AD|A)[-–:](\d+|AD|A)$/);
      if (!m) return { err: "Couldn’t read “" + toks[t] + "”. Write each score as 6-4, points as 30-15 or AD-40." };
      var won = lvWon(st);
      if (won[0] === need || won[1] === need) return { err: "The match is already over after " + st.sets.map(function (s) { return s.join("-"); }).join(" ") + "." };
      var r = lvRules(bo, fmt, st.sets.length);
      if (stage === 0) {
        var a = +m[1], b = +m[2]; if (isNaN(a) || isNaN(b)) return { err: "“" + toks[t] + "” looks like points; enter the games of the current set first (e.g. 2-1 30-15)." };
        var hi2 = Math.max(a, b), lo2 = Math.min(a, b), df = hi2 - lo2;
        // completed: winner has 6+ and leads by 2 (exactly 2 beyond 6 games), or won the tiebreak (L+1 to L)
        var fin = (hi2 >= 6 && df >= 2 && (hi2 === 6 || df === 2) && (r.adv || hi2 <= r.L + 1)) || (!r.adv && hi2 === r.L + 1 && lo2 === r.L);
        if (fin) { st.sets.push([a, b]); continue; }
        var ok = !(hi2 >= 6 && df >= 2) && (r.adv || hi2 <= r.L);
        if (!ok) return { err: "“" + toks[t] + "” isn’t a possible set score here." };
        st.g = [a, b]; stage = 1; continue;
      }
      if (stage === 1) {
        if (lvInTb(st, bo, fmt)) {
          var x = +m[1], y = +m[2]; if (isNaN(x) || isNaN(y)) return { err: "In a tiebreak, enter points as numbers, e.g. 6-6 4-3." };
          if ((x >= r.to || y >= r.to) && Math.abs(x - y) >= 2) return { err: "That tiebreak is already over: enter the set as " + (x > y ? (r.L + 1) + "-" + r.L : r.L + "-" + (r.L + 1)) + "." };
          if (Math.abs(x - y) > 2 && Math.max(x, y) > r.to) return { err: "“" + toks[t] + "” isn’t a possible tiebreak score." };
          st.pt = [x, y];
        } else {
          var map = { "0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4 }, i = map[m[1]], j = map[m[2]];
          if (i === undefined || j === undefined) return { err: "Points are 0, 15, 30, 40 or AD (e.g. 30-15, AD-40)." };
          if ((i === 4 && j !== 3) || (j === 4 && i !== 3)) return { err: "AD only comes after deuce: AD-40 or 40-AD." };
          st.pt = [i, j];
        }
        stage = 2; continue;
      }
      return { err: "Too many scores: sets, then current-set games, then points." };
    }
    st.base = { sets: st.sets.map(function (x) { return x.slice(); }), g: st.g.slice() }; st.trk = [0, 0, 0, 0]; st.unt = st.pt[0] + st.pt[1] > 0;
    return { st: st };
  }
  function lvEval(R, st, bo, fmt, tau) {        // live probability for player 1 (completed match)
    var M = new IM(logit(R.ha), logit(R.hb), bo, fmt), p0 = R.pDone, k;
    var v0a = M.val(0, 0, 0, 0, 0), v0b = M.val(0, 0, 0, 0, 1), v0 = new Float64Array(NU);
    for (k = 0; k < NU; k++) v0[k] = 0.5 * (v0a[k] + v0b[k]);
    var won = lvWon(st), need = (bo >> 1) + 1;
    if (won[0] === need) return { over: 1 }; if (won[1] === need) return { over: 2 };
    var fs = lvFirstServers(st, bo, fmt), tb = lvInTb(st, bo, fmt);
    var V = M.pointValue(won[0], won[1], st.g[0], st.g[1], st.srv, st.pt[0], st.pt[1], tb ? fs.tbf : null);
    // form: prior N(d, tau^2) with d chosen so the prior reproduces the pre-match chance, updated by the score
    var lo = -2, hi = 2, d = 0;
    for (var it = 0; it < 40; it++) { d = (lo + hi) / 2; var sw = 0, sv = 0; for (k = 0; k < NU; k++) { var w = Math.exp(-0.5 * Math.pow((UG[k] - d) / tau, 2)); sw += w; sv += w * v0[k]; } if (sv / sw < p0) lo = d; else hi = d; }
    var L = new Float64Array(NU); for (k = 0; k < NU; k++) L[k] = -0.5 * Math.pow((UG[k] - d) / tau, 2);
    // the score alone: same prior, no evidence (so "form on the day" is exactly what the games played add)
    var pw0 = 0, pv0 = 0; for (k = 0; k < NU; k++) { var w0 = Math.exp(L[k]); pw0 += w0; pv0 += w0 * V[k]; }
    var pStatic = pv0 / pw0;
    // evidence about form: the typed scoreboard (completed sets + games), plus every point stepped through with the buttons
    var base = st.base || { sets: st.sets, g: st.g }, trk = st.trk || [0, 0, 0, 0];
    var s1 = 0, s2 = 0, Es = new Float64Array(NU);
    base.sets.forEach(function (s, i) { var l = M.llSet(s[0], s[1], fs.fs[i], s1, s2); for (var q = 0; q < NU; q++) Es[q] += l[q]; if (s[0] > s[1]) s1++; else s2++; });
    var bi = base.sets.length, bf = bi < st.sets.length ? fs.fs[bi] : fs.fc;
    var lp = M.llPartial(base.g[0], base.g[1], bf, s1, s2), Lsb = new Float64Array(NU), mx = -Infinity;
    for (k = 0; k < NU; k++) {
      var p1 = M.p[0][k], p2 = M.p[1][k];
      Lsb[k] = L[k] + Es[k] + lp[k] + trk[0] * Math.log(p1) + trk[1] * Math.log(1 - p1) + trk[2] * Math.log(p2) + trk[3] * Math.log(1 - p2);
      if (Lsb[k] > mx) mx = Lsb[k];
    }
    var Wsb = new Float64Array(NU), W = 0, U1 = 0, U2 = 0;
    for (k = 0; k < NU; k++) { Wsb[k] = Math.exp(Lsb[k] - mx); W += Wsb[k]; U1 += Wsb[k] * UG[k]; U2 += Wsb[k] * UG[k] * UG[k]; }
    var P = 0, i0 = st.pt[0], j0 = st.pt[1];
    if (i0 + j0 === 0 || !st.unt) { for (k = 0; k < NU; k++) P += Wsb[k] * V[k]; P /= W; }
    else {
      // typed score inside a game or tiebreak: the points set the chance to win it (exact chain); each outcome leads to a scoreboard
      // with its own form update (so a typed 40-0 never rates above the 1-0 it leads to). Here base = current scoreboard and nothing is tracked.
      var P1g;
      if (tb) P1g = M.tb(i0, j0, fs.tbf, M.tbto(s1, s2));
      else { var gs = st.srv === 0 ? M.game(0, i0, j0) : M.game(1, j0, i0); P1g = new Float64Array(NU); for (k = 0; k < NU; k++) P1g[k] = st.srv === 0 ? gs[k] : 1 - gs[k]; }
      var pA = 0; for (k = 0; k < NU; k++) pA += Wsb[k] * P1g[k]; pA /= W;
      var Px = [0, 1].map(function (w) {
        var ng = st.g.slice(); ng[w]++;
        var nsrv = tb ? 1 - fs.tbf : 1 - st.srv, done = tb || setWon(ng[0], ng[1]) !== 0, ev, Vx;
        if (done) {
          ev = M.llSet(ng[0], ng[1], fs.fc, s1, s2);
          Vx = ng[0] > ng[1] ? M.afterSet(s1 + 1, s2, nsrv) : M.afterSet(s1, s2 + 1, nsrv);
        } else { ev = M.llPartial(ng[0], ng[1], fs.fc, s1, s2); Vx = M.val(s1, s2, ng[0], ng[1], nsrv); }
        var mm = -Infinity, Lx = new Float64Array(NU), q;
        for (q = 0; q < NU; q++) { Lx[q] = L[q] + Es[q] + ev[q]; if (Lx[q] > mm) mm = Lx[q]; }
        var sw = 0, sv = 0; for (q = 0; q < NU; q++) { var ww = Math.exp(Lx[q] - mm); sw += ww; sv += ww * Vx[q]; }
        return sv / sw;
      });
      P = pA * Px[0] + (1 - pA) * Px[1];
    }
    var um = U1 / W, usd = Math.sqrt(Math.max(U2 / W - um * um, 0));
    return { trk: trk, p: clamp(P, 1e-6, 1 - 1e-6), pStatic: clamp(pStatic, 1e-6, 1 - 1e-6), p0: p0, form: um - d, formSd: usd, d: d, srv1: sig(logit(R.ha) + d), srv2: sig(logit(R.hb) - d),
             srv1t: sig(logit(R.ha) + um), srv2t: sig(logit(R.hb) - um) };
  }

  // ---------------- data loading ----------------
  var cache = {};
  function load(name) {
    if (!cache[name]) cache[name] = fetch(name).then(function (r) {
      if (!r.ok) throw new Error(name + " (" + r.status + ")");
      return r.json();
    });
    return cache[name];
  }
  var ODDS = {};
  function loadOdds(tour) {
    return load("odds_" + tour + ".json").then(function (d) {
      if (!d._prep) {
        d._prep = true;
        d.byName = {}; d.players.forEach(function (p, i) { p.i = i; d.byName[p.n.toLowerCase()] = p; });
        d.h2hMap = {};
        // display only: each surface rating put on the general-rating scale by matching its average
        // and spread to the general rating's (over these players). Unlike a rank map this keeps the
        // size of a lead, so a player who's No. 1 everywhere still shows where he's strongest.
        function ms(v) { var m = v.reduce(function (a, b) { return a + b; }, 0) / v.length;
          return [m, Math.sqrt(v.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / v.length)]; }
        var G = ms(d.players.map(function (p) { return p.g; }));
        [0, 1, 2].forEach(function (si) {
          var bv = d.players.map(function (p) { return blendOf(d, p, si); }), B = ms(bv);
          d.players.forEach(function (p, k) { (p.gsc = p.gsc || [])[si] = d.const.gap ? bv[k] : G[0] + (bv[k] - B[0]) * G[1] / B[1]; });
        });
        d.h2h.forEach(function (h) { var i = h[0], j = h[1]; if (i < j) d.h2hMap[i + "_" + j] = [h[2], h[3]]; else d.h2hMap[j + "_" + i] = [h[3], h[2]]; });
      }
      ODDS[tour] = d; return d;
    });
  }

  // ---------------- tabs ----------------
  var panels = { chart: document.getElementById("panelChart"), rankings: document.getElementById("panelRankings"), odds: document.getElementById("panelOdds"), profile: document.getElementById("panelProfile"), peaks: document.getElementById("panelPeaks") };
  var tabBtns = Array.prototype.slice.call(document.querySelectorAll("#tabs button"));
  var chartHost = document.getElementById("chartHost");
  function showTab(tab) {
    tabBtns.forEach(function (b) { b.setAttribute("aria-selected", b.dataset.tab === tab ? "true" : "false"); });
    panels.chart.hidden = !(tab === "men" || tab === "women");
    panels.rankings.hidden = tab !== "rankings";
    panels.odds.hidden = tab !== "odds";
    panels.profile.hidden = tab !== "profile";
    panels.peaks.hidden = tab !== "peaks";
    var sub = document.getElementById("subtitle");
    if (tab === "men" || tab === "women") {
      var tour = tab === "men" ? "atp" : "wta";
      if (window.__chartTour() !== tour) {
        chartHost.innerHTML = '<div class="loading">Loading rating histories&hellip;</div>';
        load("traj_" + tour + ".json").then(function (d) { window.__chartSetTour(tour, d); chartSub[tour] = sub.textContent; })
          .catch(function (e) { chartHost.innerHTML = '<div class="loading">Couldn’t load the rating histories: ' + esc(e.message) + '</div>'; });
      } else if (chartSub[tour]) sub.textContent = chartSub[tour];
    } else if (tab === "rankings") {
      sub.textContent = "Every active player ranked on what the full model expects them to win: average chance against the current No. 5\u201315";
      renderRankings();
    } else if (tab === "peaks") {
      sub.textContent = "The 100 greatest peaks: each player's biggest lead over that moment's No. 5\u201315, men since 1969 and women since 1992";
      renderPeaks();
    } else if (tab === "profile") {
      sub.textContent = "Every match of a player: the model's pre-match win chance and what each result did to the ratings";
      renderProfile();
    } else {
      sub.textContent = "Pick two players and a court: the serve/return model's pre-match win probability, and where the edge comes from";
      renderOdds();
    }
    try { history.replaceState(null, "", "#" + tab); } catch (e) {}
  }
  var chartSub = {};
  tabBtns.forEach(function (b) {
    b.addEventListener("click", function () { showTab(b.dataset.tab); });
    b.addEventListener("keydown", function (e) {
      var i = tabBtns.indexOf(b);
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        var n = tabBtns[(i + (e.key === "ArrowRight" ? 1 : tabBtns.length - 1)) % tabBtns.length];
        n.focus(); showTab(n.dataset.tab);
      }
    });
  });
  function segBind(id, attr, onPick) {
    var btns = Array.prototype.slice.call(document.querySelectorAll("#" + id + " button"));
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        if (b.disabled) return;
        btns.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        onPick(b.dataset[attr]);
      });
    });
    return function set(v) { btns.forEach(function (x) { x.setAttribute("aria-pressed", x.dataset[attr] === String(v) ? "true" : "false"); }); };
  }

  // ---------------- ratings tab ----------------
  var rk = { tour: "atp", surf: "Hard", sort: "strength", dir: -1, all: false, q: "" };
  var rkTable = document.getElementById("rkTable");
  segBind("rkTour", "tour", function (v) { rk.tour = v; rk.all = false; renderRankings(); });
  segBind("rkSurf", "surf", function (v) { rk.surf = v; rk.sort = "strength"; rk.dir = -1; renderRankings(); });
  document.getElementById("rkSearch").addEventListener("input", function (e) { rk.q = e.target.value.trim().toLowerCase(); renderRankings(); });
  var rkMore = document.getElementById("rkMore");
  function toggleMore() { rk.all = !rk.all; renderRankings(); }
  rkMore.addEventListener("click", toggleMore);
  rkMore.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleMore(); } });

  // Oct 2026 serve/return model: each player has a serve and a return rating, general + per surface. On a surface the
  // model uses 60% general + 40% the player's own surface rating (from their first match there); with no match on the
  // surface, the surface rating at their general serve (return) rank among active players who have played there.
  function qmapJS(x, gs, ss) {
    var n = gs.length, lo = 0, hi = n;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (gs[mid] < x) lo = mid + 1; else hi = mid; }
    var xx = lo / n * n - 0.5;
    if (xx <= 0) return ss[0];
    if (xx >= n - 1) return ss[n - 1];
    var i = Math.floor(xx), f = xx - i;
    return ss[i] * (1 - f) + ss[i + 1] * f;
  }
  function srEff(D, p, si) {      // [serve, return] on surface si (si < 0: general)
    if (si < 0) return [p.sv, p.rt];
    if (D.const.gap) return [p.sv + D.const.lam * p.ssv[si], p.rt + D.const.lam * p.srt[si]];   // general + surface gap
    var W = D.const.BLEND_W;
    if (p.ns[si] >= 1) return [W * p.sv + (1 - W) * p.ssv[si], W * p.rt + (1 - W) * p.srt[si]];
    var fp = D.fill_pool && D.fill_pool[SURF[si]];
    if (!fp) return [p.sv, p.rt];
    return [W * p.sv + (1 - W) * qmapJS(p.sv, fp[0], fp[1]), W * p.rt + (1 - W) * qmapJS(p.rt, fp[2], fp[3])];
  }
  function blendOf(D, p, si) { var e = srEff(D, p, si); return (e[0] + e[1]) / 2; }
  function winVs(C, b, c, refB, refC) {
    var qg = clamp(sig(b - refB), 1e-9, 1 - 1e-9);
    var qs = clamp(sig(logit(setProb(r4(qg))) + c - refC), 1e-6, 1 - 1e-6);
    var qm = clamp(matchFromSet(qs, 3), 1e-6, 1 - 1e-6);
    return sig(C.slope * (C.CAL_I + C.CAL_S * logit(qm)));
  }
  // "Strength": the full model's average chance against the current No. 5-15 (tour event, best of 3, standard court),
  // i.e. ranking on what a player is expected to WIN, not on the general rating alone. General = 60% hard, 30% clay, 10% grass.
  function strengthOf(D, si) {
    D._str = D._str || {}; var key = "s" + si; if (D._str[key]) return D._str[key];
    var field = D.players.slice().sort(function (a, b) { return b.g - a.g; }).slice(4, 15);   // No. 5-15 by general rating
    var surfs = si < 0 ? [[0, 0.6], [1, 0.3], [2, 0.1]] : [[si, 1]], out = {};
    D.players.forEach(function (p) {
      var tot = 0;
      surfs.forEach(function (sw) {
        var o = { surf: SURF[sw[0]], indoor: false, lvl: "A", bo: 3, speed: 0 }, s = 0, k = 0;
        field.forEach(function (q) { if (q !== p) { s += compute(D, p, q, o).p; k++; } });
        tot += sw[1] * s / k;
      });
      out[p.n] = tot;
    });
    return (D._str[key] = out);
  }
  function renderRankings() {
    if (panels.rankings.hidden) return;
    rkTable.innerHTML = '<tr><td class="loading" style="text-align:center">Loading ratings&hellip;</td></tr>';
    loadOdds(rk.tour).then(function (D) {
      var C = D.const, si = SURF.indexOf(rk.surf);
      var rows = D.players.map(function (p) {
        var e = srEff(D, p, si);
        return { p: p, model: blendOf(D, p, si), rating: si < 0 ? p.g : p.gsc[si], hard: p.gsc[0], clay: p.gsc[1], grass: p.gsc[2], sve: e[0], rte: e[1] };
      });
      var STR = strengthOf(D, si);
      // classic Elo (surface views: 50% overall + 50% surface Elo), ranked among these active players
      rows.forEach(function (r) { r.elo = si < 0 ? r.p.elo : 0.5 * r.p.elo + 0.5 * r.p.eloS[si]; });
      rows.slice().sort(function (a, b) { return b.elo - a.elo; }).forEach(function (r, i) { r.erank = i + 1; });
      rows.forEach(function (r) { r.win = STR[r.p.n]; });
      rows.sort(function (a, b) { return b.win - a.win; });
      rows.forEach(function (r, i) { r.pos = i + 1; });
      rows.forEach(function (r) { r.age = r.p.age; r.closing = r.p.c; r.clutch = r.p.cl; r.trank = r.p.rk || 99999; r.matches = r.p.m; r.name = r.p.n; r.serve = r.sve; r.ret = r.rte; });
      var sorted = rows.slice();
      if (rk.sort === "strength" && rk.dir > 0) sorted.reverse();
      if (rk.sort !== "strength") sorted.sort(function (a, b) {
        var x = a[rk.sort], y = b[rk.sort];
        if (typeof x === "string") return rk.dir * x.localeCompare(y);
        return rk.dir * ((x == null ? -1e9 : x) - (y == null ? -1e9 : y));
      });
      if (rk.q) sorted = sorted.filter(function (r) { return r.name.toLowerCase().indexOf(rk.q) !== -1; });
      var lim = rk.all || rk.q ? sorted.length : 50;
      var surfName = rk.surf || "General";
      var cols = [
        ["pos", "#", "l"], ["name", "Player", "l"], ["age", "Age"], ["win", "Strength"], ["rating", surfName + " rating"],
        ["hard", "Hard"], ["clay", "Clay"], ["grass", "Grass"], ["closing", "Closing"], ["clutch", "Clutch"],
        ["trank", C.tour === "atp" ? "ATP rank" : "WTA rank"], ["matches", "Matches"]
      ];
      var hasSR = true;
      cols.splice(5, 0, ["serve", "Serve"], ["ret", "Return"]);
      cols.push(["erank", "Elo rank"], ["elo", "Elo"]);
      var head = "<thead><tr>" + cols.map(function (c) {
        var active = (rk.sort === c[0]) || (rk.sort === "strength" && (c[0] === "pos" || c[0] === "win"));
        return '<th class="' + (c[2] || "") + (active ? " sorted" : "") + '"><button data-sort="' + c[0] + '">' + c[1] + (active && c[0] !== "pos" ? (rk.dir < 0 ? " ↓" : " ↑") : "") + "</button></th>";
      }).join("") + "</tr></thead>";
      function sc(v, p, i) {
        var thin = p.ns[i] < C.MIN_SURF;
        return '<td class="num"' + (thin ? ' style="opacity:.45" title="Never played on this surface: rated the same as their general rating"' : "") + ">" + v.toFixed(2) + "</td>";
      }
      var body = sorted.slice(0, lim).map(function (r) {
        var p = r.p, w = r.win * 100;
        return "<tr>" +
          '<td class="pos l">' + r.pos + "</td>" +
          '<td class="l"><span class="pname link" data-n="' + esc(p.n) + '" title="Open player profile">' + esc(p.n) + '</span><span class="ioc">' + esc(p.ioc || "") + "</span></td>" +
          '<td class="num">' + (p.age != null ? p.age.toFixed(1) : "–") + "</td>" +
          '<td><span class="winbar"><span class="num" style="font-family:var(--mono);font-size:12px;font-weight:700">' + w.toFixed(1) + '%</span><span class="track"><i style="width:' + clamp(w, 0, 100).toFixed(1) + '%"></i></span></span></td>' +
          '<td class="num">' + r.rating.toFixed(3) + "</td>" +
          '<td class="num">' + r.sve.toFixed(2) + '</td><td class="num">' + r.rte.toFixed(2) + "</td>" +
          sc(r.hard, p, 0) + sc(r.clay, p, 1) + sc(r.grass, p, 2) +
          '<td class="num">' + (p.c >= 0 ? "+" : "") + p.c.toFixed(2) + "</td>" +
          '<td class="num">' + (p.cl >= 0 ? "+" : "") + p.cl.toFixed(3) + "</td>" +
          '<td class="num">' + (p.rk || "–") + (p.rk && p.rk !== r.pos ? ' <span style="opacity:.55;font-size:11px">' + (p.rk > r.pos ? "▼" : "▲") + Math.abs(p.rk - r.pos) + "</span>" : "") + "</td>" +
          '<td class="num">' + p.m + "</td>" +
          '<td class="num" style="color:var(--ink-soft)">' + r.erank + (r.erank !== r.pos ? ' <span style="opacity:.55;font-size:11px">' + (r.erank > r.pos ? "▼" : "▲") + Math.abs(r.erank - r.pos) + "</span>" : "") + "</td>" +
          '<td class="num" style="color:var(--ink-soft)">' + Math.round(r.elo) + "</td>" + "</tr>";
      }).join("");
      rkTable.innerHTML = head + "<tbody>" + (body || '<tr><td colspan="16" class="l" style="color:var(--ink-soft)">No active player matches that name.</td></tr>') + "</tbody>";
      Array.prototype.forEach.call(rkTable.querySelectorAll("th button"), function (b) {
        b.addEventListener("click", function () {
          var k = b.dataset.sort;
          if (k === "pos" || k === "win") { if (rk.sort === "strength") rk.dir = -rk.dir; else { rk.sort = "strength"; rk.dir = -1; } }
          else if (rk.sort === k) rk.dir = -rk.dir;
          else { rk.sort = k; rk.dir = (k === "name" || k === "trank" || k === "age" || k === "erank") ? 1 : -1; }
          renderRankings();
        });
      });
      Array.prototype.forEach.call(rkTable.querySelectorAll(".pname.link"), function (s) { s.addEventListener("click", function () { openProfile(rk.tour, s.dataset.n); window.scrollTo(0, 0); }); });
      rkMore.hidden = !!rk.q || sorted.length <= 50;
      rkMore.textContent = rk.all ? "Show top 50" : "Show all " + sorted.length;
      document.getElementById("rkAsOf").textContent = "Ratings as of " + fmtDate(C.asof) + ", the last match in the data · " +
        D.players.length + " active players (30+ matches, played in the past year)";
      document.getElementById("rkNote").innerHTML = "<strong>Every player has a serve rating and a return rating</strong>: how much better than an average tour player they win points on serve, and on return, adjusted for who they faced. The <em>rating</em> is their average. <strong>Surface ratings</strong> are the general rating plus that player's own gap on the surface, and the gaps balance out (weighted hard 60%, clay 30%, grass 10%), so the general rating is always that weighted average of the three surface ratings — never above a player's best surface or below their worst. <strong>Serve</strong> and <strong>Return</strong> show the two halves on the surface picked above. The model predicts each match from serve-versus-return point chances → service holds → sets (tiebreaks point by point) → match. Faded cells: never played there, so rated the same as general. " +
        "<strong>Elo</strong> is a classic Elo rating for comparison, on the same matches: wins and losses only (Tennis Abstract style, K = 250/(n+5)^0.4; surface views average overall and surface Elo). <strong>Elo rank</strong> is the rank it gives among these players; the arrows next to <strong>Elo rank</strong> and the official " + (C.tour === "atp" ? "ATP" : "WTA") + " rank show how many places that ranking puts a player below (▼) or above (▲) the model's Strength rank (the official ranking counts every player, including injured ones, so gaps there can be larger). On held-out matches (2017+ men, late 2018+ women) the model beats Elo clearly: logloss 0.610 vs 0.634 and accuracy 66.1% vs 63.4% for men, 0.592 vs 0.616 and 68.0% vs 65.8% for women. " +
        "<strong>Strength</strong> ranks players on what the full model expects them to <em>win</em>: their average chance against the players currently ranked 5th to 15th on the general rating (tour event, best of 3, standard court; a player in that group is not counted against themselves), including closing, clutch and head-to-heads; <em>General</em> weights hard 60%, clay 30%, grass 10%. <strong>Closing</strong> is the set-level finishing rating and <strong>Clutch</strong> the deciding-set rating. " +
        (C.tour === "atp" ? "" : "Women's correction weights (clutch, top-4 at Slams, age, height, court speed) are fitted on WTA matches; fewer women qualify as active (30+ main-tour matches in the past year), because the data has no ITF or qualifying matches. ") + "Click a column to sort, or a name to open that player\u2019s profile.";
    }).catch(function (e) { rkTable.innerHTML = '<tr><td class="loading">Couldn’t load ratings: ' + esc(e.message) + "</td></tr>"; });
  }

  // ---------------- all-time peaks tab ----------------
  var pk = { tour: "atp", by: "lead" };
  var pkTable = document.getElementById("pkTable");
  segBind("pkTour", "tour", function (v) { pk.tour = v; renderPeaks(); });
  segBind("pkBy", "by", function (v) { pk.by = v; renderPeaks(); });
  function sgn(x, d) { return (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(d); }
  function renderPeaks() {
    pkTable.innerHTML = '<tr><td class="loading" style="text-align:center">Loading peaks&hellip;</td></tr>';
    var tour = pk.tour;
    load("peaks_" + tour + ".json").then(function (data) {
      if (pk.tour !== tour) return;
      var sorted = data[pk.by];          // two lists: highest level, and biggest lead over No. 5-15 (each at its own moment)
      var head = "<thead><tr>" + [["#", "l"], ["Player", "l"], [pk.by === "lv" ? "Peak rating \u2193" : "Rating then", ""],
        [pk.by === "lead" ? "Lead over No. 5\u201315 \u2193" : "Lead over No. 5\u201315", ""], ["Serve", ""], ["Return", ""], ["Date", ""], ["Age", ""],
        ["Matches then", ""], ["Slams", ""]].map(function (c) {
          var on = c[0].indexOf("\u2193") >= 0;
          return '<th class="' + c[1] + (on ? " sorted" : "") + '"><button tabindex="-1">' + c[0] + "</button></th>";
        }).join("") + "</tr></thead>";
      var body = sorted.map(function (r, i) {
        return "<tr>" +
          '<td class="pos l">' + (i + 1) + "</td>" +
          '<td class="l"><span class="pname link" data-n="' + esc(r.n) + '" title="Open player profile">' + esc(r.n) + "</span></td>" +
          '<td class="num" style="font-weight:700">' + r.lv.toFixed(3) + "</td>" +
          '<td class="num">' + sgn(r.lead, 3) + "</td>" +
          '<td class="num">' + r.sv.toFixed(2) + '</td><td class="num">' + r.rt.toFixed(2) + "</td>" +
          '<td class="num">' + fmtDate(r.d) + "</td>" +
          '<td class="num">' + (r.age != null ? r.age.toFixed(1) : "\u2013") + "</td>" +
          '<td class="num">' + r.m + "</td>" +
          '<td class="num">' + (r.slams || "") + "</td></tr>";
      }).join("");
      pkTable.innerHTML = head + "<tbody>" + body + "</tbody>";
      Array.prototype.forEach.call(pkTable.querySelectorAll(".pname.link"), function (s) {
        s.addEventListener("click", function () { openProfile(tour, s.dataset.n); window.scrollTo(0, 0); });
      });
    }).catch(function (e) { pkTable.innerHTML = '<tr><td class="loading">Couldn\u2019t load the peaks: ' + esc(e.message) + "</td></tr>"; });
  }

  // ---------------- match odds tab ----------------
  var od = { tour: "atp", surf: "Hard", indoor: false, lvl: "A", bo: 3, speed: 0, preset: "", names: { atp: ["Jannik Sinner", "Carlos Alcaraz"], wta: null } };
  var inA = document.getElementById("odA"), inB = document.getElementById("odB"), dl = document.getElementById("odList");
  var presetSel = document.getElementById("odPreset"), speedIn = document.getElementById("odSpeed"), indoorIn = document.getElementById("odIndoor");
  var setOdTour = segBind("odTour", "tour", function (v) { od.names[od.tour] = [inA.value, inB.value]; od.tour = v; od.preset = ""; od.speed = 0; initOddsTour(); });
  var setOdSurf = segBind("odSurf", "surf", function (v) { od.surf = v; od.preset = ""; od.speed = 0; if (v !== "Hard") od.indoor = false; syncControls(); renderOdds(); });
  var setOdLvl = segBind("odLevel", "lvl", function (v) { od.lvl = v; od.preset = ""; od.bo = (v === "G" && od.tour === "atp") ? 5 : 3; syncControls(); renderOdds(); });
  var setOdBo = segBind("odBo", "bo", function (v) { od.bo = +v; renderOdds(); });
  indoorIn.addEventListener("change", function () { od.indoor = indoorIn.checked; od.preset = ""; syncControls(); renderOdds(); });
  speedIn.addEventListener("input", function () { od.speed = +speedIn.value; od.preset = ""; syncControls(); renderOdds(); });
  presetSel.addEventListener("change", function () {
    var D = ODDS[od.tour]; if (!D) return;
    var i = presetSel.value; od.preset = i;
    if (i !== "") {
      var pr = D._presets[+i];
      od.surf = pr.s; od.indoor = !!pr.i; od.speed = pr.v || 0;
      od.lvl = pr.l === "G" ? "G" : pr.l === "M" ? "M" : "A";
      od.bo = (od.lvl === "G" && od.tour === "atp") ? 5 : 3;
    }
    syncControls(); renderOdds();
  });
  document.getElementById("odSwap").addEventListener("click", function () { var t = inA.value; inA.value = inB.value; inB.value = t; renderOdds(); });
  // while typing, the suggestion list holds at most 5 matching players (best-rated first)
  function fillList(el) {
    var D = ODDS[od.tour]; if (!D) return;
    var q = el.value.trim().toLowerCase();
    var m = D.players.filter(function (p) { return !q || p.n.toLowerCase().indexOf(q) !== -1; }).slice(0, 5);
    dl.innerHTML = m.map(function (p) { return '<option value="' + esc(p.n) + '"></option>'; }).join("");
  }
  [inA, inB].forEach(function (el) {
    el.addEventListener("input", function () { fillList(el); renderOdds(); });
    el.addEventListener("focus", function () { fillList(el); });
    el.addEventListener("change", renderOdds);
  });

  function syncControls() {
    setOdSurf(od.surf); setOdLvl(od.lvl); setOdBo(od.bo); setOdTour(od.tour);
    indoorIn.checked = od.indoor;
    indoorIn.closest("label").classList.toggle("disabled", od.surf !== "Hard");
    speedIn.value = od.speed;
    presetSel.value = od.preset;
    var D = ODDS[od.tour];
    var hasSpeed = D && D.const.speed_range[1] > D.const.speed_range[0];
    document.getElementById("odSpeedField").hidden = !hasSpeed;
    document.getElementById("odLvlM").textContent = od.tour === "atp" ? "Masters 1000" : "WTA 1000";
    var v = od.speed;
    document.getElementById("odSpeedVal").textContent = Math.abs(v) < 0.005 ? "standard" : (v > 0 ? "+" : "−") + Math.round(Math.abs(Math.exp(v) - 1) * 100) + "% aces";
    var surfName = od.surf === "Hard" && od.indoor ? "indoor hard" : od.surf.toLowerCase();
    document.getElementById("odSpeedStd").textContent = "standard " + surfName + " court";
  }

  function initOddsTour() {
    loadOdds(od.tour).then(function (D) {
      var P = D.players;
      dl.innerHTML = P.slice(0, 5).map(function (p) { return '<option value="' + esc(p.n) + '"></option>'; }).join("");
      var names = od.names[od.tour] || [P[0].n, P[1].n];
      inA.value = names[0]; inB.value = names[1];
      var C = D.const;
      speedIn.min = C.speed_range[0]; speedIn.max = C.speed_range[1];
      if (!D._presets) {
        D._presets = D.presets.length ? D.presets : [
          { t: "Australian Open", s: "Hard", i: false, l: "G", v: 0 }, { t: "Roland Garros", s: "Clay", i: false, l: "G", v: 0 },
          { t: "Wimbledon", s: "Grass", i: false, l: "G", v: 0 }, { t: "US Open", s: "Hard", i: false, l: "G", v: 0 }];
      }
      var groups = { G: "Grand Slams", M: C.tour === "atp" ? "Masters 1000" : "WTA 1000", F: "Tour Finals", "500": "500 events" };
      var html = '<option value="">Custom: set surface and speed below</option>', lastL = null;
      D._presets.forEach(function (pr, i) {
        if (pr.l !== lastL) { if (lastL) html += "</optgroup>"; html += '<optgroup label="' + (groups[pr.l] || "Other") + '">'; lastL = pr.l; }
        var sp = pr.v ? " · " + (pr.v > 0 ? "+" : "−") + Math.round(Math.abs(Math.exp(pr.v) - 1) * 100) + "% aces" : "";
        html += '<option value="' + i + '">' + esc(pr.t) + " (" + (pr.i ? "indoor " : "") + pr.s.toLowerCase() + sp + ")</option>";
      });
      presetSel.innerHTML = html + (lastL ? "</optgroup>" : "");
      if (od.lvl === "G" && od.tour === "wta") od.bo = 3;
      od.inited = od.tour;
      syncControls(); renderOdds();
    });
  }

  function findP(D, name) { return D.byName[(name || "").trim().toLowerCase()]; }

  function compute(D, A, B, o) {
    var C = D.const, si = SURF.indexOf(o.surf);
    var both = A.ns[si] >= 1 && B.ns[si] >= 1, pool = D.fill_pool && D.fill_pool[o.surf];
    var ok = C.gap || both || pool;
    var ea = ok ? srEff(D, A, si) : [A.sv, A.rt], eb = ok ? srEff(D, B, si) : [B.sv, B.rt];
    var mu = C.mu[o.surf] != null ? C.mu[o.surf] : 0.55, k = C.hm / C.pt_m;
    var ha = clamp(sig(mu + k * (ea[0] - eb[1])), 0.02, 0.98), hb = clamp(sig(mu + k * (eb[0] - ea[1])), 0.02, 0.98);
    var qsRaw = clamp(0.5 * (srSet(r3(ha), r3(hb)) + 1 - srSet(r3(hb), r3(ha))), 1e-6, 1 - 1e-6);
    var zRating = C.CAL_I + C.CAL_S * logit(clamp(matchFromSet(qsRaw, o.bo), 1e-6, 1 - 1e-6));
    var qs = clamp(sig(logit(qsRaw) + A.c - B.c), 1e-6, 1 - 1e-6);
    var zClose = C.CAL_I + C.CAL_S * logit(clamp(matchFromSet(qs, o.bo), 1e-6, 1 - 1e-6));
    var zH = 0, rec = null;
    var key = A.i < B.i ? A.i + "_" + B.i : B.i + "_" + A.i;
    var hh = D.h2hMap[key];
    if (hh) {
      var aw = A.i < B.i ? hh[0] : hh[1], bw = A.i < B.i ? hh[1] : hh[0], n = aw + bw;
      rec = [aw, bw];
      var t = C.h2h_tbl, sl = n <= 0 ? 0 : n === 1 ? t["1"] : n === 2 ? t["2"] : n <= 4 ? t["3-4"] : t["5+"];
      zH = sl * logit(clamp((aw + 0.5) / (n + 1), 1e-6, 1 - 1e-6));
    }
    var fast = (o.surf === "Grass" || o.indoor) ? 1 : 0, big = (o.lvl === "G" || o.lvl === "M") ? 1 : 0;
    function side(Q) {
      var ag = Q.age != null ? Q.age : 26, ht = Q.ht || C.HT_MEAN;
      return { big_x_age27: Math.max(0, ag - 27) * big, height_x_fast: (ht - C.HT_MEAN) / 10 * fast, age_over32: C.age_cap ? Math.min(Math.max(0, ag - 32), C.age_cap) : Math.max(0, ag - 32),
               speedw_x_ace_gap: o.speed * Q.ace, deciding_set_clutch: Q.cl, slam_x_atp_top4: (o.lvl === "G" && Q.rk && Q.rk <= 4) ? 1 : 0 };
    }
    var fa = side(A), fb = side(B), bt = C.beta;
    function term(k) { return bt[k] * (fa[k] - fb[k]); }
    var parts = [
      { k: "rating", z: C.slope * zRating },
      { k: "closing", z: C.slope * (zClose - zRating) },
      { k: "h2h", z: C.slope * zH },
      { k: "clutch", z: term("deciding_set_clutch") },
      { k: "age", z: term("age_over32") + term("big_x_age27") },
      { k: "height", z: term("height_x_fast") },
      { k: "speed", z: term("speedw_x_ace_gap") },
      { k: "top4", z: term("slam_x_atp_top4") }
    ];
    var z = parts.reduce(function (s, p) { return s + p.z; }, 0), pDone = sig(z);
    // about 3% of matches end in a retirement or walkover, and then strength barely counts: the chance to ADVANCE
    var R = C.ret || { r_slam: 0, b_slam: 0, r_other: 0, b_other: 0 }, rr = o.lvl === "G" ? R.r_slam : R.r_other, bb = o.lvl === "G" ? R.b_slam : R.b_other;
    var pAdv = clamp((1 - rr) * pDone + rr * sig(bb * z), 1e-9, 1 - 1e-9);
    parts.push({ k: "ret", z: logit(pAdv) - z });
    return { p: pAdv, pDone: pDone, z: logit(pAdv), parts: parts, useS: both || !!pool, rec: rec, fast: fast, si: si, ea: ea, eb: eb, ha: ha, hb: hb, rr: rr };
  }

  function renderOdds() {
    if (panels.odds.hidden) return;
    var D = ODDS[od.tour];
    if (!D || od.inited !== od.tour) { initOddsTour(); return; }
    var C = D.const;
    var A = findP(D, inA.value), B = findP(D, inB.value);
    var nmA = document.getElementById("odNameA"), nmB = document.getElementById("odNameB");
    var brk = document.getElementById("odBreak"), sets = document.getElementById("odSets");
    if (!A || !B || A === B) {
      nmA.textContent = A ? A.n : (inA.value ? "“" + inA.value + "” isn’t an active player" : "Pick player 1");
      nmB.textContent = B ? (A === B ? "Pick a different player" : B.n) : (inB.value ? "“" + inB.value + "” isn’t an active player" : "Pick player 2");
      document.getElementById("odPA").textContent = "–"; document.getElementById("odPB").textContent = "–";
      document.getElementById("odBar").style.width = "50%";
      document.getElementById("odFairA").textContent = ""; document.getElementById("odFairB").textContent = "";
      brk.innerHTML = '<tr><td class="d">Choose two different players from the list (active in the past year, 30+ matches).</td></tr>'; sets.innerHTML = "";
      renderLive(null, null, null); return;
    }
    var o = { surf: od.surf, indoor: od.indoor && od.surf === "Hard", lvl: od.lvl, bo: od.bo, speed: C.speed_range[1] > C.speed_range[0] ? od.speed : 0 };
    var R = compute(D, A, B, o), p = R.p;
    nmA.textContent = A.n; nmB.textContent = B.n;
    document.getElementById("odPA").textContent = (p * 100).toFixed(1) + "%";
    document.getElementById("odPB").textContent = ((1 - p) * 100).toFixed(1) + "%";
    document.getElementById("odBar").style.width = (p * 100).toFixed(2) + "%";
    document.getElementById("odFairA").textContent = "fair odds " + (1 / p).toFixed(2);
    document.getElementById("odFairB").textContent = "fair odds " + (1 / (1 - p)).toFixed(2);

    // cumulative percentage-point shifts, in model order
    var sn = od.surf.toLowerCase(), si = R.si;
    var filled = [A, B].filter(function (x) { return x.ns[si] < C.MIN_SURF; }).map(function (x) { return x.n.split(" ").slice(-1)[0]; });
    function sgn(x, d) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d); }
    var desc = {
      rating: ["Serve & return on " + sn, "serve " + R.ea[0].toFixed(2) + " vs " + R.eb[0].toFixed(2) + ", return " + R.ea[1].toFixed(2) + " vs " + R.eb[1].toFixed(2) +
          " \u00b7 holds " + (hold(R.ha) * 100).toFixed(0) + "% vs " + (hold(R.hb) * 100).toFixed(0) + "% of service games" +
          (filled.length ? " \u00b7 " + filled.join(" and ") + " never played on " + sn + ": same as general" : "")],
      closing: ["Closing", "set-level finishing rating " + sgn(A.c, 2) + " vs " + sgn(B.c, 2)],
      h2h: ["Head-to-head", R.rec ? A.n.split(" ").slice(-1)[0] + " leads " + R.rec[0] + "–" + R.rec[1] : "no previous meetings"],
      clutch: ["Deciding-set clutch", sgn(A.cl, 3) + " vs " + sgn(B.cl, 3)],
      age: ["Age", (A.age != null ? A.age.toFixed(0) : "?") + " vs " + (B.age != null ? B.age.toFixed(0) : "?") + (o.lvl !== "A" ? " · older players do a little worse at big events" : "")],
      height: ["Height on a fast court", (A.ht || "?") + " vs " + (B.ht || "?") + " cm"],
      speed: ["Court speed × serve", "aces ×" + Math.exp(A.ace).toFixed(2) + " vs ×" + Math.exp(B.ace).toFixed(2) + " of average, on a " + (o.speed > 0 ? "faster" : "slower") + "-than-standard court"],
      top4: [(C.tour === "atp" ? "ATP" : "WTA") + " top-4 at a Slam", "elite players lift at Slams"],
      ret: ["Retirement or walkover risk", (R.rr * 100).toFixed(1) + "% of such matches end early, and then the result is close to a coin flip \u00b7 " + A.n.split(" ").slice(-1)[0] + " wins " + (R.pDone * 100).toFixed(1) + "% if the match is completed"],
      profile: ["Serve & return profile", A.sv !== undefined ? "serve " + sgn(A.sv, 2) + " vs " + sgn(B.sv, 2) + ", return " + sgn(A.rt, 2) + " vs " + sgn(B.rt, 2) + " \u00b7 the better server outperforms the points-based rating, most in close matches" : ""],
      sr: ["Serve & return", A.sv !== undefined ? "serve " + sgn(A.sv, 2) + " vs " + sgn(B.sv, 2) + ", return " + sgn(A.rt, 2) + " vs " + sgn(B.rt, 2) + " (separate serve and return ratings)" : ""]
    };
    var zc = 0, rows = "";
    R.parts.forEach(function (pt) {
      var always = pt.k === "rating" || pt.k === "closing" || pt.k === "h2h" || pt.k === "clutch" || pt.k === "ret";
      if (!always && Math.abs(pt.z) < 1e-9) return;
      var before = sig(zc); zc += pt.z; var d = (sig(zc) - before) * 100;
      var w = Math.min(Math.abs(d) / 30, 1) * 45;
      var bar = '<span class="effbar"><i class="' + (d < 0 ? "neg" : "") + '" style="' + (d >= 0 ? "left:50%" : "right:50%") + ";width:" + w.toFixed(1) + '%"></i></span>';
      rows += "<tr><td><strong>" + desc[pt.k][0] + '</strong><div class="d">' + esc(desc[pt.k][1]) + "</div></td><td>" + bar + '</td><td class="v">' + (Math.abs(d) < 0.05 ? "0.0" : sgn(d, 1)) + " pp</td></tr>";
    });
    rows += '<tr class="tot"><td>' + esc(A.n) + " advances</td><td></td><td class=\"v\">" + (p * 100).toFixed(1) + "%</td></tr>";
    brk.innerHTML = rows;

    // scorelines: per-set probability that reproduces the match probability
    var q = invMatch(p, o.bo), need = o.bo === 3 ? 2 : 3, out = [];
    for (var l = 0; l < need; l++) out.push({ s: need + "–" + l, a: true, pr: comb(need - 1 + l, l) * Math.pow(q, need) * Math.pow(1 - q, l) });
    for (var l2 = need - 1; l2 >= 0; l2--) out.push({ s: l2 + "–" + need, a: false, pr: comb(need - 1 + l2, l2) * Math.pow(1 - q, need) * Math.pow(q, l2) });
    sets.innerHTML = out.map(function (x) { return '<div class="ss' + (x.a ? " a" : "") + '"><div class="sc">' + x.s + '</div><div class="pc">' + (x.pr * 100).toFixed(0) + "%</div></div>"; }).join("");

    var neutral = "Everything else is set to neutral: both players rested, neither playing at home, both direct entrants, no recent retirement or long layoff.";
    document.getElementById("odSpeedNote").innerHTML = C.tour === "atp"
      ? "Court speed shifts the odds through the two players' <strong>ace skill</strong> (how many more aces than average they hit): a faster court helps the bigger server. The zero point is an average court of that surface; tournament presets use each event's measured speed over the last three seasons."
      : "Court speed shifts the odds through the two players' <strong>ace skill</strong>, as for the men (the weight came out almost identical when refit on WTA matches). The zero point is an average court of that surface; tournament presets use each event's measured speed over the last three seasons.";
    document.getElementById("odNote").innerHTML = "Ratings as of " + fmtDate(C.asof) + ". " +
      "The headline is the chance to <em>advance</em>: retirements and walkovers (about 3% of matches, close to coin flips) are included; the breakdown also gives the chance if the match is completed. The chain: each player's serve rating against the other's return rating on that surface gives the chance to win a point on serve → service holds → sets, with tiebreaks played point by point (plus closing) → match → calibration, then head-to-head and the validated corrections. " + neutral +
      " Scorelines assume each set is an independent draw at the same per-set chance, so the upset scorelines are slightly understated.";
    renderLive(R, A, B);
  }

  // ---------------- live (in-match) card ----------------
  var lv = { st: { sets: [], g: [0, 0], pt: [0, 0], srv: 0, base: { sets: [], g: [0, 0] }, trk: [0, 0, 0, 0], unt: false }, hist: [], fmtUser: false, R: null, A: null, B: null, k: 1 };
  segBind("lvForm", "k", function (v) { lv.k = +v; renderLive(); });
  var lvScore = document.getElementById("lvScore"), lvFinal = document.getElementById("lvFinal"), lvMsg = document.getElementById("lvMsg");
  function lvFmt() { return lv.fmtUser ? lvFinal.value : (od.lvl === "G" ? "tb10" : "tb7"); }
  var setLvSrv = segBind("lvSrv", "srv", function (v) {
    var st = lv.st; st.srv = +v; lv.hist = [];
    st.base = { sets: st.sets.map(function (x) { return x.slice(); }), g: st.g.slice() }; st.trk = [0, 0, 0, 0]; st.unt = st.pt[0] + st.pt[1] > 0;   // server changed by hand: start over from this scoreboard
    renderLive();
  });
  function lvSetState(st, keepText) { lv.st = st; setLvSrv(st.srv); if (!keepText) lvScore.value = lvText(st, od.bo, lvFmt()); }
  lvScore.addEventListener("input", function () {
    var r = lvParse(lvScore.value, od.bo, lvFmt(), lv.st.srv);
    if (r.err) { lvMsg.textContent = r.err; lvMsg.className = "lv-msg err"; return; }
    lv.hist = []; lvSetState(r.st, true); renderLive();
  });
  lvFinal.addEventListener("change", function () { lv.fmtUser = true; lvReparse(); renderLive(); });
  function lvPoint(w) {
    var won = lvWon(lv.st), need = (od.bo >> 1) + 1; if (won[0] === need || won[1] === need) return;
    lv.hist.push(lv.st); lvSetState(lvAdvance(lv.st, w, od.bo, lvFmt())); renderLive();
  }
  document.getElementById("lvPtA").addEventListener("click", function () { lvPoint(0); });
  document.getElementById("lvPtB").addEventListener("click", function () { lvPoint(1); });
  document.getElementById("lvUndo").addEventListener("click", function () { if (lv.hist.length) { lvSetState(lv.hist.pop()); renderLive(); } });
  document.getElementById("lvClear").addEventListener("click", function () { lv.hist = []; lvSetState({ sets: [], g: [0, 0], pt: [0, 0], srv: lv.st.srv, base: { sets: [], g: [0, 0] }, trk: [0, 0, 0, 0], unt: false }); lvScore.value = ""; renderLive(); });
  function lvReparse() {                        // best of / deciding-set format changed: re-read the typed score
    if (lvScore.value === lvText(lv.st, od.bo, lvFmt()) || (!lvScore.value.trim() && !lv.st.sets.length && !lv.st.g[0] && !lv.st.g[1] && !lv.st.pt[0] && !lv.st.pt[1])) return;   // unchanged: keep the point history
    if (!lvScore.value.trim()) { lv.st = { sets: [], g: [0, 0], pt: [0, 0], srv: lv.st.srv, base: { sets: [], g: [0, 0] }, trk: [0, 0, 0, 0], unt: false }; return; }
    var r = lvParse(lvScore.value, od.bo, lvFmt(), lv.st.srv);
    if (!r.err) lv.st = r.st;
  }

  function renderLive(R, A, B) {
    if (R !== undefined) { lv.R = R; lv.A = A; lv.B = B; lvReparse(); }
    R = lv.R; A = lv.A; B = lv.B;
    if (!lv.fmtUser) lvFinal.value = lvFmt();
    var bo = od.bo, fmt = lvFmt(), card = document.getElementById("odLive");
    var sa = A ? A.n.split(" ").slice(-1)[0] : "Player 1", sb = B ? B.n.split(" ").slice(-1)[0] : "Player 2";
    document.getElementById("lvSrvA").textContent = sa; document.getElementById("lvSrvB").textContent = sb;
    document.getElementById("lvPtA").textContent = "Point " + sa; document.getElementById("lvPtB").textContent = "Point " + sb;
    document.getElementById("lvUndo").disabled = !lv.hist.length;
    var check = lvParse(lvScore.value, bo, fmt, lv.st.srv);
    if (check.err && lvScore.value.trim()) { lvMsg.textContent = check.err; lvMsg.className = "lv-msg err"; }
    else { lvMsg.className = "lv-msg"; lvMsg.textContent = lvScore.value.trim() ? "" : "Empty = start of the match (0–0)."; }
    if (!R) { card.classList.add("dim"); return; }
    card.classList.remove("dim");
    var st = lv.st, tb = lvInTb(st, bo, fmt), won = lvWon(st), need = (bo >> 1) + 1, over = won[0] === need || won[1] === need;
    ["lvPtA", "lvPtB"].forEach(function (id) { document.getElementById(id).disabled = over; });
    // scoreboard
    var cols = st.sets.length + (over ? 0 : 1), html = "";
    [0, 1].forEach(function (s) {
      var nm = s === 0 ? A.n : B.n, row = '<tr><td class="n">' + esc(nm) + (!over && st.srv === s ? '<span class="dot" title="serving"></span>' : "") + "</td>";
      st.sets.forEach(function (x) { row += '<td class="' + (x[s] > x[1 - s] ? "w" : "") + '">' + x[s] + "</td>"; });
      if (!over) {
        row += '<td class="cur">' + st.g[s] + "</td>";
        var a = st.pt[0], b = st.pt[1], ptxt;
        if (tb) ptxt = String(st.pt[s]);
        else if (a >= 3 && b >= 3) ptxt = a === b ? "40" : ((s === 0) === (a > b) ? "AD" : "40");
        else ptxt = PTS[st.pt[s]];
        row += '<td class="pt">' + ptxt + "</td>";
      }
      html += row + "</tr>";
    });
    document.getElementById("lvBoard").innerHTML = html;
    document.getElementById("lvNameA").textContent = A.n; document.getElementById("lvNameB").textContent = B.n;
    var tau = IM_TAU[od.tour] || 0.2;
    function ev(x) { var e = lvEval(R, x, bo, fmt, tau); if (!e.over) e.p = e.pStatic + lv.k * (e.p - e.pStatic); return e; }   // Half / Off scale the form part
    var E = ev(st);
    var brk = document.getElementById("lvBreak"), nxt = document.getElementById("lvNext");
    if (E.over) {
      var wn = E.over === 1 ? A.n : B.n;
      document.getElementById("lvPA").textContent = E.over === 1 ? "100%" : "0%"; document.getElementById("lvPB").textContent = E.over === 1 ? "0%" : "100%";
      document.getElementById("lvBar").style.width = E.over === 1 ? "100%" : "0%";
      brk.innerHTML = '<tr><td class="d">Match over: ' + esc(wn) + " won " + st.sets.map(function (x) { return x.join("–"); }).join(" ") + ".</td></tr>";
      nxt.innerHTML = ""; return;
    }
    var p = E.p;
    document.getElementById("lvPA").textContent = (p * 100).toFixed(1) + "%"; document.getElementById("lvPB").textContent = ((1 - p) * 100).toFixed(1) + "%";
    document.getElementById("lvBar").style.width = (p * 100).toFixed(2) + "%";
    function pp(x) { return (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(1) + " pp"; }
    function bar(d) { var w = Math.min(Math.abs(d) * 100 / 30, 1) * 45; return '<span class="effbar"><i class="' + (d < 0 ? "neg" : "") + '" style="' + (d >= 0 ? "left:50%" : "right:50%") + ";width:" + w.toFixed(1) + '%"></i></span>'; }
    var f1 = (E.srv1t - E.srv1) * 100, f2 = (E.srv2t - E.srv2) * 100, started = st.sets.length || st.g[0] || st.g[1] || st.pt[0] || st.pt[1];
    var tr = E.trk || [0, 0, 0, 0], trTxt = tr[0] + tr[1] + tr[2] + tr[3] > 0 ? " · from the points stepped through: " + sa + " won " + tr[0] + " of " + (tr[0] + tr[1]) + " on serve, " + sb + " " + tr[2] + " of " + (tr[2] + tr[3]) : "";
    var formTxt = lv.k === 0 ? "switched off" : !started ? "nothing played yet" : (lv.k < 1 ? "at half strength; " : "") +
      "estimated from the score: " + sa + " wins " + (E.srv1t * 100).toFixed(1) + "% of points on serve (" + (E.srv1 * 100).toFixed(1) + "% expected before the match), " +
      sb + " " + (E.srv2t * 100).toFixed(1) + "% (" + (E.srv2 * 100).toFixed(1) + "%)" + trTxt;
    brk.innerHTML =
      "<tr><td><strong>Before the match</strong><div class=\"d\">chance if completed (the headline above also counts retirements)</div></td><td></td><td class=\"v\">" + (E.p0 * 100).toFixed(1) + "%</td></tr>" +
      "<tr><td><strong>The score</strong><div class=\"d\">exact chance from here if both play at their pre-match level</div></td><td>" + bar(E.pStatic - E.p0) + '</td><td class="v">' + pp(E.pStatic - E.p0) + "</td></tr>" +
      "<tr><td><strong>Form on the day</strong><div class=\"d\">" + esc(formTxt) + "</div></td><td>" + bar(p - E.pStatic) + '</td><td class="v">' + pp(p - E.pStatic) + "</td></tr>" +
      '<tr class="tot"><td>' + esc(A.n) + ' wins</td><td></td><td class="v">' + (p * 100).toFixed(1) + "%</td></tr>";
    var nA = ev(lvAdvance(st, 0, bo, fmt)), nB = ev(lvAdvance(st, 1, bo, fmt));
    function pv(e) { return e.over ? (e.over === 1 ? 100 : 0).toFixed(1) + "%" : (e.p * 100).toFixed(1) + "%"; }
    var server = tb ? (st.srv === 0 ? sa : sb) : (st.srv === 0 ? sa : sb);
    nxt.innerHTML = "<div>If " + esc(sa) + " wins it<b>" + pv(nA) + "</b></div><div>If " + esc(sb) + " wins it<b>" + pv(nB) + "</b></div>";
    document.getElementById("lvNote").innerHTML = "<strong>" + esc(server) + " is serving.</strong> " +
      "The live chance plays out the rest of the match exactly, point by point, from this score (games, tiebreaks and the deciding-set format), with each player's chance to win a point on serve from the ratings. " +
      "The score also says something about how well both are playing today, so the model updates that: before the match it allows for a typical day-to-day swing in form. Points you step through with the buttons all count (a love hold says more than a hold from deuce); for a typed score only the completed sets and games are known, and inside a typed game the points decide the chance to win that game, whose result then updates the form. " +
      "Tested on 9,500 point-by-point charted matches (Match Charting Project), this form update clearly beats keeping strength fixed: with fixed strength, a favourite broken early or a set down is rated far too high. " +
      "Its speed is fitted on those matches. In typical situations it lands within 2 points of what actually happened (favourites broken in the first game won 61% for men, 61.5% predicted; 64% for women, 61% predicted). <em>Half</em> and <em>Off</em> scale the form part down if you want to see how much of the swing it explains. " +
      "There is no separate momentum term: recent sets were no more telling than earlier ones, and having just won a set added nothing beyond what the score already says about form. " +
      "Shown as the chance to win if the match is completed.";
  }

  // ---------------- player profile tab ----------------
  var pf = { tour: "atp", name: null, recs: null, year: "", surf: "", all: false };
  var pfSearch = document.getElementById("pfSearch"), pfSugg = document.getElementById("pfSugg"), pfHead = document.getElementById("pfHead");
  var pfYear = document.getElementById("pfYear"), pfMore = document.getElementById("pfMore");
  var setPfTour = segBind("pfTour", "tour", function (v) { pf.tour = v; pf.name = null; pf.recs = null; pfSearch.value = ""; renderProfile(); });
  var setPfSurf = segBind("pfSurf", "surf", function (v) { pf.surf = v; pf.all = false; renderMatches(); });
  pfYear.addEventListener("change", function () { pf.year = pfYear.value; pf.all = false; renderMatches(); });
  pfMore.addEventListener("click", function () { pf.all = !pf.all; renderMatches(); });
  function pfIndex(tour) {
    return load("prof_" + tour + "_index.json").then(function (d) {
      if (!d._by) { d._by = {}; d.players.forEach(function (x) { d._by[x[0].toLowerCase()] = x; }); }
      return d;
    });
  }
  var sugIdx = -1;
  function suggest() {
    var q = pfSearch.value.trim().toLowerCase();
    if (!q) { pfSugg.hidden = true; return; }
    pfIndex(pf.tour).then(function (d) {
      var m = d.players.filter(function (x) { return x[0].toLowerCase().indexOf(q) !== -1; }).slice(0, 5);   // max 5 suggestions
      sugIdx = -1;
      pfSugg.innerHTML = m.length ? m.map(function (x) { return '<button data-n="' + esc(x[0]) + '"><span>' + esc(x[0]) + '</span><span class="ct">' + x[2] + " matches · " + x[3].slice(0, 4) + "</span></button>"; }).join("")
        : '<div class="pf-empty" style="padding:8px 11px">No player with that name</div>';
      pfSugg.hidden = false;
      Array.prototype.forEach.call(pfSugg.querySelectorAll("button"), function (b) {
        b.addEventListener("mousedown", function (e) { e.preventDefault(); openProfile(pf.tour, b.dataset.n); });
      });
    });
  }
  pfSearch.addEventListener("input", suggest);
  pfSearch.addEventListener("focus", suggest);
  pfSearch.addEventListener("blur", function () { setTimeout(function () { pfSugg.hidden = true; }, 120); });
  pfSearch.addEventListener("keydown", function (e) {
    var bs = pfSugg.querySelectorAll("button"); if (!bs.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); sugIdx = (sugIdx + (e.key === "ArrowDown" ? 1 : bs.length - 1)) % bs.length;
      Array.prototype.forEach.call(bs, function (b, i) { b.classList.toggle("on", i === sugIdx); });
    } else if (e.key === "Enter") { openProfile(pf.tour, bs[Math.max(sugIdx, 0)].dataset.n); }
  });

  function openProfile(tour, name) {
    pf.tour = tour; setPfTour(tour); pf.year = ""; pf.surf = ""; setPfSurf(""); pf.all = false;
    pfSugg.hidden = true; pfSearch.value = name; pfSearch.blur();
    if (panels.profile.hidden) { pf.name = name; showTab("profile"); return; }
    pfHead.innerHTML = '<div class="pf-empty">Loading ' + esc(name) + '&hellip;</div>';
    pfIndex(tour).then(function (d) {
      var x = d._by[name.toLowerCase()];
      if (!x) { pfHead.innerHTML = '<div class="pf-empty">' + esc(name) + ' has fewer than 10 rated matches in the data.</div>'; hideCards(); return; }
      return load("prof_" + tour + "_" + x[1] + ".json").then(function (part) {
        pf.name = x[0]; pf.recs = part[x[0]] || []; pf.idx = x; renderProfileBody();
      });
    }).catch(function (e) { pfHead.innerHTML = '<div class="pf-empty">Couldn’t load the profile: ' + esc(e.message) + "</div>"; });
    try { history.replaceState(null, "", "#profile"); } catch (e) {}
  }
  window.__openProfile = openProfile;
  function hideCards() { document.getElementById("pfSeasonsCard").hidden = true; document.getElementById("pfMatchesCard").hidden = true; }
  function renderProfile() {
    if (panels.profile.hidden) return;
    if (pf.name && !pf.recs) { openProfile(pf.tour, pf.name); return; }
    if (!pf.name) {
      hideCards();
      pfHead.innerHTML = '<div class="pf-empty">Type a name above to open a player’s profile' +
        (pf.tour === "atp" ? " — e.g. <span class=\"opp\" data-n=\"Jannik Sinner\">Jannik Sinner</span> or <span class=\"opp\" data-n=\"Carlos Alcaraz\">Carlos Alcaraz</span>" : " — e.g. <span class=\"opp\" data-n=\"Aryna Sabalenka\">Aryna Sabalenka</span> or <span class=\"opp\" data-n=\"Iga Swiatek\">Iga Swiatek</span>") +
        ". You can also click any name in the Current ratings table.</div>";
      Array.prototype.forEach.call(pfHead.querySelectorAll(".opp"), function (s) { s.addEventListener("click", function () { openProfile(pf.tour, s.dataset.n); }); });
      return;
    }
    renderProfileBody();
  }
  // R: [date, tournament, level, surface, round, opponent, won, score, win_chance, level_before, dLevel, dServe, dReturn, dSurface, oppLevel, servePct, returnPct, rated]
  function flipScore(s) {
    return s.split(" ").map(function (t) { var m = /^(\d+)-(\d+)(\(\d+\))?$/.exec(t); return m ? m[2] + "-" + m[1] + (m[3] || "") : t; }).join(" ");
  }
  function sgn(x, d) { var a = Math.abs(x).toFixed(d); return (+a === 0 ? "" : x > 0 ? "+" : "−") + a; }
  function renderProfileBody() {
    var R = pf.recs, rated = R.filter(function (r) { return r[17]; }), full = R.filter(function (r) { return r[17] === 1; });
    var w = R.filter(function (r) { return r[6]; }).length, l = R.length - w;
    var peak = null;
    rated.forEach(function (r) { var v = r[9] + r[10]; if (!peak || v > peak.v) peak = { v: v, d: r[0] }; });
    var last = rated[rated.length - 1], now = last ? last[9] + last[10] : null;
    var tile = function (k, v, s) { return '<div class="pf-tile"><div class="k">' + k + '</div><div class="v">' + v + '</div>' + (s ? '<div class="s">' + s + "</div>" : "") + "</div>"; };
    var first = R.length ? R[0][0] : "", lastD = R.length ? R[R.length - 1][0] : "";
    var html = '<div class="pf-name">' + esc(pf.name) + '</div><div class="pf-sub">' + R.length + " matches in the data (" + full.length + " rated) · " + first.slice(0, 4) + "–" + lastD.slice(0, 4) + " · " + w + " wins, " + l + " losses</div>";
    loadOdds(pf.tour).then(function (D) {
      var P = findP(D, pf.name), tiles = "";
      if (P) {
        var se = [0, 1, 2].map(function (i) { return blendOf(D, P, i); });
        tiles += tile("Rating now", P.g.toFixed(2), "serve " + P.sv.toFixed(2) + " · return " + P.rt.toFixed(2));
        tiles += tile("Hard / Clay / Grass", se.map(function (v) { return v.toFixed(2); }).join(" / "), "general + surface gap");
        tiles += tile("Closing · Clutch", sgn(P.c, 2) + " · " + sgn(P.cl, 3), "set finishing · deciding sets");
      } else if (now != null) tiles += tile("Rating after last match", now.toFixed(2), "not active in the past year");
      if (peak) tiles += tile("Career peak", peak.v.toFixed(2), fmtDate(peak.d));
      var X = pf.idx;   // [.., slam entries, slams won, slams predicted, events, titles, titles predicted]
      if (X && X.length >= 11) {
        tiles += tile("Grand Slams", X[6] + " won", (X[7]).toFixed(1) + " predicted · " + X[5] + " played");
        tiles += tile("All titles", X[9] + " won", (X[10]).toFixed(1) + " predicted · " + X[8] + " events");
      }
      // upsets: the win the model rated least likely, and the loss it rated most likely to be a win
      var wins = rated.filter(function (r) { return r[6] && r[8] != null; }), losses = rated.filter(function (r) { return !r[6] && r[8] != null; });
      var up = wins.sort(function (a, b) { return a[8] - b[8]; })[0], dn = losses.sort(function (a, b) { return b[8] - a[8]; })[0];
      var ev = function (r) { return esc(r[5]) + ", " + esc(r[1]) + " " + r[0].slice(0, 4); };
      if (up) tiles += tile("Biggest upset win", (up[8] * 100).toFixed(0) + "%", "win chance vs " + ev(up));
      if (dn) tiles += tile("Biggest upset loss", (dn[8] * 100).toFixed(0) + "%", "win chance vs " + ev(dn));
      pfHead.innerHTML = html + '<div class="pf-tiles">' + tiles + "</div>";
    });
    pfHead.innerHTML = html;
    // seasons
    var ys = {}, order = [];
    R.forEach(function (r) {
      var y = r[0].slice(0, 4); if (!ys[y]) { ys[y] = { w: 0, l: 0, d: 0, ds: 0, dr: 0, end: null, n: 0 }; order.push(y); }
      var s = ys[y]; if (r[6]) s.w++; else s.l++;
      if (r[17]) { s.d += r[10]; s.ds += r[11]; s.dr += r[12]; s.end = r[9] + r[10]; s.n++; }
    });
    var cls = function (v) { return Math.abs(v) < 0.0005 ? "num" : v > 0 ? "num up" : "num dn"; };
    document.getElementById("pfSeasons").innerHTML = "<thead><tr><th class=\"l\"><button>Season</button></th><th><button>Won–lost</button></th><th><button>Rating at end</button></th><th><button>Δ Rating</button></th><th><button>Δ Serve</button></th><th><button>Δ Return</button></th></tr></thead><tbody>" +
      order.slice().reverse().map(function (y) {
        var s = ys[y];
        return '<tr><td class="l">' + y + '</td><td class="num">' + s.w + "–" + s.l + '</td><td class="num">' + (s.end != null ? s.end.toFixed(3) : "–") +
          '</td><td class="' + cls(s.d) + '">' + (s.n ? sgn(s.d, 3) : "–") + '</td><td class="' + cls(s.ds) + '">' + (s.n ? sgn(s.ds, 3) : "–") + '</td><td class="' + cls(s.dr) + '">' + (s.n ? sgn(s.dr, 3) : "–") + "</td></tr>";
      }).join("") + "</tbody>";
    document.getElementById("pfSeasonsCard").hidden = false;
    pfYear.innerHTML = '<option value="">All seasons</option>' + order.slice().reverse().map(function (y) { return '<option value="' + y + '">' + y + "</option>"; }).join("");
    pfYear.value = pf.year;
    renderMatches();
  }
  var RND = { F: "Final", SF: "Semi", QF: "Quarter", RR: "Round robin", BR: "Bronze" };
  function renderMatches() {
    var R = pf.recs; if (!R) return;
    var list = R.filter(function (r) { return (!pf.year || r[0].slice(0, 4) === pf.year) && (!pf.surf || r[3] === pf.surf); }).slice().reverse();
    var lim = pf.all ? list.length : 60;
    var cls = function (v) { return v == null || Math.abs(v) < 0.0005 ? "num" : v > 0 ? "num up" : "num dn"; };
    var f = function (v, d, s) { return v == null ? "–" : (s ? sgn(v, d) : v.toFixed(d)); };
    var head = "<thead><tr>" + [["Date", "l"], ["Tournament", "l"], ["Round", "l"], ["Surface", "l"], ["Opponent", "l"], ["Opp. rating"], ["Result", "l"], ["Win chance"], ["Rating after"], ["Δ Rating"], ["Δ Serve"], ["Δ Return"], ["Δ Surface"], ["Serve pts"], ["Return pts"]]
      .map(function (c) { return '<th class="' + (c[1] || "") + '"><button tabindex="-1">' + c[0] + "</button></th>"; }).join("") + "</tr></thead>";
    var body = list.slice(0, lim).map(function (r) {
      var won = r[6], sc = won ? r[7] : flipScore(r[7]), rated = r[17];
      return "<tr" + (!rated ? ' title="Not rated: walkover, default, or retirement without match stats"' : rated === 2 ? ' title="Retirement: only the player who did not retire learns from the points played"' : "") + '><td class="l num">' + r[0] + '</td><td class="l">' + esc(r[1]) + '</td><td class="l">' + (RND[r[4]] || r[4]) +
        '</td><td class="l">' + (r[3] || "–") + '</td><td class="l"><span class="opp" data-n="' + esc(r[5]) + '">' + esc(r[5]) + '</span></td><td class="num">' + f(r[14], 2) +
        '</td><td class="l"><span class="res" style="color:' + (won ? "var(--wm)" : "var(--accent)") + '">' + (won ? "W" : "L") + "</span> " + esc(sc) +
        '</td><td class="num">' + (r[8] == null ? "–" : (r[8] * 100).toFixed(0) + "%") + '</td><td class="num">' + (rated ? (r[9] + r[10]).toFixed(3) : "–") +
        '</td><td class="' + cls(r[10]) + '">' + f(r[10], 3, 1) + '</td><td class="' + cls(r[11]) + '">' + f(r[11], 3, 1) + '</td><td class="' + cls(r[12]) + '">' + f(r[12], 3, 1) +
        '</td><td class="' + cls(r[13]) + '">' + f(r[13], 3, 1) + '</td><td class="num">' + (r[15] == null ? "–" : r[15].toFixed(0) + "%") + '</td><td class="num">' + (r[16] == null ? "–" : r[16].toFixed(0) + "%") + "</td></tr>";
    }).join("");
    document.getElementById("pfMatches").innerHTML = head + "<tbody>" + (body || '<tr><td class="l" colspan="15">No matches for this filter.</td></tr>') + "</tbody>";
    Array.prototype.forEach.call(document.querySelectorAll("#pfMatches .opp"), function (s) { s.addEventListener("click", function () { openProfile(pf.tour, s.dataset.n); window.scrollTo(0, 0); }); });
    pfMore.hidden = list.length <= 60; pfMore.textContent = pf.all ? "Show latest 60" : "Show all " + list.length;
    document.getElementById("pfMatchesCard").hidden = false;
  }

  // ---------------- boot ----------------
  var start = (location.hash || "").replace("#", "");
  if (["men", "women", "rankings", "peaks", "odds", "profile"].indexOf(start) < 0) start = "men";
  if (start !== "men") load("traj_atp.json");   // warm the default chart in the background
  showTab(start);
  window.__oddsCompute = function (tour, a, b, o) { var D = ODDS[tour]; return compute(D, findP(D, a), findP(D, b), o).p; };
  window.__loadOdds = loadOdds;
})();
