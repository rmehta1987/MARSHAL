/* MARSHAL monitor display.
 *
 * Arena: plays saved games (from /api/games, or window.MARSHAL_EMBED in an
 * exported file) move by move: each model's reasoning types out, then its move
 * lands on the board. Live games stream from /api/live/stream.
 * Results: paged slides built from /api/results: what MARSHAL is, the Arena
 * evaluation (base Qwen3-4B vs the MARSHAL checkpoint), the bring-up runs, their
 * training curves, and what the numbers mean.
 * In tour mode (the default) the screen alternates a game and a results page.
 */
(() => {
  "use strict";

  // ------------------------------------------------------------ constants
  const params = new URLSearchParams(location.search);
  const EMBED = window.MARSHAL_EMBED || null;
  const HOLD = { intro: 6500, answer: 1100, move: 1700, bot: 900, end: 14000, slide: 21000 };
  const CPS = 110; // reasoning typed at ~110 characters a second at 1x
  const MIN_TYPE = 1200;
  const MAX_TYPE = 12000;
  const IDLE_RESUME = 180000; // a person touched it: resume the tour after 3 minutes
  const LIVE_RETURN = 60000; // after a live game ends, return to replays
  const SLIDES = [
    { key: "overview", label: "What MARSHAL is" },
    { key: "arena", label: "Arena results" },
    { key: "decrypto", label: "MARSHAL in Decrypto" },
    { key: "runs", label: "Bring-up runs" },
    { key: "curves", label: "Training curves" },
    { key: "findings", label: "What the numbers say" },
    { key: "paper", label: "vs the paper" },
  ];
  const BLURB = {
    tictactoe: "Three in a row on a 3×3 grid. One of MARSHAL's three self-play training games.",
    kuhn_poker: "Three cards (J < Q < K), one each, one round of betting: bluffing and reading the other player's bets. A MARSHAL training game.",
    hanabi: "Both players win or lose together. Each sees the partner's cards but not their own, and spends hints to build the fireworks. A MARSHAL training game.",
    connect_four: "Four in a row on a 6×7 grid. Held out: MARSHAL never trains on it.",
    leduc_poker: "Six cards, a private card each, a public card and two betting rounds. Held out: MARSHAL never trains on it.",
  };
  const OPPONENT_NOTE = {
    "MCTS bot": "Monte Carlo tree search: plays out 100 random games per move and picks the best. No reasoning to show.",
    "CFR equilibrium bot": "Plays a precomputed equilibrium strategy for Kuhn Poker (counterfactual regret minimization).",
    "Nash-equilibrium bot": "Plays a precomputed Nash-equilibrium strategy for Leduc Hold'em.",
  };

  const S = {
    scene: params.get("scene") === "results" ? "results" : "arena",
    mode: "replay",
    tour: params.get("tour") !== "0",
    playing: params.get("pause") !== "1",
    speed: clampSpeed(parseFloat(params.get("speed")) || 1),
    list: [], playlist: [], pos: 0, cache: {},
    rec: null, applied: 0, typing: null, phase: "idle",
    clock: 0, last: 0, tasks: [], gen: 0,
    results: null, slide: Math.max(0, SLIDES.findIndex((s) => s.key === params.get("slide"))),
    slidesShown: 0, curvesRun: null,
    live: { enabled: false, options: null, snap: null, es: null, returnAt: 0 },
    idleUntil: 0, footerHidden: false, showPrompt: true, filterIds: null,
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function clampSpeed(v) { return Math.min(4, Math.max(0.25, v)); }
  const minus = (s) => String(s).replace(/^-/, "−");
  function fmtNum(v, digits = 2) {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    let s;
    if (a >= 1000) s = Math.round(v).toLocaleString("en-US");
    else if (a >= 100) s = v.toFixed(0);
    else if (a >= 10) s = v.toFixed(1);
    else if (a === 0) s = "0";
    else if (a < 0.001) s = v.toExponential(1);
    else s = String(+v.toFixed(a < 0.1 ? 4 : digits));
    return minus(s);
  }
  const fmtSigned = (v) => (v > 0 ? "+" : "") + fmtNum(v, 2);
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const svgX = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5.5 5.5l13 13M18.5 5.5l-13 13" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round"/></svg>';
  const svgO = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7.2" fill="none" stroke="currentColor" stroke-width="3.2"/></svg>';
  const seatVar = (p) => `var(--seat${p})`;
  const asset = (path) => (EMBED && EMBED.assets && EMBED.assets[path]) || path;

  // ------------------------------------------------------------ tooltip
  const tip = $("tooltip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - r.width - 8, x + 14) + "px";
    tip.style.top = Math.max(8, Math.min(window.innerHeight - r.height - 8, y - r.height - 10)) + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  // ------------------------------------------------------------ data
  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }
  async function loadList() {
    if (EMBED) return EMBED.games.map(summaryOf);
    return (await getJSON("/api/games")).games;
  }
  async function loadGame(id) {
    if (S.cache[id]) return S.cache[id];
    const rec = EMBED ? EMBED.games.find((g) => g.id === id) : await getJSON(`/api/game?id=${encodeURIComponent(id)}`);
    S.cache[id] = rec;
    return rec;
  }
  async function loadResults() {
    if (EMBED) return EMBED.results;
    return getJSON("/api/results");
  }
  function summaryOf(r) {
    return { id: r.id, game: r.game, title: r.title, split: r.split, source: r.source, started: r.started,
      moves: r.moves.length, players: r.players, result: r.result, settings: r.settings };
  }

  // ------------------------------------------------------------ actions & text
  function pretty(game, action) {
    if (!action) return "—";
    if (game === "tictactoe") { const m = action.match(/\((\d),(\d)\)/); return m ? `(${m[1]},${m[2]})` : action; }
    if (game === "connect_four") { const m = action.match(/[XO]\((\d)\)>*\)?>?$/) || action.match(/\((\d)\)/); return m ? `column ${m[1]}` : action; }
    if (game === "kuhn_poker" || game === "leduc_poker") { const a = action.replace(/[<>]/g, ""); return a.charAt(0) + a.slice(1).toLowerCase(); }
    if (game === "hanabi") {
      const a = action.replace(/[<>]/g, "");
      let m = a.match(/Reveal player \+1 color (\w)/); if (m) return `hint: color ${m[1]}`;
      m = a.match(/Reveal player \+1 rank (\d)/); if (m) return `hint: rank ${m[1]}`;
      return a.replace(/^Play card/, "play card").replace(/^Discard card/, "discard card");
    }
    return action;
  }
  function highlight(text, legal) {
    let out = esc(text);
    const tokens = (legal || []).map(esc).filter(Boolean).sort((a, b) => b.length - a.length);
    if (tokens.length) {
      const re = new RegExp(tokens.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g");
      out = out.replace(re, (m) => `<mark>${m}</mark>`);
    }
    return out.replace(/&lt;\/?(think|answer)&gt;/g, (m) => `<mark>${m}</mark>`);
  }
  function playerName(rec, p) { return rec.players[p]?.name || `Player ${p}`; }
  function seatLabel(rec, p) {
    if (rec.game === "tictactoe" || rec.game === "connect_four") return `${p === 0 ? "X" : "O"} · ${p === 0 ? "moves first" : "moves second"}`;
    if (rec.game === "hanabi") return `Player ${p} · ${p === 0 ? "starts" : "partner"}`;
    return `Player ${p} · ${p === 0 ? "acts first" : "acts second"}`;
  }
  function seatMark(rec, p) {
    if (rec.game === "tictactoe" || rec.game === "connect_four") return p === 0 ? svgX : svgO;
    return `P${p}`;
  }
  function forfeitWhy(rec, m) {
    const limit = rec.settings?.max_new_tokens;
    if (m.forfeit === "overlong_response") return `it was still thinking at ${m.tokens.toLocaleString("en-US")} tokens, and 4096 is the limit`;
    if (m.forfeit === "overlong_sequence") return `fewer than 1000 tokens were left in the ${rec.settings?.sequence_length}-token game`;
    if (m.by === "model" && limit && m.tokens >= limit - 1) return `it was still thinking when the ${limit}-token answer limit cut it off`;
    if (m.answer) return `"${m.answer}" isn't one legal move`;
    return "its answer didn't end with <answer>…</answer>";
  }
  function bigOpponent(name) {
    const key = Object.keys(OPPONENT_NOTE).find((k) => (name || "").startsWith(k));
    return key ? OPPONENT_NOTE[key] : "";
  }
  const hasHuman = (rec) => rec.players.some((p) => p.kind === "human");
  const isSecretGame = (rec) => ["kuhn_poker", "leduc_poker", "hanabi"].includes(rec.game);

  // ------------------------------------------------------------ clock & tasks
  function after(ms, fn) { const g = S.gen; S.tasks.push({ at: S.clock + ms, fn, g }); }
  function cancelTasks() { S.gen += 1; S.tasks = []; S.typing = null; }
  function frame(now) {
    const dt = S.last ? Math.min(250, now - S.last) : 0;
    S.last = now;
    if (S.playing) S.clock += dt * S.speed;
    const due = S.tasks.filter((t) => t.at <= S.clock && t.g === S.gen);
    S.tasks = S.tasks.filter((t) => t.at > S.clock && t.g === S.gen);
    for (const t of due) t.fn();
    if (S.typing) updateTyping();
    if (S.tour && S.mode === "replay" && Date.now() > S.idleUntil && S.scene === "results" && S.phase === "slide" && S.clock >= S.slideUntil) {
      nextAfterSlide();
    }
    if (S.mode === "live-done" && Date.now() > S.live.returnAt) backToReplays();
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------ scenes
  function setScene(scene) {
    S.scene = scene;
    $("scene-arena").hidden = scene !== "arena";
    $("scene-results").hidden = scene !== "results";
    document.querySelectorAll(".scene-tab").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.scene === scene)));
    if (scene === "results") renderSlide();
    updateContext();
  }
  function touched() { S.idleUntil = Date.now() + IDLE_RESUME; }
  function goResults() {
    if (S.mode === "replay") cancelTasks();
    hideOverlays();
    S.phase = "slide";
    S.slidesShown = 0;
    S.slideUntil = S.clock + HOLD.slide;
    setScene("results");
  }
  function goArena() {
    setScene("arena");
    if (S.mode === "replay" && (S.phase === "slide" || S.phase === "idle" || !S.rec)) playNext(0);
  }

  function updateContext() {
    const ctx = $("context");
    const badge = $("mode-badge");
    badge.classList.toggle("live", S.mode === "live");
    if (S.mode === "live" || S.mode === "live-done") badge.textContent = S.mode === "live" ? "Live" : "Live · finished";
    else badge.textContent = S.playing ? (S.speed !== 1 ? `Replay · ${S.speed}×` : "Replay") : "Paused";
    if (S.scene === "results") { ctx.innerHTML = `<b>Results</b> · ${esc(SLIDES[S.slide].label)}`; return; }
    if (!S.rec) { ctx.textContent = ""; return; }
    const r = S.rec;
    const tag = r.source === "eval" ? "Arena evaluation game" : r.source === "bringup-settings" ? "Bring-up limits (1024-token answers)" : r.source === "live" ? "Played live" : "Recorded game";
    ctx.innerHTML = `<b>${esc(playerName(r, 0))}</b> vs <b>${esc(playerName(r, 1))}</b> · ${esc(tag)}`;
  }

  // ------------------------------------------------------------ playlist
  async function refreshList() {
    S.list = await loadList();
    const only = params.get("game");
    const filter = (params.get("filter") || "").toLowerCase();
    let pool = S.list.filter((g) => !only || g.id === only);
    if (filter) pool = pool.filter((g) => matches(g, filter));
    if (!pool.length) pool = S.list.slice();
    S.playlist = params.get("shuffle") === "0" ? pool : varied(pool);
    S.pos = 0;
  }
  function matches(g, f) {
    const hay = [g.title, g.game, g.source, ...g.players.map((p) => `${p.name} ${p.model || ""}`)].join(" ").toLowerCase();
    return f.split(/\s+/).every((w) => hay.includes(w));
  }
  function varied(pool) {
    // Rotate through games so the loop doesn't play twelve poker hands in a row.
    const by = {};
    for (const g of shuffle(pool.slice())) (by[g.game] = by[g.game] || []).push(g);
    const keys = shuffle(Object.keys(by));
    const out = [];
    while (keys.some((k) => by[k].length)) for (const k of keys) if (by[k].length) out.push(by[k].pop());
    return out;
  }
  async function playNext(offset = 1) {
    if (!S.playlist.length) await refreshList();
    if (!S.playlist.length) { showEmpty(); return; }
    S.pos = (S.pos + offset + S.playlist.length) % S.playlist.length;
    await startReplay(S.playlist[S.pos].id);
  }
  function showEmpty() {
    $("game-title").textContent = "No saved games yet";
    $("narration").innerHTML = 'Record some with <code>python -m display.record</code> or start a live game.';
  }

  // ------------------------------------------------------------ replay
  async function startReplay(id) {
    cancelTasks();
    hideOverlays();
    const rec = await loadGame(id);
    S.mode = "replay";
    S.rec = rec;
    S.applied = 0;
    S.phase = "intro";
    setScene("arena");
    renderArena();
    showIntro(rec);
    after(HOLD.intro, () => { hide("intro"); S.phase = "play"; stepMove(); });
    updateContext();
  }
  function stepMove() {
    const rec = S.rec;
    if (S.applied >= rec.moves.length) { finishReplay(); return; }
    const m = rec.moves[S.applied];
    if (m.by === "bot") {
      S.typing = { seat: m.player, i: S.applied, text: "", dur: 1, start: S.clock, bot: true };
      renderArena();
      after(HOLD.bot, () => applyMove());
      return;
    }
    const text = (m.think || "").trim() || (m.by === "human" ? "" : (m.raw || ""));
    const dur = Math.min(MAX_TYPE, Math.max(MIN_TYPE, (text.length / CPS) * 1000));
    S.typing = { seat: m.player, i: S.applied, text, dur, start: S.clock, shown: -1 };
    renderArena();
    after(dur, () => {
      S.typing.done = true;
      updateTyping(true);
      after(HOLD.answer, () => applyMove());
    });
  }
  function applyMove() {
    S.typing = null;
    S.applied += 1;
    renderArena();
    after(HOLD.move, () => stepMove());
  }
  function finishReplay() {
    S.phase = "end";
    renderArena();
    showEnd(S.rec);
    after(HOLD.end, () => {
      hide("end");
      if (S.tour && S.mode === "replay" && Date.now() > S.idleUntil && S.results) {
        S.slide = stepSlide(S.slide, 1);
        goResults();
      } else playNext(1);
    });
  }
  function nextAfterSlide() {
    S.slidesShown += 1;
    if (S.slidesShown >= 2) { S.phase = "idle"; setScene("arena"); playNext(1); return; }
    S.slide = stepSlide(S.slide, 1);
    S.slideUntil = S.clock + HOLD.slide;
    renderSlide();
    updateContext();
  }
  function stepManual(delta) {
    touched();
    if (S.scene === "results") { gotoSlide(S.slide + delta); return; }
    if (!S.rec || S.mode !== "replay") return;
    cancelTasks();
    hideOverlays();
    S.playing = false;
    S.applied = Math.max(0, Math.min(S.rec.moves.length, S.applied + delta));
    S.phase = "manual";
    renderArena();
    if (S.applied === S.rec.moves.length) showEnd(S.rec);
    updateContext();
  }

  // ------------------------------------------------------------ arena rendering
  function currentView(rec, applied) {
    if (applied <= 0) return rec.initial?.view || {};
    return rec.moves[applied - 1].view || rec.initial?.view || {};
  }
  function lastMoveBy(rec, applied, p) {
    for (let i = applied - 1; i >= 0; i--) if (rec.moves[i].player === p) return rec.moves[i];
    return null;
  }
  function renderArena() {
    const rec = S.rec;
    if (!rec) return;
    $("game-title").textContent = rec.title;
    const split = $("game-split");
    split.textContent = rec.split === "held-out" ? "Held-out game" : "Training game";
    split.className = "split " + (rec.split || "");
    $("game-family").textContent = rec.family || "";
    const pendingSeat = S.mode.startsWith("live") ? livePendingSeat() : S.typing ? S.typing.seat : null;
    for (const p of [0, 1]) renderSeat(rec, p, pendingSeat === p);
    renderBoard(rec);
    renderNarration(rec);
    renderStrip(rec);
    renderPrompt(rec);
    const n = rec.moves.length || 1;
    $("progress-bar").style.width = `${Math.round((applied() / n) * 100)}%`;
    $("source").textContent = `${rec.id}${rec.seed != null ? ` · seed ${rec.seed}` : ""}${rec.settings ? ` · ${rec.settings.max_new_tokens}-token answers, ${rec.settings.sequence_length}-token games` : ""}`;
  }
  const applied = () => (S.mode.startsWith("live") ? (S.rec?.moves.length || 0) : S.applied);

  function hiddenFromViewer(rec, p) {
    // A person at the screen mustn't see what their seat couldn't: the other player's
    // reasoning (it names their cards) until the game is over.
    if (!hasHuman(rec) || !isSecretGame(rec) || rec.result) return false;
    return rec.players[p].kind !== "human";
  }

  function renderSeat(rec, p, active) {
    const el = $(`seat-${p}`);
    const pl = rec.players[p];
    el.classList.toggle("turn", !!active);
    const sub = pl.kind === "bot" ? "Built-in opponent" : pl.kind === "human" ? "Playing on this screen" : (pl.model || pl.kind);
    const done = applied();
    const last = lastMoveBy(rec, done, p);
    let mindHead = "Reasoning";
    let meter = "";
    let body = "";
    let answer = "";
    const hide = hiddenFromViewer(rec, p);
    const livePending = S.mode === "live" && S.live.snap?.pending && S.live.snap.pending.player === p ? S.live.snap.pending : null;
    if (pl.kind === "bot") {
      mindHead = "Search";
      body = `<p class="mind-empty">${esc(bigOpponent(pl.name) || "The game's own opponent.")}</p>`;
      if (active) body += `<p class="mind-empty">Thinking…</p>`;
    } else if (hide) {
      body = `<p class="mind-empty">Hidden while you play: this reasoning talks about cards you can't see. It shows when the hand ends.</p>`;
      if (active || livePending) meter = "thinking…";
    } else if (livePending) {
      const text = (livePending.think ? livePending.think + "\n\n" : "") + (livePending.text || "");
      mindHead = pl.kind === "human" ? "Your move" : "Thinking";
      meter = `${Math.round(S.live.snap.now - livePending.started)} s`;
      body = pl.kind === "human" ? `<p class="mind-empty">Pick a move below.</p>` : `<div class="inner">${highlight(text, livePending.legal)}<span class="caret"></span></div>`;
    } else if (S.typing && S.typing.seat === p && !S.typing.bot) {
      mindHead = "Thinking";
      body = `<div class="inner" data-typing></div>`;
    } else if (last && last.by !== "bot") {
      mindHead = last.by === "human" ? "Last move" : "Last reasoning";
      meter = last.tokens ? `${last.tokens.toLocaleString("en-US")} tokens · ${last.seconds ?? "–"} s` : "";
      body = last.think ? `<div class="inner">${highlight(last.think, last.legal)}</div>` : `<p class="mind-empty">${last.by === "human" ? "Chosen on this screen." : "No reasoning."}</p>`;
    } else {
      body = `<p class="mind-empty">${pl.kind === "human" ? "Waiting for your turn." : pl.kind === "claude" ? "Claude, via headless Claude Code." : done ? "Waiting for its turn." : "Waiting for the first move."}</p>`;
    }
    const typingDone = S.typing && S.typing.seat === p && S.typing.done;
    const showMove = typingDone ? rec.moves[S.typing.i] : (!(S.typing && S.typing.seat === p) ? last : null);
    if (showMove && !hide) {
      if (showMove.forfeit) {
        answer = `<div class="answer forfeit"><span class="label">Forfeit</span><span>${esc(forfeitWhy(rec, showMove))}</span></div>`;
      } else {
        const nice = pretty(rec.game, showMove.action);
        const same = nice.replace(/\s/g, "").toLowerCase() === showMove.action.replace(/[<>\sXO]/g, "").toLowerCase();
        answer = `<div class="answer"><span class="label">Plays</span><code>${esc(showMove.action)}</code>${same ? "" : `<span class="verdict">${esc(nice)}</span>`}</div>`;
      }
    } else {
      answer = `<div class="answer"><span class="label">${active ? "Deciding" : "Waiting"}</span></div>`;
    }
    const final = !!rec.result && done >= rec.moves.length;
    const rewards = final ? (rec.turn_rewards?.[String(p)] || []) : rec.moves.slice(0, done).filter((m) => m.player === p).map((m) => m.reward);
    const total = rewards.reduce((a, b) => a + b, 0);
    const chips = rewards.map((v) => `<li class="chip ${v <= -5 ? "big" : v < 0 ? "neg" : v >= 0.5 ? "pos" : ""}">${fmtSigned(v)}</li>`).join("");
    el.innerHTML = `
      <div class="seat-head">
        <div class="seat-mark" style="color:${seatVar(p)}">${seatMark(rec, p)}</div>
        <div class="seat-id">
          <span class="seat-label">${esc(seatLabel(rec, p))}</span>
          <span class="seat-name">${esc(pl.name)}</span>
          <span class="seat-sub">${esc(sub)}</span>
        </div>
      </div>
      <div class="mind">
        <div class="mind-head"><span>${esc(mindHead)}</span><span class="meter">${esc(meter)}</span></div>
        <div class="mind-text">${body}</div>
        ${answer}
      </div>
      <div class="ledger">
        <div class="ledger-head"><span>${final ? "Turn rewards MARSHAL trains on" : "Turn rewards so far"}</span><span>${rewards.length ? `total ${fmtSigned(total)}` : ""}</span></div>
        <ol class="chips">${chips || '<li class="chip">none yet</li>'}</ol>
      </div>`;
    if (S.typing && S.typing.seat === p && !S.typing.bot) { S.typing.shown = -1; updateTyping(true); }
  }

  function updateTyping(force) {
    const t = S.typing;
    if (!t || t.bot) return;
    const box = document.querySelector(`#seat-${t.seat} [data-typing]`);
    if (!box) return;
    const frac = t.done ? 1 : Math.min(1, (S.clock - t.start) / t.dur);
    const n = Math.floor(t.text.length * frac);
    if (!force && n === t.shown) return;
    t.shown = n;
    const m = S.rec.moves[t.i];
    box.innerHTML = highlight(t.text.slice(0, n), m.legal) + (t.done ? "" : '<span class="caret"></span>');
  }

  function renderNarration(rec) {
    const el = $("narration");
    const done = applied();
    if (S.mode === "live" && S.live.snap?.pending) {
      const p = S.live.snap.pending.player;
      el.innerHTML = `<b>${esc(playerName(rec, p))}</b> <span class="why">is ${rec.players[p].kind === "human" ? "choosing a move" : "thinking"}…</span>`;
      return;
    }
    if (S.typing && !S.typing.done) {
      const p = S.typing.seat;
      el.innerHTML = `<b>${esc(playerName(rec, p))}</b> <span class="why">is ${S.typing.bot ? "searching" : "thinking"}…</span>`;
      return;
    }
    if (done === 0) { el.innerHTML = `<span class="why">${esc(openingLine(rec))}</span>`; return; }
    const m = rec.moves[done - 1];
    const who = `<b>${esc(playerName(rec, m.player))}</b>`;
    let line = m.forfeit ? `${who} forfeits <span class="why">— ${esc(forfeitWhy(rec, m))}</span>` : `${who} plays ${esc(pretty(rec.game, m.action))}`;
    if (done === rec.moves.length && rec.result) line += ` <span class="why">· ${esc(resultLine(rec))}</span>`;
    el.innerHTML = line;
  }
  function openingLine(rec) {
    if (rec.game === "kuhn_poker" || rec.game === "leduc_poker") return "Cards are dealt. Both players have put 1 chip in the pot.";
    if (rec.game === "hanabi") return "Each player holds three cards they can't see. Build red and yellow 1 → 2 together.";
    return `${playerName(rec, 0)} moves first.`;
  }
  function resultLine(rec) {
    const r = rec.result || {};
    if (r.cooperative) return r.reason?.startsWith("forfeit") ? `the team scores 0 after a forfeit` : `fireworks score ${fmtNum(r.score ?? 0)} of ${rec.game === "hanabi" ? 4 : "?"}`;
    if (r.reason?.startsWith("forfeit")) return `${playerName(rec, 1 - r.forfeit_by)} wins by forfeit`;
    if (r.draw) return "draw";
    if (r.winner === 0 || r.winner === 1) {
      const chips = r.returns && r.returns[r.winner];
      const poker = rec.game.endsWith("poker");
      return `${playerName(rec, r.winner)} wins${poker && chips != null ? ` ${fmtNum(chips)} chip${Math.abs(chips) === 1 ? "" : "s"}` : ""}`;
    }
    return r.truncated ? "cut off" : "game over";
  }
  function renderStrip(rec) {
    const done = applied();
    const items = rec.moves.map((m, i) => {
      const cls = [i < done ? "done" : "", i === done - 1 ? "now" : "", m.forfeit && i < done ? "forfeit" : ""].join(" ");
      const label = i < done ? (m.forfeit ? "forfeit" : pretty(rec.game, m.action)) : "·";
      return `<li class="${cls}"><i style="background:${seatVar(m.player)}"></i>${esc(label)}</li>`;
    });
    $("move-strip").innerHTML = S.mode.startsWith("live") ? items.join("") : items.join("");
  }
  function renderPrompt(rec) {
    const box = $("prompt");
    box.hidden = !S.showPrompt;
    let m = null;
    if (S.typing) m = rec.moves[S.typing.i];
    else if (S.mode === "live" && S.live.snap?.pending) m = S.live.snap.pending;
    else if (applied() > 0) m = rec.moves[applied() - 1];
    if (m && m.by === "human" && !m.observation) { box.hidden = true; return; }
    if (!m) { $("prompt-title").textContent = "What the player is told this turn"; $("prompt-text").textContent = rec.prompts?.["0"]?.user || "—"; return; }
    const who = playerName(rec, m.player);
    if (hiddenFromViewer(rec, m.player)) { $("prompt-title").textContent = `What ${who} is told`; $("prompt-text").textContent = "Hidden until the hand ends."; return; }
    if (m.prompt_tail) {
      $("prompt-title").textContent = `What ${who} is told this turn (MARSHAL's training prompt)`;
      $("prompt-text").textContent = m.prompt_tail.trim();
    } else {
      $("prompt-title").textContent = `The game state ${who} moved from`;
      $("prompt-text").textContent = (m.observation || "").trim() || "—";
    }
  }

  // ------------------------------------------------------------ boards
  function renderBoard(rec) {
    const view = currentView(rec, applied());
    const prev = applied() > 0 ? currentView(rec, applied() - 1) : null;
    const box = $("board");
    if (view.type === "grid") box.innerHTML = gridBoard(rec, view, prev);
    else if (view.type === "kuhn" || view.type === "leduc") box.innerHTML = pokerBoard(rec, view);
    else if (view.type === "hanabi") box.innerHTML = hanabiBoard(rec, view);
    else box.innerHTML = `<pre class="prompt-text">${esc(view.text || "")}</pre>`;
    if (S.mode === "live" && S.live.snap?.awaiting && view.type === "grid") wirePickableCells(rec);
  }
  function winLine(cells, need) {
    const R = cells.length, C = cells[0].length;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const v = cells[r][c];
      if (!v) continue;
      for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        const line = [];
        for (let k = 0; k < need; k++) {
          const rr = r + dr * k, cc = c + dc * k;
          if (rr < 0 || rr >= R || cc < 0 || cc >= C || cells[rr][cc] !== v) break;
          line.push(`${rr},${cc}`);
        }
        if (line.length === need) return new Set(line);
      }
    }
    return new Set();
  }
  function gridBoard(rec, view, prev) {
    const cells = view.cells || [];
    const R = cells.length, C = cells[0]?.length || 0;
    const c4 = rec.game === "connect_four";
    const wins = rec.result && applied() === rec.moves.length ? winLine(cells, c4 ? 4 : 3) : new Set();
    let html = `<div class="grid-board ${c4 ? "c4" : ""}" style="--rows:${R};--cols:${C};grid-template-columns:repeat(${C},1fr)">`;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const v = cells[r][c];
      const isLast = prev && prev.cells && v && !prev.cells[r][c];
      const cls = ["cell", isLast ? "last" : "", wins.has(`${r},${c}`) ? "win" : ""].join(" ");
      let piece = "";
      if (v) piece = c4 ? `<span class="piece disc" style="background:${seatVar(v === "X" ? 0 : 1)}"></span>`
        : `<span class="piece" style="color:${seatVar(v === "X" ? 0 : 1)}">${v === "X" ? svgX : svgO}</span>`;
      const coord = c4 ? "" : `<span class="coord">${r},${c}</span>`;
      html += `<div class="${cls}" data-r="${r}" data-c="${c}">${coord}${piece}</div>`;
    }
    html += "</div>";
    if (c4) html = `<div class="c4-wrap">${html}<div class="col-labels" style="grid-template-columns:repeat(${C},1fr)">${Array.from({ length: C }, (_, i) => `<span>${i}</span>`).join("")}</div></div>`;
    return html;
  }
  function cardHtml(rank, p, opts = {}) {
    if (!rank || opts.hidden) return `<div class="card back" style="--seat-color:${p == null ? "var(--brand)" : seatVar(p)}">?</div>`;
    return `<div class="card ${opts.public ? "public" : ""}" style="--seat-color:${p == null ? "var(--brand)" : seatVar(p)}"><small>${rank}</small>${rank}</div>`;
  }
  function pokerBoard(rec, v) {
    const over = !!rec.result && applied() === rec.moves.length;
    const humanSeat = rec.players.findIndex((p) => p.kind === "human");
    const hideCard = (p) => humanSeat >= 0 && p !== humanSeat && !over;
    const coins = Array.from({ length: Math.min(v.pot || 0, 24) }, () => '<i class="coin"></i>').join("");
    const hand = (p) => `<div class="hand"><span class="hand-label">${esc(playerName(rec, p))}</span>${cardHtml(v.cards?.[p], p, { hidden: hideCard(p) })}<span class="bets">in pot: ${v.bets?.[p] ?? "–"}${over && v.returns ? ` · ${fmtSigned(v.returns[p])}` : ""}</span></div>`;
    let center = `<div class="pot">`;
    if (v.type === "leduc") center += `<span class="hand-label">Public card</span>${v.public ? cardHtml(v.public, null, { public: true }) : '<div class="card back" style="--seat-color:var(--rule)">?</div>'}`;
    center += `<span class="hand-label">Pot</span><b>${v.pot ?? 0}</b><div class="chips-stack">${coins}</div></div>`;
    let betting = "";
    if (v.type === "kuhn") {
      betting = `<ol class="betting">${(v.actions || []).map((a) => `<li><i style="background:${seatVar(a.player)}"></i>${esc(playerName(rec, a.player))}: ${a.action === "BET" ? "bet" : "pass"}</li>`).join("") || '<li>No bets yet</li>'}</ol>`;
    } else {
      betting = (v.sequences || []).map((seq, i) => seq.length ? `<div class="round-label">Round ${i + 1}${i === 1 ? " (after the public card)" : ""}</div><ol class="betting">${seq.map((a, k) => `<li><i style="background:${seatVar(k % 2)}"></i>${esc(playerName(rec, k % 2))}: ${esc(a.toLowerCase())}</li>`).join("")}</ol>` : "").join("");
    }
    return `<div class="cards-table">${hand(0)}${center}${hand(1)}${betting}</div>`;
  }
  function hanabiBoard(rec, v) {
    const over = !!rec.result && applied() === rec.moves.length;
    const humanSeat = rec.players.findIndex((p) => p.kind === "human");
    const colors = Object.keys(v.fireworks || {});
    const fw = colors.map((c) => {
      const n = v.fireworks[c];
      return `<div class="firework">${n > 0 ? `<div class="hcard ${c}">${n}</div>` : `<div class="hcard empty">${c}</div>`}<span class="hand-label">${c === "R" ? "Red" : c === "Y" ? "Yellow" : c === "G" ? "Green" : c === "B" ? "Blue" : "White"}</span></div>`;
    }).join("");
    const hand = (p) => {
      const cards = (v.hands?.[p] || []).map((card) => {
        const hidden = humanSeat === p && !over;
        const c = card.card[0], r = card.card.slice(1);
        const hints = `${card.color_known ? `<span>${c}</span>` : ""}${card.rank_known ? `<span>${r}</span>` : ""}`;
        return `<div class="hcard ${hidden ? "hidden" : c}">${hidden ? "" : r}<div class="hint">${hints}</div></div>`;
      }).join("");
      return `<div class="hanabi-row"><span class="hand-row-label" style="color:var(--mist)"><i class="dot seat${p}"></i> ${esc(playerName(rec, p))}</span>${cards}</div>`;
    };
    const life = v.life ?? "–", info = v.info ?? "–";
    return `<div class="hanabi">
      <div class="fireworks">${fw}</div>
      <div class="tokens"><span>Lives <b>${life}</b></span><span>Hint tokens <b>${info}</b></span><span>Deck <b>${v.deck ?? "–"}</b></span><span>Discards <b>${(v.discards || []).join(" ") || "none"}</b></span></div>
      ${hand(0)}${hand(1)}
      <p class="note" style="text-align:center;margin:0">Small tags on a card are what its holder has been told about it.</p>
    </div>`;
  }

  // ------------------------------------------------------------ overlays
  function hide(id) { $(id).hidden = true; }
  function hideOverlays() { ["intro", "end"].forEach(hide); }
  function seatSide(rec, p, right) {
    const pl = rec.players[p];
    const sub = pl.kind === "bot" ? "Built-in opponent" : pl.kind === "human" ? "Person at this screen" : (pl.model || "");
    return `<div class="side ${right ? "right" : ""}">${right ? "" : `<div class="seat-mark" style="color:${seatVar(p)};border-color:${seatVar(p)}">${seatMark(rec, p)}</div>`}
      <div><b>${esc(pl.name)}</b><span>${esc(sub)}</span></div>${right ? `<div class="seat-mark" style="color:${seatVar(p)};border-color:${seatVar(p)}">${seatMark(rec, p)}</div>` : ""}</div>`;
  }
  function showIntro(rec) {
    const s = rec.settings || {};
    const tag = rec.source === "eval" ? "Arena evaluation" : rec.source === "bringup-settings" ? "Bring-up limits" : rec.source === "live" ? "Live game" : "Recorded";
    const limits = s.max_new_tokens === 1024 ? `<span class="fact"><b>1024</b>-token answers, like the Midway and Polaris runs</span>` : `<span class="fact"><b>${s.max_new_tokens}</b>-token answers, as in the paper</span>`;
    $("intro").innerHTML = `<div class="panel">
      <span class="eyebrow">${esc(tag)} · ${esc(rec.split === "held-out" ? "held-out game" : "training game")}</span>
      <h2>${esc(rec.title)}</h2>
      <p>${esc(BLURB[rec.game] || "")}</p>
      <div class="matchup">${seatSide(rec, 0, false)}<span class="vs">VS</span>${seatSide(rec, 1, true)}</div>
      <div class="facts">${limits}<span class="fact">Seed <b>${rec.seed}</b></span><span class="fact">${esc(rec.family || "")}</span>${rec.started ? `<span class="fact">${esc(rec.started.slice(0, 10))}</span>` : ""}</div>
    </div>`;
    $("intro").hidden = false;
  }
  function showEnd(rec) {
    const r = rec.result || {};
    let head;
    if (r.cooperative) head = r.reason?.startsWith("forfeit") ? "Forfeit — the team scores 0" : `Fireworks: ${fmtNum(r.score ?? 0)} of 4`;
    else if (r.reason?.startsWith("forfeit")) head = `${playerName(rec, r.forfeit_by)} forfeits`;
    else if (r.draw) head = "Draw";
    else if (r.winner === 0 || r.winner === 1) head = `${playerName(rec, r.winner)} wins`;
    else head = "Game over";
    const fm = rec.moves.find((m) => m.forfeit);
    const why = r.reason?.startsWith("forfeit") && fm ? `Why: ${forfeitWhy(rec, fm)}.` : resultLine(rec);
    const rows = [0, 1].map((p) => {
      const tr = rec.turn_rewards?.[String(p)] || [];
      const total = tr.reduce((a, b) => a + b, 0);
      return `<tr><td><span class="dot seat${p}"></span> ${esc(playerName(rec, p))}${rec.players[p].kind === "bot" ? ' <span class="note">(not trained)</span>' : ""}</td>
        <td><ol class="chips">${tr.map((v) => `<li class="chip ${v <= -5 ? "big" : v < 0 ? "neg" : v >= 0.5 ? "pos" : ""}">${fmtSigned(v)}</li>`).join("") || '<li class="chip">no turns</li>'}</ol></td>
        <td class="total">${fmtSigned(total)}</td></tr>`;
    }).join("");
    $("end").innerHTML = `<div class="panel">
      <span class="eyebrow">${esc(rec.title)} · result</span>
      <h2>${esc(head)}</h2>
      <p class="result-line">${esc(why)}</p>
      <table class="reward-table"><thead><tr><th>Player</th><th>Reward for each of its turns</th><th style="text-align:right">Game score</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="note">MARSHAL's turn-level advantage estimator learns from these per-turn rewards: +0.05 for a valid answer, +1 / −1 for a win / loss on the last turn, +0.1 each for a draw, −11 for a forfeit${rec.game.endsWith("poker") ? ", chips won or lost in poker" : rec.game === "hanabi" ? ", +1 per firework card played in Hanabi" : ""}. The game score is what <code>critic/score/mean</code> averages during training.</p>
    </div>`;
    $("end").hidden = false;
  }

  // ------------------------------------------------------------ picker
  function openPicker() {
    touched();
    const list = $("picker-list");
    const draw = () => {
      const f = $("picker-filter").value.trim().toLowerCase();
      const items = S.list.filter((g) => (!S.filterIds || S.filterIds.has(g.id)) && (!f || matches(g, f)));
      list.innerHTML = items.map((g) => `<li><button type="button" data-id="${esc(g.id)}">
        <span class="g">${esc(g.title)}</span>
        <span class="who">${esc(g.players[0].name)} vs ${esc(g.players[1].name)}</span>
        <span class="res">${esc(shortResult(g))}</span>
        <span class="tag">${esc(g.source || "")}</span></button></li>`).join("") || '<li class="empty">No games match.</li>';
    };
    $("picker-filter").oninput = draw;
    draw();
    $("picker").hidden = false;
    setTimeout(() => $("picker-filter").focus(), 30);
  }
  function shortResult(g) {
    const r = g.result || {};
    if (r.cooperative) return r.reason?.startsWith("forfeit") ? "forfeit" : `score ${fmtNum(r.score ?? 0)}`;
    if (r.reason?.startsWith("forfeit")) return `${g.players[r.forfeit_by]?.name} forfeits`;
    if (r.draw) return "draw";
    if (r.winner === 0 || r.winner === 1) return `${g.players[r.winner].name} wins`;
    return r.reason || "";
  }
  $("picker-list").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    $("picker").hidden = true;
    S.filterIds = null;
    S.playing = true;
    const idx = S.playlist.findIndex((g) => g.id === b.dataset.id);
    if (idx >= 0) S.pos = idx;
    startReplay(b.dataset.id);
  });

  // ------------------------------------------------------------ live
  async function initLive() {
    if (EMBED) return;
    try { S.live.options = await getJSON("/api/live/options"); } catch { return; }
    S.live.enabled = !!S.live.options.enabled;
    $("btn-live").hidden = !S.live.enabled;
    if (!S.live.enabled) return;
    const es = new EventSource("/api/live/stream");
    S.live.es = es;
    es.onmessage = (e) => onLive(JSON.parse(e.data));
  }
  function livePendingSeat() { return S.live.snap?.pending?.player ?? S.live.snap?.awaiting?.player ?? null; }
  function onLive(snap) {
    const prev = S.live.snap;
    if (snap.partial && prev) snap = { ...prev, ...snap };
    S.live.snap = snap;
    if (snap.status === "running") {
      if (S.mode !== "live") { cancelTasks(); hideOverlays(); $("setup").hidden = true; S.mode = "live"; setScene("arena"); }
      if (snap.record) S.rec = snap.record;
      renderArena(); renderDock(); updateContext();
    } else if (snap.status === "finished" && S.mode === "live") {
      S.rec = snap.record || S.rec;
      S.mode = "live-done";
      S.live.returnAt = Date.now() + LIVE_RETURN;
      renderArena(); renderDock(); showEnd(S.rec); updateContext();
    } else if ((snap.status === "error" || snap.status === "stopped") && S.mode === "live") {
      S.mode = "live-done";
      S.live.returnAt = Date.now() + 8000;
      $("narration").innerHTML = `<b>Live game ${snap.status === "error" ? "failed" : "stopped"}</b> <span class="why">${esc(snap.error || "")}</span>`;
      renderDock(); updateContext();
    }
  }
  async function backToReplays() {
    S.mode = "replay";
    $("dock").hidden = true;
    hideOverlays();
    await refreshList();
    playNext(0);
  }
  function renderDock() {
    const dock = $("dock");
    const a = S.live.snap?.awaiting;
    if (!a || S.mode !== "live") { dock.hidden = true; return; }
    const rec = S.rec;
    dock.style.setProperty("--seat-color", seatVar(a.player));
    dock.innerHTML = `<h3><span class="dot seat${a.player}"></span> ${esc(playerName(rec, a.player))}: your move</h3>
      <div class="moves">${a.legal.map((m) => `<button type="button" class="btn" data-move="${esc(m)}">${esc(pretty(rec.game, m))}</button>`).join("")}</div>`;
    dock.hidden = false;
  }
  $("dock").addEventListener("click", (e) => { const b = e.target.closest("[data-move]"); if (b) sendMove(b.dataset.move); });
  async function sendMove(action) {
    const a = S.live.snap?.awaiting;
    if (!a) return;
    const r = await fetch("/api/live/input", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ player: a.player, action }) });
    if (!r.ok) { const j = await r.json().catch(() => ({})); $("narration").textContent = j.error || "That move didn't go through."; }
  }
  function wirePickableCells(rec) {
    const a = S.live.snap.awaiting;
    const legal = new Map();
    for (const m of a.legal) {
      if (rec.game === "tictactoe") { const x = m.match(/\((\d),(\d)\)/); if (x) legal.set(`${x[1]},${x[2]}`, m); }
      else { const x = m.match(/\((\d)\)/); if (x) for (let r = 0; r < 6; r++) legal.set(`${r},${x[1]}`, m); }
    }
    document.querySelectorAll("#board .cell").forEach((c) => {
      const m = legal.get(`${c.dataset.r},${c.dataset.c}`);
      if (!m) return;
      c.classList.add("pickable");
      c.style.setProperty("--seat-pick", seatVar(a.player));
      c.onclick = () => sendMove(m);
    });
  }
  function openSetup() {
    if (!S.live.enabled) return;
    touched();
    const o = S.live.options;
    const gameSel = $("setup-game");
    gameSel.innerHTML = o.games.map((g) => `<option value="${g.key}">${esc(g.title)} — ${g.split === "held-out" ? "held-out" : "training"} game</option>`).join("");
    gameSel.value = o.defaults.game;
    const fill = () => {
      const g = o.games.find((x) => x.key === gameSel.value);
      for (const i of [0, 1]) {
        const sel = $(`setup-seat-${i}`);
        const keep = sel.value || o.defaults.seats[i];
        sel.innerHTML = o.players.filter((p) => p.kind !== "bot" || g.bot).map((p) =>
          `<option value="${esc(p.key)}">${esc(p.kind === "bot" ? `${g.bot}` : p.name)}${p.kind === "model" ? "" : p.kind === "claude" ? " (Claude Code)" : ""}</option>`).join("");
        sel.value = [...sel.options].some((x) => x.value === keep) ? keep : sel.options[0].value;
      }
      $("setup-note").textContent = `Models answer under the paper's limits: ${o.limits.max_new_tokens} tokens per answer, ${o.limits.sequence_length} per game. A long answer can take a minute or two.`;
    };
    gameSel.onchange = fill;
    fill();
    $("setup-error").textContent = "";
    $("setup").hidden = false;
  }
  $("setup-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("setup-start").disabled = true;
    $("setup-error").textContent = "Checking the players…";
    try {
      const body = { game: $("setup-game").value, seats: [$("setup-seat-0").value, $("setup-seat-1").value], seed: $("setup-seed").value };
      const r = await fetch("/api/live/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok) { $("setup-error").textContent = j.error || "Couldn't start the game."; return; }
      $("setup").hidden = true;
    } finally { $("setup-start").disabled = false; }
  });

  // ------------------------------------------------------------ results
  async function ensureResults() {
    if (!S.results) {
      try { S.results = await loadResults(); } catch (err) { S.results = { error: String(err) }; }
    }
    return S.results;
  }
  function slideAvailable(i) {
    if (SLIDES[i].key === "decrypto") return !!(S.results?.decrypto?.pairings?.length);
    return true;
  }
  function stepSlide(from, delta) {
    let i = from;
    for (let n = 0; n < SLIDES.length; n++) {
      i = (i + delta + SLIDES.length) % SLIDES.length;
      if (slideAvailable(i)) return i;
    }
    return from;
  }
  function gotoSlide(i) {
    const target = (i + SLIDES.length) % SLIDES.length;
    S.slide = slideAvailable(target) ? target : stepSlide(target, i < S.slide ? -1 : 1);
    S.slideUntil = S.clock + HOLD.slide;
    renderSlide();
    updateContext();
  }
  function renderSlide() {
    const R = S.results;
    const box = $("slides");
    if (!R) { box.innerHTML = '<p class="empty">Loading results…</p>'; ensureResults().then(renderSlide); return; }
    if (R.error) { box.innerHTML = `<p class="empty">Couldn't load results: ${esc(R.error)}</p>`; return; }
    if (!slideAvailable(S.slide)) S.slide = stepSlide(S.slide, 1);
    $("slide-tabs").innerHTML = SLIDES.map((s, i) => slideAvailable(i) ? `<button type="button" class="slide-tab" role="tab" aria-selected="${i === S.slide}" data-slide="${i}"><span class="tick"></span>${esc(s.label)}</button>` : "").join("");
    const key = SLIDES[S.slide].key;
    const fn = { overview: slideOverview, arena: slideArena, decrypto: slideDecrypto, runs: slideRuns, curves: slideCurves, findings: slideFindings, paper: slidePaper }[key];
    box.innerHTML = `<div class="slide">${fn(R)}</div>`;
    if (key === "curves") drawCurves(R);
    if (key === "arena" || key === "decrypto") wireArena();
  }

  function slideDecrypto(R) {
    const D = R.decrypto || { pairings: [] };
    const P = D.pairings;
    const keys = Object.keys(D.models || {});
    const cell = (team, eve) => {
      const p = P.find((x) => x.team === team && x.eve === eve);
      if (!p || !p.episodes) return `<div class="box dcell"><p class="note">Not played yet.</p></div>`;
      const segs = [["survived", "win", "Team survived 8 rounds"], ["miscommunicated", "forfeit", "Team miscommunicated twice"], ["intercepted", "loss", "Eve intercepted twice"], ["both", "other", "Both in the same round"]]
        .map(([k, cls, label]) => p[k] ? `<span class="seg ${cls}" style="flex:${p[k]}" data-tip="${esc(`${p.team_name} team vs ${p.eve_name} as Eve|${label}: ${p[k]} of ${p.episodes} games|${p.avg_rounds} rounds on average`)}">${p[k] / p.episodes >= 0.09 ? p[k] : ""}</span>` : "").join("");
      const retry = (r) => fmtNum(p.attempts[r]);
      return `<div class="box dcell">
        <div class="seg-bar">${segs}</div>
        <p><b style="color:var(--text)">${p.survived}</b> of ${p.episodes} survived · ${p.intercepted + p.both} intercepted · ${p.miscommunicated + p.both} miscommunicated · ${fmtNum(p.avg_rounds)} rounds</p>
        <p class="note">Answer retries per move: Alice ${retry("encoder")}, Bob ${retry("decoder")}, Eve ${retry("interceptor")}${p.fails.encoder + p.fails.decoder + p.fails.interceptor ? ` · ${p.fails.encoder + p.fails.decoder + p.fails.interceptor} gave up` : ""}</p>
      </div>`;
    };
    const total = P.reduce((a, p) => a + p.episodes, 0);
    const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
    const clueRows = keys.filter((k) => D.clues?.[k]?.clues).map((k) => {
      const c = D.clues[k];
      return `<tr><td class="k">${esc(c.name)}</td><td><b>${c.stem_clues}</b> of ${c.clues} (${pct(c.stem_share)})</td>
        <td>${pct(c.intercept_rate_with_stem)} <span class="note">(${c.intercepts_with_stem}/${c.rounds_with_stem})</span></td>
        <td>${pct(c.intercept_rate_without)} <span class="note">(${c.intercepts - c.intercepts_with_stem}/${c.rounds - c.rounds_with_stem})</span></td>
        <td class="note">${esc((c.examples || []).slice(0, 3).join(", "))}</td></tr>`;
    }).join("");
    const clueBox = clueRows ? `<div class="box"><h3>How each model writes clues as Alice</h3>
      <table class="drift"><thead><tr><th>Team model</th><th>Clue copies its keyword's stem</th><th>Eve intercepts, rounds with one</th><th>…rounds without</th><th>Examples (keyword → clue)</th></tr></thead><tbody>${clueRows}</tbody></table></div>` : "";
    const capCell = (s) => (s && s.moves ? `${s.capped}/${s.moves} <span class="note">(${pct(s.cap_rate)}, ~${s.mean_tokens} tokens)</span>` : "–");
    const capRows = keys.filter((k) => D.cap?.[k]).map((k) => `<tr><td class="k">${esc(D.models[k])}</td><td>${capCell(D.cap[k].encoder)}</td><td>${capCell(D.cap[k].decoder)}</td><td>${capCell(D.cap[k].interceptor)}</td></tr>`).join("");
    const capBox = capRows ? `<div class="box"><h3>Thinking cut off by Decrypto's 1024-token budget</h3>
      <table class="drift"><thead><tr><th>Model</th><th>as Alice</th><th>as Bob</th><th>as Eve</th></tr></thead><tbody>${capRows}</tbody></table>
      <p class="note" style="margin-top:0.4rem">Moves whose thinking reached the budget (≥ 1015 tokens) and was cut off before the model concluded; Decrypto then forces an answer.</p></div>` : "";
    const grid = `<div class="dgrid" style="grid-template-columns:12rem repeat(${keys.length},minmax(0,1fr))">
      <div></div>${keys.map((k) => `<div class="dhead">Eve: <b>${esc(D.models[k])}</b></div>`).join("")}
      ${keys.map((t) => `<div class="dhead row">Team (Alice + Bob):<br><b>${esc(D.models[t])}</b></div>${keys.map((e) => cell(t, e)).join("")}`).join("")}
    </div>`;
    return head("MARSHAL in Decrypto, a game it never trained on",
      `Decrypto is a word game about reading people: Alice hints at a secret code so Bob can decode it, while Eve, on the other team, tries to intercept it. It needs MARSHAL's skills (signaling a partner, reasoning about hidden information, modeling an opponent) in a new form. ${total} game${total === 1 ? "" : "s"} so far, run by Decrypto's own code with the same settings for both models.`) +
      `<div class="slide-body arena-grid">
        <div style="display:grid;grid-template-rows:auto auto auto;gap:0.8rem;min-height:0;align-content:start;overflow:hidden">
          <div class="legend"><span><i style="background:var(--win)"></i>Team survives 8 rounds</span><span><i style="background:var(--forfeit)"></i>Team miscommunicates twice</span><span><i style="background:var(--loss)"></i>Eve intercepts twice</span><span><i style="background:var(--axis)"></i>Both at once</span></div>
          ${grid}
          ${clueBox}
          ${capBox}
        </div>
        <div class="side-notes">
          <div class="box"><h3>How to read it</h3><p>Compare along a column to see which <b>team</b> communicates better against the same Eve. Compare along a row to see which <b>Eve</b> reads the same team better. Miscommunications are the team's own failures; interceptions are Eve's wins.</p></div>
          <div class="box"><h3>Settings</h3><p>Decrypto's live-display settings for both: temperature 0.6, a 1024-token thinking budget, then an <code>ANSWER:</code> line. MARSHAL was trained to answer in <code>&lt;answer&gt;</code> tags, so the retries count how often each model missed Decrypto's format.</p></div>
          <div class="box"><h3>Watch the games</h3><p>Every game replays on the Decrypto display (port 8765) with each model's reasoning: experiment <code>${esc(D.experiment)}</code>. A handful of games per cell: read only large gaps.</p></div>
        </div>
      </div>`;
  }
  const head = (title, lede) => `<div class="slide-head"><h2>${title}</h2>${lede ? `<p>${lede}</p>` : ""}</div>`;

  function slideOverview(R) {
    const P = R.paper || {};
    return head("MARSHAL: multi-agent reasoning through self-play",
      `${esc(P.venue || "")} · arXiv ${esc(P.arxiv || "")}. A ${esc(P.base_model || "Qwen3-4B")} model plays strategic games against itself, competitive and cooperative, and is trained with reinforcement learning on every turn it takes.`) +
      `<div class="slide-body overview">
        <div class="overview-left">
          <div class="hero-row">${(P.headline || []).map((h) => `<div class="hero"><b>${esc(h.value)}</b><span>${esc(h.label)}</span></div>`).join("")}</div>
          <div class="mechs">${(P.mechanisms || []).map((m) => `<div class="box"><h3>${esc(m.name)}</h3><p>${esc(m.text)}</p></div>`).join("")}</div>
          <div class="games-list">
            <div class="box"><span class="eyebrow">Trains on (self-play)</span><ul>${(P.games?.training || []).map((g) => `<li>${esc(g)}</li>`).join("")}</ul></div>
            <div class="box"><span class="eyebrow">Tested on, never trained on</span><ul>${(P.games?.held_out || []).map((g) => `<li>${esc(g)}</li>`).join("")}</ul></div>
            <div class="box" style="grid-column:1/-1"><span class="eyebrow">On this screen</span>
              <p style="margin-top:0.35rem">Every game in the Arena runs MARSHAL's own game code and the exact prompt it was trained with, served on this machine: base Qwen3-4B on one server, the published MARSHAL checkpoint on another. A model that writes past its token limit or breaks the answer format forfeits, as in training. The rewards under each player are the ones MARSHAL's trainer would assign.</p></div>
          </div>
        </div>
        <div class="figure-wrap"><div class="figure-box"><img src="${asset("/static/method_overview.svg")}" alt="MARSHAL method overview: self-play trajectories, naive advantage estimation, and MARSHAL's turn-level, per-agent advantage"></div>
          <span class="caption">Figure 1 of the paper. Left: self-play trajectories. Middle: one advantage for the whole game (e.g. GRPO) blurs credit across turns. Right: MARSHAL's per-turn, per-seat advantages.</span></div>
      </div>`;
  }

  function cellStat(c) {
    let main;
    if (c.game === "hanabi") main = c.mean_score != null ? `score <b>${fmtNum(c.mean_score)}</b> / 4` : `<b>0</b> finished`;
    else if (c.game.endsWith("poker")) main = c.mean_return != null ? `<b>${fmtSigned(c.mean_return)}</b> chips/hand` : "–";
    else main = `wins <b>${c.outcomes.win}</b> of ${c.games}`;
    const tok = c.tokens_per_answer ? `<br><span class="note">${c.tokens_per_answer.toLocaleString("en-US")} tokens/answer</span>` : "";
    return main + tok;
  }
  function slideArena(R) {
    const A = R.arena || { cells: [] };
    if (!A.cells.length) return head("Arena results", "") + '<p class="empty">No evaluation games yet. Run <code>bash display/record_eval.sh</code>.</p>';
    const order = ["tictactoe", "kuhn_poker", "hanabi", "connect_four", "leduc_poker"];
    const titles = { tictactoe: "Tic-Tac-Toe", kuhn_poker: "Kuhn Poker", hanabi: "Mini Hanabi", connect_four: "Connect Four", leduc_poker: "Leduc Hold'em" };
    const rows = order.filter((g) => A.cells.some((c) => c.game === g)).map((g) => {
      const cells = A.cells.filter((c) => c.game === g).sort((a, b) => (a.model || "").includes("MARSHAL") - (b.model || "").includes("MARSHAL"));
      const opp = cells[0].opponent;
      const split = ["connect_four", "leduc_poker"].includes(g) ? "held-out" : "training";
      const bars = cells.map((c) => {
        const segs = ["win", "draw", "loss", "forfeit", "other"].map((k) => {
          const n = c.outcomes[k] || 0;
          if (!n) return "";
          const pct = (n / c.games) * 100;
          const label = pct >= 9 ? n : "";
          const ci = k === "win" && c.win_ci ? ` · 95% range ${Math.round(c.win_ci[0] * 100)}–${Math.round(c.win_ci[1] * 100)}%` : "";
          return `<span class="seg ${k}" style="flex:${n}" data-tip="${esc(`${c.name} vs ${c.opponent} · ${titles[g]}|${k === "other" ? "Hanabi game finished" : k[0].toUpperCase() + k.slice(1)}: ${n} of ${c.games} games (${Math.round(pct)}%)${ci}|${c.tokens_per_answer ? `~${c.tokens_per_answer.toLocaleString("en-US")} tokens per answer` : ""}`)}">${label}</span>`;
        }).join("");
        return `<div class="bar-line" tabindex="0" data-ids="${esc(c.ids.join(","))}" aria-label="${esc(`${c.name}: ${c.outcomes.win} wins, ${c.outcomes.draw} draws, ${c.outcomes.loss} losses, ${c.outcomes.forfeit} forfeits of ${c.games}`)}">
          <span class="who">${esc(c.name)}</span><div class="seg-bar">${segs}</div><span class="stat">${cellStat(c)}</span></div>`;
      }).join("");
      return `<div class="arena-row"><div class="gname"><b>${titles[g]}</b><span>${split === "held-out" ? "held out · " : ""}vs ${esc(g === "hanabi" ? "itself (team play)" : opp)}</span></div><div class="bars">${bars}</div></div>`;
    }).join("");
    const total = A.cells.reduce((a, c) => a + c.games, 0);
    const byModel = {};
    for (const c of A.cells) {
      const m = (byModel[c.model] = byModel[c.model] || { name: c.name, games: 0, forfeits: 0, tok: 0, answers: 0 });
      m.games += c.games;
      m.forfeits += c.outcomes.forfeit || 0;
      m.tok += (c.tokens_per_answer || 0) * (c.answers || 0);
      m.answers += c.answers || 0;
    }
    const models = Object.values(byModel);
    const standout = models.length ? `<div class="box"><h3>What stands out</h3>
      <p>Games forfeited (bad format, or an answer past 4096 tokens): ${models.map((m) => `<b>${esc(m.name)}</b> ${m.forfeits} of ${m.games}`).join(" · ")}.</p>
      <p style="margin-top:0.35rem">Average answer: ${models.map((m) => `${esc(m.name)} ${m.answers ? Math.round(m.tok / m.answers).toLocaleString("en-US") : "–"} tokens`).join(" · ")}.</p></div>` : "";
    const h2h =(A.head_to_head || []).map((h) => `<p>${esc(h.title)}: ${h.models.map((m) => `${esc(h.names[m] || m)} ${h.wins[m]} wins`).join(", ")}, ${h.draws} draws (${h.games} games).</p>`).join("");
    return head("Arena: base Qwen3-4B vs MARSHAL, against the same opponents",
      `${total} games played on this machine with MARSHAL's own game code and prompts, under the paper's limits (4096-token answers). Each model moved first in half its games. Click a bar to watch those games.`) +
      `<div class="slide-body arena-grid">
        <div style="display:grid;grid-template-rows:auto minmax(0,1fr);gap:0.6rem;min-height:0">
          <div class="legend"><span><i style="background:var(--win)"></i>Win</span><span><i style="background:var(--draw)"></i>Draw</span><span><i style="background:var(--loss)"></i>Loss</span><span><i style="background:var(--forfeit)"></i>Forfeit (format or length)</span><span><i style="background:var(--axis)"></i>Hanabi game played out</span></div>
          <div class="arena-rows">${rows}</div>
        </div>
        <div class="side-notes">
          ${standout}
          <div class="box"><h3>The models</h3><p><b>Qwen3-4B</b> is the base model. <b>MARSHAL 4B</b> is the published <code>MARSHAL-Generalist-Qwen3-4B</code>, trained by self-play on Tic-Tac-Toe, Kuhn Poker and Mini Hanabi. Connect Four and Leduc Hold'em are new to it.</p></div>
          <div class="box"><h3>The opponents</h3><p>MARSHAL's evaluation bots: MCTS with 100 simulations for the board games, an equilibrium strategy for each poker game. Hanabi is two copies of the same model playing as a team (scores 0–4).</p></div>
          <div class="box"><h3>How much to trust it</h3><p>8–12 games per bar. A gap of a game or two is noise; only large gaps mean much. More games: <code>bash display/record_eval.sh</code>.</p>${h2h}</div>
        </div>
      </div>`;
  }
  function wireArena() {
    document.querySelectorAll(".seg[data-tip]").forEach((s) => {
      s.addEventListener("mousemove", (e) => { const [a, b, c] = s.dataset.tip.split("|"); showTip(`<b>${esc(a)}</b><div class="t-row">${esc(b)}</div>${c ? `<div class="t-row">${esc(c)}</div>` : ""}`, e.clientX, e.clientY); });
      s.addEventListener("mouseleave", hideTip);
    });
    document.querySelectorAll(".bar-line[data-ids]").forEach((b) => {
      if (!b.dataset.ids) return;
      const open = () => { hideTip(); S.filterIds = new Set(b.dataset.ids.split(",")); $("picker-filter").value = ""; openPicker(); };
      b.addEventListener("click", open);
      b.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    });
  }

  function slideRuns(R) {
    const lanes = {};
    for (const r of R.runs || []) (lanes[r.cluster] = lanes[r.cluster] || []).push(r);
    const lane = (name, runs) => `<div class="lane"><h3>${esc(name)} <span>${esc(runs[0].scheduler || "")} · ${esc(runs[0].hardware || "")}</span></h3>
      <div class="lane-cards">${runs.map((r) => `<button type="button" class="run-card" data-run="${esc(r.id)}">
        <div class="top-line"><span>${esc(r.date || "")} · job ${esc(r.jid || "–")}</span><span class="status">${esc(r.status || "")}</span></div>
        <b>${esc(r.title)}</b>
        <dl><dt>Model</dt><dd>${esc(r.model || "–")}</dd><dt>Training</dt><dd>${esc(r.strategy || "–")}</dd><dt>Steps</dt><dd>${esc(r.steps ?? "–")} in ${esc(r.elapsed || "–")}</dd><dt>Answers</dt><dd>${esc(r.settings?.max_new_tokens ?? "–")} tokens max</dd></dl>
        <span class="proof">${esc(r.proof || "")}</span>
        <span class="data">${r.points ? `${r.points} steps of metrics · ${esc(r.source || "")}` : r.milestones ? "Milestones from the notes; copy the TensorBoard folder to see curves" : "No metrics on this machine"}</span>
      </button>`).join("")}</div></div>`;
    return head("The bring-up runs", "Getting MARSHAL's training loop to run on two clusters: Midway (RCC, SLURM) and Polaris (ALCF, PBS). Every run here finished and wrote a checkpoint. Click one for its curves.") +
      `<div class="slide-body runs-board">${Object.entries(lanes).map(([k, v]) => lane(k, v)).join("")}</div>`;
  }

  const CHARTS = [
    { title: "Game score", sub: "critic/score/mean: a player's summed turn rewards", series: [{ key: "critic/score/mean", name: "Score", color: "var(--seat1)" }] },
    { title: "Forfeits by seat", sub: "share of games lost for format", series: [{ key: "env/TicTacToe/player_0_lose_for_wrong_format", name: "X", color: "var(--seat0)" }, { key: "env/TicTacToe/player_1_lose_for_wrong_format", name: "O", color: "var(--seat1)" }] },
    { title: "Tokens per game, by seat", sub: "response_length_player_0 / _1", series: [{ key: "env/TicTacToe/response_length_player_0", name: "X", color: "var(--seat0)" }, { key: "env/TicTacToe/response_length_player_1", name: "O", color: "var(--seat1)" }] },
    { title: "Policy-gradient loss", sub: "actor/pg_loss: 0 means no learning signal", series: [{ key: "actor/pg_loss", name: "pg_loss", color: "var(--seat1)" }] },
    { title: "KL to the reference model", sub: "actor/kl_loss", series: [{ key: "actor/kl_loss", name: "KL", color: "var(--seat1)" }] },
    { title: "Throughput", sub: "system/tps, tokens per second", series: [{ key: "system/tps", name: "tok/s", color: "var(--seat1)" }] },
  ];
  function slideCurves(R) {
    const runs = R.runs || [];
    if (!S.curvesRun) S.curvesRun = (runs.find((r) => r.id.startsWith("midway-scaleup")) || runs[0] || {}).id;
    const run = runs.find((r) => r.id === S.curvesRun) || runs[0];
    const chips = runs.map((r) => `<button type="button" class="run-chip" data-run="${esc(r.id)}" aria-pressed="${r.id === run.id}">${esc(r.cluster.split(" ")[0])} · ${esc(r.title)}</button>`).join("");
    const src = run.points ? `${run.points} steps from ${esc(run.source)}` : run.milestones ? "Only milestone values are on this machine (from the Polaris notes); the dots are those values, not a curve." : "No metrics on this machine.";
    return head(`Training curves · ${esc(run.cluster)} · ${esc(run.title)}`, `${esc(run.model || "")}, ${esc(run.strategy || "")}, ${esc(run.settings?.max_new_tokens ?? "")}-token answers. ${src}`) +
      `<div class="slide-body curves"><div class="run-picker"><span class="lbl">Run</span>${chips}</div>
        <div class="charts">${CHARTS.map((c, i) => `<div class="chart" data-chart="${i}"><h4>${esc(c.title)}</h4><div class="sub">${esc(c.sub)}</div><div class="plot" style="min-height:0;position:relative"></div></div>`).join("")}</div></div>`;
  }
  function drawCurves(R) {
    const run = (R.runs || []).find((r) => r.id === S.curvesRun) || (R.runs || [])[0];
    if (!run) return;
    document.querySelectorAll(".chart").forEach((box) => {
      const spec = CHARTS[+box.dataset.chart];
      const plot = box.querySelector(".plot");
      const series = spec.series.map((s) => {
        let pts = run.series?.[s.key] || [];
        let milestone = false;
        if (!pts.length && run.milestones && Array.isArray(run.milestones[s.key])) { pts = run.milestones[s.key]; milestone = true; }
        return { ...s, points: pts, milestone };
      }).filter((s) => s.points.length);
      if (!series.length) {
        const ms = run.milestones || {};
        const text = Object.entries(ms).filter(([k]) => spec.series.some((s) => s.key === k)).map(([, v]) => (typeof v === "string" ? v : "")).join(" ");
        plot.innerHTML = `<div class="nodata">${esc(text || (run.points ? "This metric isn't logged for this run." : "No per-step data for this run here."))}</div>`;
        return;
      }
      if (series.length > 1) box.querySelector(".sub").innerHTML = `<span class="chart-legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</span>`;
      lineChart(plot, series, spec);
    });
    const grad = run.milestones?.["actor_train/grad_norm"];
    if (typeof grad === "string") {
      const kl = document.querySelector('.chart[data-chart="4"] .sub');
      if (kl) kl.textContent = `actor/kl_loss · grad norm ${grad}`;
    }
  }
  function niceTicks(lo, hi, n = 4) {
    if (lo === hi) { const d = Math.abs(lo) || 1; lo -= d * 0.5; hi += d * 0.5; }
    const span = hi - lo;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const start = Math.floor(lo / step) * step, end = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let v = start; v <= end + step / 2; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }
  function lineChart(plot, series, spec) {
    const W = Math.max(200, plot.clientWidth), H = Math.max(90, plot.clientHeight);
    const m = { l: 46, r: 52, t: 10, b: 22 };
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = series.flatMap((s) => s.points.map((p) => p[1]));
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    const ticks = niceTicks(Math.min(...ys, ys.every((y) => y >= 0) ? 0 : Math.min(...ys)), Math.max(...ys));
    const y0 = ticks[0], y1 = ticks[ticks.length - 1];
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(spec.title)}">`;
    for (const t of ticks) svg += `<line class="${t === y0 ? "baseline" : "gridline"}" x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text class="tick" x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${fmtNum(t)}</text>`;
    const xt = niceTicks(x0, x1, 4).filter((t) => t >= x0 && t <= x1 && Number.isInteger(t));
    for (const t of xt) svg += `<text class="tick" x="${X(t)}" y="${H - 5}" text-anchor="middle">${t}</text>`;
    for (const s of series) {
      if (s.milestone) {
        // Reported values only: no line between them, since the path between is unknown.
        for (const p of s.points) {
          svg += `<circle class="dot" cx="${X(p[0])}" cy="${Y(p[1])}" r="5" fill="${s.color}"/>`;
          svg += `<text class="end-label" x="${X(p[0]) + 8}" y="${Y(p[1]) - 7}">${fmtNum(p[1])}</text>`;
        }
        continue;
      }
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      svg += `<path class="series" d="${d}" stroke="${s.color}"/>`;
      const dots = s.points.length <= 25 ? s.points : [s.points[s.points.length - 1]];
      for (const p of dots) svg += `<circle class="dot" cx="${X(p[0])}" cy="${Y(p[1])}" r="4.5" fill="${s.color}"/>`;
      const last = s.points[s.points.length - 1];
      if (series.length === 1 || s === series[series.length - 1] || Math.abs(Y(last[1]) - Y(series[0].points[series[0].points.length - 1][1])) > 14) {
        svg += `<text class="end-label" x="${X(last[0]) + 8}" y="${Y(last[1]) + 4}">${fmtNum(last[1])}</text>`;
      }
    }
    svg += `<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" visibility="hidden"/><rect class="hit" x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}"/></svg>`;
    plot.innerHTML = svg;
    const el = plot.querySelector("svg"), cross = el.querySelector(".cross");
    const steps = [...new Set(xs)].sort((a, b) => a - b);
    el.querySelector(".hit").addEventListener("mousemove", (e) => {
      const r = el.getBoundingClientRect();
      const vx = x0 + ((e.clientX - r.left) * (W / r.width) - m.l) / (W - m.l - m.r) * (x1 - x0);
      const step = steps.reduce((a, b) => (Math.abs(b - vx) < Math.abs(a - vx) ? b : a));
      cross.setAttribute("x1", X(step)); cross.setAttribute("x2", X(step)); cross.setAttribute("visibility", "visible");
      const rows = series.map((s) => { const p = s.points.find((q) => q[0] === step); return p ? `<div class="t-row"><span><i style="background:${s.color}"></i>${esc(s.name)}</span><b>${fmtNum(p[1], 3)}</b></div>` : ""; }).join("");
      showTip(`<b>Step ${step}</b>${rows}`, e.clientX, e.clientY);
    });
    el.querySelector(".hit").addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  }

  function slideFindings(R) {
    const F = R.findings || [];
    const decode = `<div class="decode"><span class="chip pos">+0.05 valid answer</span><span class="chip pos">+1 win</span><span class="chip neg">−1 loss</span><span class="chip">+0.1 draw</span><span class="chip big">−11 forfeit</span></div>`;
    return head("What the numbers say", "Read from the per-step logs and MARSHAL's reward code. Cards marked estimate rest on milestone values only.") +
      `<div class="slide-body findings">${F.map((f, i) => `<div class="finding ${i === 1 ? "wide" : ""}">${f.estimate ? '<span class="badge">estimate</span>' : ""}<h3>${esc(f.title)}</h3><p>${esc(f.body)}</p>${f.id === "score-is-raw" ? decode : ""}<span class="src">${esc(f.source || "")}</span></div>`).join("")}</div>`;
  }
  function slidePaper(R) {
    const P = R.paper || {};
    const rows = (R.drift || []).map((d) => `<tr><td class="k">${esc(d.knob)}</td><td>${esc(d.paper)}</td><td class="${d.same ? "same" : ""}"><span class="${d.same ? "mark-same" : "mark-diff"}"></span>${esc(d.bringup)}</td></tr>`).join("");
    return head("Bring-up vs the paper", "The bring-up kept MARSHAL's method and base model and cut the scale to fit one node. Its numbers are training diagnostics, not the paper's evaluations, so they can't be compared directly.") +
      `<div class="slide-body paper-grid">
        <div class="box" style="overflow:hidden"><table class="drift"><thead><tr><th>Setting</th><th>Paper config</th><th>Bring-up runs</th></tr></thead><tbody>${rows}</tbody></table></div>
        <div class="side-notes">
          <div class="box"><h3>Published checkpoints</h3><p>${(P.models || []).map((m) => `<code>${esc(m.id.split("/")[1])}</code> — ${esc(m.trained_on)}`).join("<br>")}</p></div>
          <div class="box"><h3>What the paper measures</h3><p>Win rates against fixed opponents on held-out games, and benchmark accuracy (AIME, GPQA-Diamond, …) when the model sits inside multi-agent systems such as MAD and AutoGen.</p></div>
          <div class="box"><h3>What this screen adds</h3><p>The Arena page is the first kind of measurement, small and local: the published Generalist checkpoint and its base model against MARSHAL's evaluation bots under the paper's token limits.</p></div>
        </div>
      </div>`;
  }
  $("slides").addEventListener("click", (e) => {
    const card = e.target.closest("[data-run]");
    if (card) {
      touched();
      S.curvesRun = card.dataset.run;
      gotoSlide(SLIDES.findIndex((s) => s.key === "curves"));
    }
  });
  $("slide-tabs").addEventListener("click", (e) => { const b = e.target.closest("[data-slide]"); if (b) { touched(); gotoSlide(+b.dataset.slide); } });
  window.addEventListener("resize", () => { if (S.scene === "results") renderSlide(); });

  // ------------------------------------------------------------ controls
  document.querySelectorAll(".scene-tab").forEach((b) => b.addEventListener("click", () => {
    touched();
    if (b.dataset.scene === "results") goResults();
    else goArena();
  }));
  $("btn-games").addEventListener("click", () => { S.filterIds = null; openPicker(); });
  $("btn-live").addEventListener("click", openSetup);
  document.addEventListener("click", (e) => { const c = e.target.closest("[data-close]"); if (c) { c.closest(".overlay").hidden = true; } });
  function togglePause() {
    touched();
    S.playing = !S.playing;
    if (S.playing && S.phase === "manual" && S.rec) { S.phase = "play"; if (S.applied < S.rec.moves.length) stepMove(); else finishReplay(); }
    updateContext();
  }
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea")) { if (e.key === "Escape") e.target.blur(); return; }
    const k = e.key;
    if (k === "Escape") { document.querySelectorAll(".sheet").forEach((o) => { o.hidden = true; }); return; }
    if (!$("picker").hidden || !$("setup").hidden) return;
    if (k === " ") { e.preventDefault(); togglePause(); }
    else if (k === "ArrowRight") stepManual(1);
    else if (k === "ArrowLeft") stepManual(-1);
    else if (k === "n" || k === "N") { touched(); S.playing = true; playNext(1); }
    else if (k === "r" || k === "R") { touched(); goResults(); }
    else if (k === "a" || k === "A") { touched(); goArena(); }
    else if (k === "g" || k === "G") { S.filterIds = null; openPicker(); }
    else if (k === "l" || k === "L") openSetup();
    else if (k === "p" || k === "P") { S.showPrompt = !S.showPrompt; renderArena(); }
    else if (k === "+" || k === "=") { S.speed = clampSpeed(S.speed * 1.25); updateContext(); }
    else if (k === "-" || k === "_") { S.speed = clampSpeed(S.speed / 1.25); updateContext(); }
    else if (k === "h" || k === "H") { S.footerHidden = !S.footerHidden; $("bottom").classList.toggle("hidden-ui", S.footerHidden); }
    else if (k === "f" || k === "F") { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.(); }
    else if (k === "?") $("help").hidden = !$("help").hidden;
  });

  // ------------------------------------------------------------ start
  async function init() {
    if (EMBED) document.querySelectorAll('img[src^="/static/"]').forEach((img) => { img.src = asset(img.getAttribute("src")); });
    try { navigator.wakeLock?.request("screen").catch(() => {}); } catch { /* not supported */ }
    requestAnimationFrame(frame);
    ensureResults();
    initLive();
    await refreshList();
    if (S.scene === "results") { S.phase = "slide"; S.slidesShown = -4; S.slideUntil = HOLD.slide; setScene("results"); if (!S.playlist.length) return; S.rec = await loadGame(S.playlist[0].id); return; }
    if (S.playlist.length) startReplay(S.playlist[0].id); else showEmpty();
  }
  init();
})();
