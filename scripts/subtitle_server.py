"""Local subtitle overlay server for OBS (Mojicast-style).

Serves an overlay page at http://localhost:<port>/ suitable for an OBS
browser source (transparent background), and streams partial/final subtitle
events over SSE at /events. Designed to be driven by realtime_transcribe.py.
"""
import http.server
import json
import queue
import sys
import threading
from typing import Callable

OVERLAY_HTML = """<!doctype html>
<meta charset="utf-8">
<style>
  html, body { margin: 0; background: transparent; overflow: hidden; }
  #box {
    position: absolute; left: 0; right: 0; bottom: 4vh;
    text-align: center; font-family: "Yu Gothic UI", "Meiryo", sans-serif;
    font-size: 5vh; line-height: 1.4; color: #fff;
    text-shadow: 0 0 8px #000, 0 0 4px #000, 2px 2px 2px #000;
  }
  #final-line, #partial-line { display: block; min-height: 1.4em; }
  #partial-line { opacity: 0.75; font-style: italic; font-size: 0.8em; }
</style>
<div id="box"><span id="final-line"></span><span id="partial-line"></span></div>
<script>
  // ?show=final / ?show=partial renders only that row, so OBS can place the
  // confirmed line and the in-progress line as two independent sources.
  const mode = new URLSearchParams(location.search).get("show") || "both";
  const fin = document.getElementById("final-line");
  const par = document.getElementById("partial-line");
  if (mode === "final") par.style.display = "none";
  if (mode === "partial") fin.style.display = "none";
  let clearTimer = null;
  const es = new EventSource("/events");
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    // Any message -- partial included -- means the session is still live,
    // so postpone clearing the confirmed line: a run of partials right
    // after a final (the speaker keeps talking) must not let the 6s timer
    // wipe the final out from under them.
    if (clearTimer) { clearTimeout(clearTimer); clearTimer = null; }
    if (ev.type === "partial") {
      par.textContent = ev.text;
    } else if (ev.type === "final") {
      fin.textContent = ev.text;
      par.textContent = "";
    }
    clearTimer = setTimeout(() => { fin.textContent = ""; }, 6000);
  };
</script>
"""


TRANSCRIPT_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Transcript</title>
<style>
  body { margin: 0; padding: 2rem; background: #111; color: #eee;
         font-family: "Yu Gothic UI", "Meiryo", sans-serif; font-size: 1.1rem; line-height: 1.8; }
  #lines p { margin: 0.2rem 0; }
  .lang { color: #888; font-size: 0.8em; margin-right: 0.6em; }
</style>
<div id="lines"></div>
<script>
  const box = document.getElementById("lines");
  const es = new EventSource("/events");
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type !== "refine") return;
    const p = document.createElement("p");
    p.innerHTML = '<span class="lang">[' + (ev.lang || "?") + ']</span>';
    p.appendChild(document.createTextNode(ev.text));
    box.appendChild(p);
    window.scrollTo(0, document.body.scrollHeight);
  };
</script>
"""


DASHBOARD_HTML = '<!doctype html>\n<html lang="ja">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>hayamimi</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link href="https://fonts.googleapis.com/css2?family=Shippori+Mincho+B1:wght@600;800&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">\n<style>\n  :root {\n    --ink: #0d0f13; --ink-2: #14171d; --line: #2a2f3a;\n    --paper: #ece5d8; --paper-dim: #b0a898;\n    --shu: #e04f2f;\n    --c-ja: #e04f2f; --c-en: #5b7fd4; --c-zh: #d4a13c; --c-ko: #3fae9d; --c-yue: #b06bc6; --c-xx: #7a8194;\n    --mono: "IBM Plex Mono", monospace;\n  }\n  * { box-sizing: border-box; }\n  html, body { height: 100%; }\n  body {\n    margin: 0; background: var(--ink); color: var(--paper);\n    font-family: "Zen Kaku Gothic New", "Yu Gothic UI", sans-serif;\n    background-image:\n      radial-gradient(1200px 500px at 85% -10%, rgba(224,79,47,.07), transparent 60%),\n      repeating-linear-gradient(0deg, transparent 0 2px, rgba(255,255,255,.006) 2px 4px);\n    display: flex; flex-direction: column; overflow: hidden;\n  }\n  header {\n    display: flex; align-items: baseline; gap: 1.1rem;\n    padding: .9rem 1.6rem .7rem; border-bottom: 1px solid var(--line);\n    flex: none;\n  }\n  .logotype { font-family: "Shippori Mincho B1", serif; font-weight: 800;\n    font-size: 1.7rem; letter-spacing: .12em; }\n  .logotype em { font-style: normal; color: var(--shu); }\n  .romaji { font-family: var(--mono); font-size: .72rem; letter-spacing: .3em;\n    color: var(--paper-dim); }\n  .spacer { flex: 1; }\n  .chip { font-family: var(--mono); font-size: .72rem; color: var(--paper-dim);\n    border: 1px solid var(--line); border-radius: 3px; padding: .25rem .6rem; }\n  .chip b { color: var(--paper); font-weight: 600; }\n  #live-dot { width: .55rem; height: .55rem; border-radius: 50%;\n    background: var(--c-xx); align-self: center; }\n  #live-dot.on { background: var(--shu);\n    animation: pulse 1.6s ease-out infinite; }\n  @keyframes pulse {\n    from { box-shadow: 0 0 0 0 rgba(224,79,47,.5); }\n    to { box-shadow: 0 0 0 10px rgba(224,79,47,0); } }\n\n  #partial-strip {\n    flex: none; padding: 1.2rem 1.6rem 1.1rem; min-height: 5.6rem;\n    border-bottom: 1px solid var(--line); background: var(--ink-2);\n    position: relative; overflow: hidden;\n  }\n  #partial-strip::before {\n    content: "いま聞き取り中"; position: absolute; top: .55rem; left: 1.65rem;\n    font-size: .62rem; letter-spacing: .35em; color: var(--paper-dim);\n  }\n  #partial-text {\n    font-family: "Shippori Mincho B1", serif; font-weight: 600;\n    font-size: 1.5rem; line-height: 1.5; margin-top: 1rem; min-height: 2.2rem;\n  }\n  #partial-text.idle { color: var(--paper-dim); opacity: .5; }\n  .caret { display: inline-block; width: .5em; height: 1.05em;\n    background: var(--shu); vertical-align: -0.12em; margin-left: .18em;\n    animation: blink 1s steps(2) infinite; }\n  @keyframes blink { 50% { opacity: 0; } }\n\n  main { flex: 1; display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);\n    min-height: 0; }\n  section { display: flex; flex-direction: column; min-height: 0; }\n  section + section { border-left: 1px solid var(--line); }\n  h2 { flex: none; margin: 0; padding: .7rem 1.6rem; font-size: .68rem;\n    font-weight: 700; letter-spacing: .35em; color: var(--paper-dim);\n    border-bottom: 1px solid var(--line); }\n  h2 span { color: var(--shu); }\n  .scroll { flex: 1; overflow-y: auto; padding: 1rem 1.6rem 2rem;\n    scrollbar-width: thin; scrollbar-color: var(--line) transparent; }\n\n  .final-card {\n    display: grid; grid-template-columns: auto 1fr auto; gap: .7rem;\n    align-items: baseline; padding: .55rem 0;\n    border-bottom: 1px dashed rgba(42,47,58,.7);\n    animation: rise .28s cubic-bezier(.2,.9,.3,1) both;\n  }\n  @keyframes rise { from { opacity: 0; transform: translateY(8px); } }\n  .badges { display: flex; gap: .35rem; align-items: center; }\n  .lang-badge { font-family: var(--mono); font-size: .62rem; font-weight: 600;\n    padding: .12rem .42rem; border-radius: 2px; color: var(--ink);\n    letter-spacing: .08em; }\n  .spk { font-family: var(--mono); font-size: .62rem; color: var(--paper-dim);\n    border: 1px solid var(--line); padding: .1rem .34rem; border-radius: 2px; }\n  .final-text { font-size: 1.02rem; line-height: 1.65; }\n  .lat { font-family: var(--mono); font-size: .66rem; color: var(--paper-dim); }\n  .trans-line { grid-column: 2; font-size: .85rem; color: var(--paper-dim);\n    line-height: 1.5; }\n  .trans-line b { font-family: var(--mono); font-size: .6rem; font-weight: 600;\n    color: var(--shu); margin-right: .45em; letter-spacing: .1em; }\n\n  .refine-p { margin: 0 0 1rem; font-family: "Shippori Mincho B1", serif;\n    font-weight: 600; font-size: 1.02rem; line-height: 1.9;\n    animation: rise .3s ease both; }\n  .refine-p .meta { font-family: var(--mono); font-size: .62rem; font-weight: 400;\n    color: var(--paper-dim); display: block; margin-bottom: .18rem;\n    letter-spacing: .12em; }\n  .refine-p .meta i { font-style: normal; color: var(--shu); }\n\n  footer { flex: none; display: flex; gap: 1.4rem; padding: .5rem 1.6rem;\n    border-top: 1px solid var(--line); font-family: var(--mono);\n    font-size: .66rem; color: var(--paper-dim); }\n  @media (max-width: 860px) { main { grid-template-columns: 1fr; }\n    section + section { border-left: 0; border-top: 1px solid var(--line); } }\n</style>\n<body>\n<header>\n  <div id="live-dot"></div>\n  <div class="logotype">早<em>耳</em></div>\n  <div class="romaji">hayamimi</div>\n  <div class="spacer"></div>\n  <div class="chip">確定 <b id="n-finals">0</b></div>\n  <div class="chip">平均応答 <b id="mean-lat">&ndash;</b> ms</div>\n  <div class="chip">言語 <b id="langs-seen">&ndash;</b></div>\n</header>\n\n<div id="partial-strip">\n  <div id="partial-text" class="idle">マイクの音声を待っています<span class="caret"></span></div>\n</div>\n\n<main>\n  <section>\n    <h2>確定フィード <span>LIVE</span></h2>\n    <div class="scroll" id="feed"></div>\n  </section>\n  <section>\n    <h2>清書 <span>REFINED</span></h2>\n    <div class="scroll" id="refined"></div>\n  </section>\n</main>\n\n<footer>\n  <div id="conn">connecting&hellip;</div>\n  <div>overlay: /</div>\n  <div>transcript: /transcript</div>\n</footer>\n\n<script>\n  const LANG_COLOR = { ja: "var(--c-ja)", en: "var(--c-en)", zh: "var(--c-zh)",\n                       ko: "var(--c-ko)", yue: "var(--c-yue)" };\n  const feed = document.getElementById("feed");\n  const refined = document.getElementById("refined");\n  const partial = document.getElementById("partial-text");\n  const dot = document.getElementById("live-dot");\n  let finals = 0, latSum = 0, langs = new Set(), lastCard = null, idleTimer = null;\n\n  function badge(lang) {\n    const b = document.createElement("span");\n    b.className = "lang-badge";\n    b.style.background = LANG_COLOR[lang] || "var(--c-xx)";\n    b.textContent = lang || "??";\n    return b;\n  }\n  function stick(el) { el.scrollTop = el.scrollHeight; }\n\n  const es = new EventSource("/events");\n  es.onopen = () => { document.getElementById("conn").textContent = "connected"; dot.classList.add("on"); };\n  es.onerror = () => { document.getElementById("conn").textContent = "reconnecting…"; dot.classList.remove("on"); };\n  es.onmessage = (e) => {\n    const ev = JSON.parse(e.data);\n    if (ev.type === "partial") {\n      partial.classList.remove("idle");\n      partial.textContent = ev.text;\n      const c = document.createElement("span"); c.className = "caret";\n      partial.appendChild(c);\n      clearTimeout(idleTimer);\n      idleTimer = setTimeout(() => partial.classList.add("idle"), 4000);\n    } else if (ev.type === "final") {\n      const card = document.createElement("div"); card.className = "final-card";\n      const badges = document.createElement("div"); badges.className = "badges";\n      badges.appendChild(badge(ev.lang));\n      if (ev.speaker) { const s = document.createElement("span");\n        s.className = "spk"; s.textContent = ev.speaker; badges.appendChild(s); }\n      const t = document.createElement("div"); t.className = "final-text";\n      t.textContent = ev.text;\n      const lat = document.createElement("div"); lat.className = "lat";\n      lat.textContent = ev.latency_ms != null ? Math.round(ev.latency_ms) + "ms" : "";\n      card.append(badges, t, lat);\n      feed.appendChild(card); lastCard = card; stick(feed);\n      finals++; langs.add(ev.lang);\n      if (ev.latency_ms != null) latSum += ev.latency_ms;\n      document.getElementById("n-finals").textContent = finals;\n      document.getElementById("mean-lat").textContent = Math.round(latSum / finals);\n      document.getElementById("langs-seen").textContent = [...langs].join(" ");\n    } else if (ev.type === "translation") {\n      if (!lastCard) return;\n      const tr = document.createElement("div"); tr.className = "trans-line";\n      const b = document.createElement("b"); b.textContent = "→" + ev.lang;\n      tr.appendChild(b); tr.appendChild(document.createTextNode(ev.text));\n      lastCard.appendChild(tr); stick(feed);\n    } else if (ev.type === "refine") {\n      const p = document.createElement("p"); p.className = "refine-p";\n      const meta = document.createElement("span"); meta.className = "meta";\n      const spk = ev.speaker ? ev.speaker + " · " : "";\n      meta.innerHTML = spk + "<i>" + (ev.lang || "?") + "</i>";\n      p.appendChild(meta); p.appendChild(document.createTextNode(ev.text));\n      refined.appendChild(p); stick(refined);\n    }\n  };\n</script>\n</body>\n</html>\n'


# cap on a POST body for the live-dictionary endpoints below: these are
# small hand-edited JSON dicts, not file uploads, so anything past this is
# almost certainly a mistake (or worth rejecting rather than blocking on).
_MAX_DICT_BODY_BYTES = 1_000_000


class EventHub:
    """Pub/sub core for structured session events (GitHub issue #29).

    Split out of SubtitleServer so every structured event realtime_
    transcribe.py's pipeline produces (partial/final/translation/refine/
    model_load/model_fallback/warning/session_summary/recluster/
    session_reset) is available to an app embedding the engine whether or
    not `--serve`'s HTTP/SSE layer exists: realtime_transcribe.main()
    always constructs one of these, and SubtitleServer optionally fronts
    it with an HTTP server on top (see that class below).

    Two independent ways to consume events:
      * subscribe()/unsubscribe() -- pull a queue.Queue of JSON strings,
        one per event, in publish order. This is what the /events SSE
        handler and ws_ingest.py's WebSocket mirror use.
      * add_listener() -- push a plain Python dict to a synchronous
        in-process callback, no serialization round-trip. This is what an
        embedding app that links against this module directly (rather
        than talking HTTP) should use.
    """

    def __init__(self):
        self._clients: list[queue.Queue] = []
        self._listeners: list[Callable[[dict], None]] = []
        self._refines: list[str] = []  # recent refine events, replayed to new clients
        self._lock = threading.Lock()
        self._listener_errored: set[int] = set()  # id(callback) already reported to stderr

    def publish(self, event: dict) -> None:
        data = json.dumps(event, ensure_ascii=False)
        with self._lock:
            if event.get("type") == "refine":
                self._refines.append(data)
                del self._refines[:-200]
            clients = list(self._clients)
            listeners = list(self._listeners)
        for q in clients:
            q.put(data)
        # Listener callbacks run AFTER the lock is released (and over a
        # snapshot of the listener list) so a slow, re-entrant, or
        # exception-raising listener can never block publish()'s own
        # bookkeeping, or a concurrent subscribe()/add_listener() call.
        for cb in listeners:
            try:
                cb(event)
            except Exception as exc:
                key = id(cb)
                if key not in self._listener_errored:
                    # Reported once per callback, not once per event: a
                    # listener that fails on every publish() would
                    # otherwise spam stderr once per subtitle segment for
                    # the rest of the session.
                    self._listener_errored.add(key)
                    print(f"[hayamimi] event listener raised (further errors from "
                          f"this listener are suppressed): {exc!r}", file=sys.stderr)

    def partial(self, text: str) -> None:
        self.publish({"type": "partial", "text": text})

    def final(self, text: str, lang: str = "", speaker: str = "",
              latency_ms: float | None = None, tier: str = "",
              audio_s: float | None = None, lid_ms: float | None = None,
              decode_ms: float | None = None, switched: bool = False) -> None:
        self.publish({"type": "final", "text": text, "lang": lang, "speaker": speaker,
                      "latency_ms": latency_ms, "tier": tier, "audio_s": audio_s,
                      "lid_ms": lid_ms, "decode_ms": decode_ms, "switched": switched})

    def subscribe(self) -> queue.Queue:
        """Register a new consumer; past `refine` events are replayed first.

        Used by the /events SSE handler and by ws_ingest.py to mirror the
        same broadcast onto a WebSocket ingest client.
        """
        q: queue.Queue = queue.Queue()
        with self._lock:
            for past in self._refines:
                q.put(past)
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """Register an in-process callback invoked synchronously on every
        publish(), once per event, with the raw event dict (not the JSON
        string subscribe()'s queue carries). See the class docstring for
        when to use this instead of subscribe()."""
        with self._lock:
            self._listeners.append(callback)


class RuntimeControls:
    """Bundles the runtime-configurable pieces of a live realtime_
    transcribe.py session so SubtitleServer's GET/POST /config and POST
    /reset handlers can read and apply them without this module importing
    realtime_transcribe.py (this module stays a leaf: stdlib imports
    only, so it keeps importing cleanly in the CI environment described in
    the project's contributing docs).

    Attach an instance to `server.controls` after construction (same
    pattern as `server.asr` for /replacements and /itn_overrides); every
    field defaults to None so a caller can wire up only the pieces it
    actually has (e.g. no `live_vad` handle when driving the pipeline from
    a fixed --wav source with no live capture VAD to reconfigure). GET/POST
    /config and POST /reset are 404 when no controls are attached at all,
    and each individual key's setter call is skipped in favor of a 400 if
    the specific handle it needs (asr / translator_pool / live_vad /
    reset_fn) wasn't wired.

      asr             -- RoutedASR instance: forced_lang/dual_confirm/
                         punctuate/lid_switch_confirm/min_switch_s setters
      translator_pool -- TranslatorPool instance: add_target/remove_target
      live_vad        -- realtime_transcribe.LiveVad instance:
                         set_sensitivity
      reset_fn        -- zero-argument callable performing a full session
                         reset (a closure over reset_live_session() and
                         its other arguments, built by the caller)
    """

    def __init__(self, asr=None, translator_pool=None, live_vad=None, reset_fn=None):
        self.asr = asr
        self.translator_pool = translator_pool
        self.live_vad = live_vad
        self.reset_fn = reset_fn

    def get_config(self) -> dict:
        cfg = {"lang": None, "dual_confirm": None, "punctuate": None,
               "lid_switch_confirm": None, "min_switch_s": None,
               "translate": [], "vad": None}
        if self.asr is not None:
            cfg["lang"] = self.asr.forced_lang
            cfg["dual_confirm"] = self.asr.dual_confirm
            cfg["punctuate"] = self.asr._punctuate
            cfg["lid_switch_confirm"] = self.asr.lid_switch_confirm
            cfg["min_switch_s"] = self.asr.min_switch_s
        if self.translator_pool is not None:
            cfg["translate"] = self.translator_pool.targets
        if self.live_vad is not None:
            cfg["vad"] = self.live_vad.current_params()
        return cfg

    def apply_config(self, payload: dict) -> None:
        """Apply any subset of GET /config's keys from `payload`.

        Each key is applied in the order below and validated by the
        underlying setter it calls; the first invalid value raises
        ValueError immediately (the caller turns that into a 400), and
        whichever keys were already applied before that point stay
        applied -- this is a thin pass-through over each piece's own
        setter, not a transaction, matching how the individual setters
        already validate their own single value with no cross-key
        coordination needed.
        """
        if "lang" in payload:
            if self.asr is None:
                raise ValueError("no ASR engine attached")
            self.asr.set_forced_lang(payload["lang"])
        if "dual_confirm" in payload:
            if self.asr is None:
                raise ValueError("no ASR engine attached")
            self.asr.set_dual_confirm(bool(payload["dual_confirm"]))
        if "punctuate" in payload:
            if self.asr is None:
                raise ValueError("no ASR engine attached")
            self.asr.set_punctuate(bool(payload["punctuate"]))
        if "lid_switch_confirm" in payload:
            if self.asr is None:
                raise ValueError("no ASR engine attached")
            self.asr.set_lid_switch_confirm(payload["lid_switch_confirm"])
        if "min_switch_s" in payload:
            if self.asr is None:
                raise ValueError("no ASR engine attached")
            self.asr.set_min_switch_s(payload["min_switch_s"])
        if "translate" in payload:
            if self.translator_pool is None:
                raise ValueError("no translator pool attached")
            wanted = payload["translate"]
            if not isinstance(wanted, list) or not all(isinstance(x, str) for x in wanted):
                raise ValueError('"translate" must be a list of language codes')
            current = set(self.translator_pool.targets)
            wanted_set = set(wanted)
            for lang in wanted_set - current:
                self.translator_pool.add_target(lang)  # raises ValueError on an unsupported code
            for lang in current - wanted_set:
                self.translator_pool.remove_target(lang)
        if "vad" in payload:
            if self.live_vad is None:
                raise ValueError("no VAD handle attached")
            vad_cfg = payload["vad"]
            if not isinstance(vad_cfg, dict):
                raise ValueError('"vad" must be an object')
            self.live_vad.set_sensitivity(
                threshold=vad_cfg.get("threshold"),
                min_silence=vad_cfg.get("min_silence"),
                max_speech=vad_cfg.get("max_speech"))

    def reset(self) -> bool:
        """Run the session reset. Returns True if it completed, False if
        `reset_fn` reports it's still pending (GitHub issue #29 review
        round 1: reset_live_session() has to run on the same thread as the
        decode loop, not this HTTP handler thread -- see that function's
        own docstring -- so `reset_fn` typically submits it to a control
        queue and waits with a timeout; a timeout means it's still queued,
        not that it failed). The HTTP handler below turns False into a
        202, not a 500 or a false-positive 200.
        """
        if self.reset_fn is None:
            raise ValueError("no reset function attached")
        return bool(self.reset_fn())


class SubtitleServer:
    """HTTP/SSE front end over an EventHub.

    Fans a session's structured events out to any number of browser SSE
    clients (the OBS overlay, the dashboard, the transcript view) and
    exposes a handful of small JSON control endpoints. All of the actual
    pub/sub bookkeeping lives in EventHub (`self.hub`, constructed
    automatically if not passed in); this class is purely the HTTP
    plumbing on top, so realtime_transcribe.main() can create an EventHub
    unconditionally and only add this HTTP layer when `--serve` was
    passed.

    Optionally fronts a RoutedASR instance (set via `.asr` after
    construction, e.g. by realtime_transcribe.py once the engine is built)
    to expose its live replacement dictionary and ITN overrides over two
    tiny JSON endpoints, so an external UI can edit them mid-session:

      GET  /replacements            -> {"wrong": "right", ...}
      POST /replacements            -> replaces the whole dictionary
      GET  /itn_overrides           -> {"exclude": [...], "force": {...}}
      POST /itn_overrides           -> replaces the whole overrides object

    Both endpoints are no-ops (404) when no `.asr` is attached. Similarly,
    a RuntimeControls instance set via `.controls` (see that class) backs:

      GET  /config                  -> current runtime config
      POST /config                  -> apply any subset of it
      POST /reset                   -> run a full session reset

    all 404 when no `.controls` is attached.
    """

    def __init__(self, port: int = 8833, hub: "EventHub | None" = None):
        self.port = port
        self.hub = hub if hub is not None else EventHub()
        self._httpd = None
        self.asr = None       # RoutedASR, attached after construction if --serve
        self.controls = None  # RuntimeControls, attached after construction if --serve

    def publish(self, event: dict):
        self.hub.publish(event)

    def partial(self, text: str):
        self.hub.partial(text)

    def final(self, text: str, lang: str = "", speaker: str = "",
              latency_ms: float | None = None, tier: str = "",
              audio_s: float | None = None, lid_ms: float | None = None,
              decode_ms: float | None = None, switched: bool = False):
        self.hub.final(text, lang, speaker, latency_ms, tier,
                       audio_s=audio_s, lid_ms=lid_ms, decode_ms=decode_ms, switched=switched)

    def subscribe(self) -> queue.Queue:
        """Register a new consumer; past `refine` events are replayed first.

        Used by the /events SSE handler and by ws_ingest.py to mirror the
        same broadcast onto a WebSocket ingest client.
        """
        return self.hub.subscribe()

    def unsubscribe(self, q: queue.Queue):
        self.hub.unsubscribe(q)

    def start(self):
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console clean for subtitles
                pass

            def do_GET(self):
                if self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    q = server.subscribe()
                    try:
                        while True:
                            data = q.get()
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                            self.wfile.flush()
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                        pass
                    finally:
                        server.unsubscribe(q)
                elif self.path == "/dashboard":
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/transcript":
                    body = TRANSCRIPT_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/replacements":
                    if server.asr is None:
                        self._json(404, {"error": "no ASR engine attached"})
                        return
                    self._json(200, dict(server.asr._replacements))
                elif self.path == "/itn_overrides":
                    if server.asr is None:
                        self._json(404, {"error": "no ASR engine attached"})
                        return
                    overrides = server.asr._itn_overrides
                    self._json(200, {"exclude": sorted(overrides.exclude),
                                     "force": overrides.force})
                elif self.path == "/config":
                    if server.controls is None:
                        self._json(404, {"error": "no runtime controls attached"})
                        return
                    self._json(200, server.controls.get_config())
                else:
                    body = OVERLAY_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def do_POST(self):
                if self.path == "/reset":
                    if server.controls is None:
                        self._json(404, {"error": "no runtime controls attached"})
                        return
                    try:
                        completed = server.controls.reset()
                    except ValueError as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    if completed:
                        self._json(200, {"ok": True})
                    else:
                        # GitHub issue #29 review round 1: the reset was
                        # queued but hasn't run yet (reset_fn's own timeout
                        # elapsed -- see RuntimeControls.reset()'s
                        # docstring) -- 202 Accepted, not a failure.
                        self._json(202, {"ok": False, "pending": True})
                    return
                if self.path not in ("/replacements", "/itn_overrides", "/config"):
                    self._json(404, {"error": "not found"})
                    return
                if self.path == "/config":
                    if server.controls is None:
                        self._json(404, {"error": "no runtime controls attached"})
                        return
                elif server.asr is None:
                    self._json(404, {"error": "no ASR engine attached"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > _MAX_DICT_BODY_BYTES:
                    self._json(400, {"error": "missing or oversized body"})
                    return
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    self._json(400, {"error": f"invalid JSON: {exc}"})
                    return
                if not isinstance(payload, dict):
                    self._json(400, {"error": "body must be a JSON object"})
                    return
                if self.path == "/replacements":
                    server.asr.set_replacements(payload)
                    self._json(200, {"ok": True})
                elif self.path == "/itn_overrides":
                    exclude = payload.get("exclude") or []
                    force = payload.get("force") or {}
                    if not isinstance(exclude, list) or not isinstance(force, dict):
                        self._json(400, {"error": '"exclude" must be a list, "force" a dict'})
                        return
                    server.asr.set_itn_overrides(exclude=set(exclude), force=force)
                    self._json(200, {"ok": True})
                else:  # /config
                    try:
                        server.controls.apply_config(payload)
                    except (ValueError, TypeError) as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    self._json(200, {"ok": True})

            def _json(self, status: int, payload: dict):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self
