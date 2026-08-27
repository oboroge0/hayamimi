/// Transparent-background overlay page served at `/`, for use as an OBS
/// browser source. Ported from `OVERLAY_HTML` in `scripts/subtitle_server.py`
/// so both the desktop and mobile subtitle servers look identical to OBS.
const String overlayHtml = '''
<!doctype html>
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
  // confirmed line and the in-progress line as two independent sources --
  // same convention as the desktop overlay (scripts/subtitle_server.py).
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
''';
