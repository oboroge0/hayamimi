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
  #partial { opacity: 0.75; font-style: italic; }
</style>
<div id="box"><span id="final"></span> <span id="partial"></span></div>
<script>
  const fin = document.getElementById("final");
  const par = document.getElementById("partial");
  let clearTimer = null;
  const es = new EventSource("/events");
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (clearTimer) { clearTimeout(clearTimer); clearTimer = null; }
    if (ev.type === "partial") {
      par.textContent = ev.text;
    } else if (ev.type === "final") {
      fin.textContent = ev.text;
      par.textContent = "";
      clearTimer = setTimeout(() => { fin.textContent = ""; }, 6000);
    }
  };
</script>
''';
