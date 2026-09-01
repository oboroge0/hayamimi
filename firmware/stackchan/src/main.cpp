// hayamimi stackchan client (M5Stack CoreS3)
//
// Captures the CoreS3's built-in mic at 16kHz mono and streams it over a
// raw WebSocket to hayamimi's /ingest endpoint, then shows the returned
// subtitle events (partial/final/refine/translation) on the display.
//
// Wire protocol (do not change without updating the server -- see
// ../../scripts/ws_protocol.py and ../../scripts/ws_ingest.py, which are
// the source of truth):
//   1. RFC 6455 WS handshake to ws://HAYAMIMI_HOST:HAYAMIMI_PORT/ingest.
//   2. First frame: one JSON text frame {"sr":16000,"format":"pcm_s16le",
//      "channels":1}.
//   3. After that: binary frames of raw little-endian s16 PCM, sent
//      continuously for the life of the connection (silence included --
//      that's what lets the server's VAD ever see a pause and finalize).
//   4. Server replies with JSON text frames: {"type": "partial"/"final"/
//      "translation"/"refine"/"ready"/"error", "text": ..., ...}.
//
// Library choice: links2004/WebSockets (Arduino WebSockets by Links2004).
// Picked over gilmaimon/ArduinoWebsockets because it ships built-in
// reconnect handling (setReconnectInterval) which this always-on mic
// client needs, has a stable long-running maintenance history, and its
// callback signature (WStype_t, payload, length) cleanly separates text
// vs. binary frames, matching this protocol's text-then-binary shape.
#include <M5Unified.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

#include "config.h"

namespace {

constexpr uint32_t SAMPLE_RATE = 16000;
// 100ms per chunk -- matches ws_mic_client.py's CHUNK_S, a size the server
// side has already been exercised against.
constexpr size_t CHUNK_SAMPLES = SAMPLE_RATE / 10;
constexpr size_t CHUNK_BYTES = CHUNK_SAMPLES * sizeof(int16_t);

WebSocketsClient webSocket;
SemaphoreHandle_t wsMutex;      // guards all webSocket.* calls (loop/send are called from two tasks)
SemaphoreHandle_t displayMutex; // guards canvas draw + push

volatile bool wsConnected = false;
volatile bool handshakeSent = false;
// Set by webSocketEvent(WStype_CONNECTED) (called synchronously from inside
// webSocket.loop(), i.e. while loop() already holds wsMutex) and consumed by
// loop() itself, once wsMutex has been released. See sendHandshake() for why
// the handshake can't be sent directly from the event callback.
volatile bool needHandshake = false;

M5Canvas canvas(&M5.Display);

String finalText;
String partialText;
String statusText = "starting...";

void redraw() {
  if (xSemaphoreTake(displayMutex, pdMS_TO_TICKS(200)) != pdTRUE) return;

  canvas.fillScreen(TFT_BLACK);

  canvas.setFont(&fonts::lgfxJapanGothic_16);
  canvas.setTextColor(wsConnected ? TFT_DARKGREEN : TFT_RED);
  canvas.setCursor(4, 4);
  canvas.println(statusText);

  canvas.setFont(&fonts::lgfxJapanGothic_24);
  canvas.setTextColor(TFT_WHITE);
  canvas.setCursor(4, 30);
  canvas.println(finalText);

  if (partialText.length() > 0) {
    canvas.setFont(&fonts::lgfxJapanGothic_20);
    canvas.setTextColor(TFT_DARKGREY);
    canvas.setCursor(4, canvas.getCursorY() + 6);
    canvas.println(partialText);
  }

  canvas.pushSprite(0, 0);
  xSemaphoreGive(displayMutex);
}

void setStatus(const String &s) {
  statusText = s;
  redraw();
}

void handleEventJson(const uint8_t *payload, size_t len) {
  // Reused across calls instead of a fresh JsonDocument per event: this
  // fires once per server text frame (partial/final/refine/... -- i.e.
  // continuously for the life of a session), and ArduinoJson v7's
  // JsonDocument owns a heap-backed memory pool that grows on demand but
  // is only released when the document is destroyed. A local (stack)
  // JsonDocument would therefore malloc/free that pool on every single
  // event, which is exactly the allocate/free churn that fragments the
  // ESP32's heap over a long-running session. `static` keeps one instance
  // (and its already-grown pool) alive for the process's lifetime, and
  // clear() below empties its contents without releasing the pool, so
  // steady-state parsing does no allocation at all. Safe as a function-
  // static here because handleEventJson only ever runs on the main loop()
  // task (via webSocketEvent() <- webSocket.loop(), see loop()) -- never
  // from micTask on the other core -- so there's no concurrent access to
  // guard against.
  static JsonDocument doc;
  doc.clear();
  if (deserializeJson(doc, payload, len)) return; // malformed frame: drop silently

  const char *type = doc["type"] | "";
  const char *text = doc["text"] | "";

  if (strcmp(type, "final") == 0) {
    finalText = text;
    partialText = "";
    redraw();
  } else if (strcmp(type, "partial") == 0) {
    partialText = text;
    redraw();
  } else if (strcmp(type, "refine") == 0) {
    // refine replaces the text of an already-shown final with a
    // higher-quality re-decode of the same utterance.
    finalText = text;
    redraw();
  } else if (strcmp(type, "translation") == 0) {
    setStatus(String("[tr] ") + text);
  } else if (strcmp(type, "error") == 0) {
    setStatus(String("server: ") + (doc["message"] | ""));
  } else if (strcmp(type, "ready") == 0) {
    setStatus("connected");
  }
  // other event types (e.g. session_start) are ignored
}

// IMPORTANT: never call this from inside webSocketEvent() (or anywhere else
// that may already hold wsMutex). webSocketEvent() is invoked synchronously
// from within webSocket.loop(), which loop() calls while holding wsMutex;
// since wsMutex is a plain (non-recursive) mutex, taking it again here would
// deadlock loop() forever. Callers must already hold wsMutex, or -- as
// loop() does -- call this only after releasing it.
void sendHandshake() {
  JsonDocument doc;
  doc["sr"] = SAMPLE_RATE;
  doc["format"] = "pcm_s16le";
  doc["channels"] = 1;
  char buf[128];
  size_t n = serializeJson(doc, buf, sizeof(buf));

  if (xSemaphoreTake(wsMutex, portMAX_DELAY) == pdTRUE) {
    webSocket.sendTXT(buf, n);
    xSemaphoreGive(wsMutex);
  }
  handshakeSent = true;
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  // NOTE: this callback fires synchronously from inside webSocket.loop(),
  // which loop() calls while holding wsMutex. Never take wsMutex from here
  // (directly or via a function like sendHandshake()) -- it would deadlock
  // against loop(). Defer any such work with a flag that loop() polls after
  // releasing wsMutex (see needHandshake below).
  switch (type) {
    case WStype_CONNECTED:
      wsConnected = true;
      handshakeSent = false;
      needHandshake = true;
      setStatus("ws connected, handshaking...");
      break;
    case WStype_DISCONNECTED:
      wsConnected = false;
      handshakeSent = false;
      setStatus("disconnected, reconnecting...");
      break;
    case WStype_TEXT:
      handleEventJson(payload, length);
      break;
    case WStype_ERROR:
      setStatus("ws error");
      break;
    default:
      break;
  }
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  setStatus(String("wifi: connecting to ") + WIFI_SSID);
  uint32_t waitStart = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    if (millis() - waitStart > 20000) {
      // Don't give up -- just let the user see it's still trying.
      setStatus(String("wifi: still trying (") + WIFI_SSID + ")");
      waitStart = millis();
    }
  }
  setStatus("wifi connected, opening ws...");
}

// Runs on core 0: capture one mic chunk, send it if we have an open,
// handshaked connection. Pinned to its own core so a blocking Mic.record()
// call never stalls webSocket.loop() (which drives reconnects) on core 1.
void micTask(void *) {
  static int16_t buf[CHUNK_SAMPLES];
  for (;;) {
    if (!M5.Mic.isEnabled()) {
      vTaskDelay(pdMS_TO_TICKS(200));
      continue;
    }
    if (M5.Mic.record(buf, CHUNK_SAMPLES, SAMPLE_RATE)) {
      if (wsConnected && handshakeSent) {
        if (xSemaphoreTake(wsMutex, pdMS_TO_TICKS(500)) == pdTRUE) {
          webSocket.sendBIN(reinterpret_cast<uint8_t *>(buf), CHUNK_BYTES);
          xSemaphoreGive(wsMutex);
        }
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
}

} // namespace

void setup() {
  wsMutex = xSemaphoreCreateMutex();
  displayMutex = xSemaphoreCreateMutex();

  auto cfg = M5.config();
  M5.begin(cfg);

  canvas.setColorDepth(8);
  canvas.createSprite(M5.Display.width(), M5.Display.height());
  canvas.setTextWrap(true, false);
  redraw();

  auto micCfg = M5.Mic.config();
  micCfg.sample_rate = SAMPLE_RATE;
  micCfg.stereo = false;
  M5.Mic.config(micCfg);
  M5.Mic.begin();

  connectWifi();

  webSocket.begin(HAYAMIMI_HOST, HAYAMIMI_PORT, HAYAMIMI_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(3000);

  xTaskCreatePinnedToCore(micTask, "mic", 8192, nullptr, 2, nullptr, 0);
}

void loop() {
  M5.update();

  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }

  if (xSemaphoreTake(wsMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
    webSocket.loop();
    xSemaphoreGive(wsMutex);
  }

  // Handled here, after releasing wsMutex, because webSocketEvent() may have
  // set this from inside the webSocket.loop() call above while wsMutex was
  // still held -- see the comment on sendHandshake().
  if (needHandshake) {
    needHandshake = false;
    sendHandshake();
  }

  delay(2);
}
