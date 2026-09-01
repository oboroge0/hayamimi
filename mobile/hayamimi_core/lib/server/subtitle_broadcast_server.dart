import 'dart:convert';
import 'dart:io';

import 'overlay_html.dart';
import 'subtitle_event.dart';

/// LAN-facing HTTP server that mirrors the desktop hayamimi subtitle feed
/// (`scripts/subtitle_server.py`): an SSE stream at `/events` carrying the
/// same JSON event shape (`{"type": "final", "text": ..., "lang": ...,
/// "speaker": ..., "latency_ms": ...}`), and a transparent overlay page at
/// `/` suitable for an OBS browser source. Any device on the same LAN
/// (OBS, a browser) can subscribe to this phone's live transcript.
///
/// Binds to [bindAddress], which defaults to `0.0.0.0` (all interfaces)
/// rather than loopback, since the whole point of this server is
/// reachability from *other* devices on the network — a phone streaming
/// subtitles to OBS or a browser running on a different machine on the same
/// Wi-Fi. Pass a narrower [bindAddress] (e.g.
/// `InternetAddress.loopbackIPv4`) if a host app wants to restrict this to
/// same-device consumers only, though that defeats the LAN-broadcast use
/// case this class exists for.
class SubtitleBroadcastServer {
  SubtitleBroadcastServer({
    this.port = defaultPort,
    InternetAddress? bindAddress,
    this.allowOrigin = '*',
  }) : bindAddress = bindAddress ?? InternetAddress.anyIPv4;

  static const int defaultPort = 8833;

  /// Requested port. Pass `0` (used by tests) to bind an OS-assigned
  /// ephemeral port instead — read the actual port back via [boundPort].
  final int port;

  /// Interface to bind the listener to. Defaults to
  /// [InternetAddress.anyIPv4] — see the class doc for why.
  final InternetAddress bindAddress;

  /// Value sent as the `/events` response's `Access-Control-Allow-Origin`
  /// header. Defaults to `'*'` (any origin may fetch the SSE stream cross-
  /// origin), matching the desktop `subtitle_server.py`'s behavior and
  /// keeping a browser-based OBS source or a web dashboard on another
  /// origin working without extra configuration. Narrow this if a host app
  /// needs to restrict which origins may read its transcript.
  final String allowOrigin;

  HttpServer? _httpServer;
  // [isRunning] only flips once the bind completes, so without this a
  // second concurrent start() would sail past it and bind a second server
  // (or, on a fixed port, fail with an address-in-use the caller didn't
  // ask for) while the first one's HttpServer is dropped on the floor.
  bool _starting = false;
  final List<HttpResponse> _clients = [];

  bool get isRunning => _httpServer != null;

  /// The port actually bound once [start] has completed, or `null` if not
  /// running. Differs from [port] only when [port] was `0`.
  int? get boundPort => _httpServer?.port;

  /// Binds the LAN listener. A call made while the server is already
  /// running — or while an earlier [start] is still binding — is a no-op.
  Future<void> start() async {
    if (isRunning || _starting) {
      return;
    }
    _starting = true;
    try {
      final server = await HttpServer.bind(bindAddress, port);
      _httpServer = server;
      server.listen(_handleRequest, onError: (_) {});
    } finally {
      _starting = false;
    }
  }

  Future<void> stop() async {
    final server = _httpServer;
    _httpServer = null;
    if (server == null) {
      return;
    }
    for (final client in List.of(_clients)) {
      await client.close().catchError((_) {});
    }
    _clients.clear();
    await server.close(force: true);
  }

  /// Sends [event] to every currently-connected `/events` client.
  void broadcast(SubtitleEvent event) {
    final frame = utf8.encode(event.toSseFrame());
    for (final client in List.of(_clients)) {
      try {
        client.add(frame);
      } catch (_) {
        _clients.remove(client);
      }
    }
  }

  Future<void> _handleRequest(HttpRequest request) async {
    try {
      if (request.uri.path == '/events') {
        await _handleEvents(request);
      } else {
        await _handleOverlay(request);
      }
    } catch (_) {
      // Client disconnected mid-response or similar; nothing to recover.
    }
  }

  Future<void> _handleEvents(HttpRequest request) async {
    final response = request.response;
    // Explicit charset: HttpResponse.write/add otherwise falls back to
    // Latin-1 for the body, which throws on non-ASCII transcript text.
    response.headers.set(
      HttpHeaders.contentTypeHeader,
      'text/event-stream; charset=utf-8',
    );
    response.headers.set(HttpHeaders.cacheControlHeader, 'no-cache');
    response.headers.set('Access-Control-Allow-Origin', allowOrigin);
    response.bufferOutput = false;

    // Send an initial SSE comment line (ignored by EventSource, per spec)
    // to force the status line + headers out immediately. Without any
    // write, Dart's HttpResponse holds the response open rather than
    // transmitting headers, so the client would otherwise sit waiting for
    // a response that never arrives until the first real broadcast.
    response.add(utf8.encode(': connected\n\n'));
    await response.flush();

    _clients.add(response);
    try {
      // Resolves once the connection closes (client disconnect) or errors.
      await response.done;
    } catch (_) {
      // Expected on client disconnect; the stream just ends.
    } finally {
      _clients.remove(response);
    }
  }

  Future<void> _handleOverlay(HttpRequest request) async {
    final body = utf8.encode(overlayHtml);
    final response = request.response;
    response.headers.set(
      HttpHeaders.contentTypeHeader,
      'text/html; charset=utf-8',
    );
    response.headers.set(HttpHeaders.contentLengthHeader, body.length);
    response.add(body);
    await response.close();
  }
}
