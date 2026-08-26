/// Lifecycle of the mic-streaming connection to a hayamimi `/ingest` server.
enum RemoteConnectionState {
  /// Not connected and not trying to be.
  disconnected,

  /// First connection attempt in flight.
  connecting,

  /// Handshake sent, mic streaming (if enabled), receiving events.
  connected,

  /// Connection dropped unexpectedly while still desired; retrying with a
  /// fixed backoff until [RemoteTranscriber.disconnect] is called.
  reconnecting,
}
