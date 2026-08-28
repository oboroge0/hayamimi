/// Reusable core of hayamimi's mobile speech recognition pipeline:
/// on-device live transcription, a thin client for a remote hayamimi
/// server, and a LAN subtitle broadcast server — see the package README
/// for embedding instructions.
library;

// High-level facades: start here for embedding hayamimi into another app.
export 'hayamimi_live.dart';
export 'hayamimi_remote.dart';

// Shared event/result types.
export 'live/live_transcript_entry.dart';
export 'remote/remote_connection_state.dart';
export 'remote/remote_event.dart';
export 'server/subtitle_event.dart';

// Lower-level building blocks, exposed for callers that want more control
// than the HayamimiLive/HayamimiRemote facades give.
export 'bench/bench_result.dart';
export 'bench/bench_runner.dart';
export 'bench/manifest_eval_result.dart';
export 'bench/manifest_eval_runner.dart';
export 'bench/model_file_resolver.dart';
export 'bench/model_kind.dart';
export 'live/draft_pass.dart';
export 'live/live_transcriber.dart';
export 'live/pcm_frame_buffer.dart' show pcm16BytesToFloat32, PcmFrameBuffer;
export 'live/refine_pass.dart';
export 'live/speech_segment_filter.dart' show isSegmentWorthDecoding;
export 'remote/remote_handshake.dart';
export 'remote/remote_transcriber.dart';
export 'remote/wav_pcm_reader.dart';
export 'routing/lang_routing.dart';
export 'routing/routed_recognizer.dart';
export 'routing/routing_profile.dart';
export 'server/lan_address.dart';
export 'server/subtitle_broadcast_server.dart';
export 'setup/model_downloader.dart';
