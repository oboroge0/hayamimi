/// One phase of loading a native sherpa-onnx model.
///
/// `LiveTranscriber.start()`/`startDebugWavStream()` used to be one opaque
/// `Future` from a host app's point of view: nothing was observable between
/// "called start()" and "it resolved", even though loading
/// `RoutingProfile.jaSenseVoice`'s three models can take several seconds
/// (see `native_model_loader.dart`). [LiveTranscriber.modelLoads] emits one
/// of these right before and right after each background-isolate build, so
/// a host UI can show e.g. "loading SenseVoice..." instead of a single
/// spinner for the whole session start.
class ModelLoadEvent {
  const ModelLoadEvent({required this.model, required this.phase, this.ms});

  /// Which native model this phase is about: `"vad"` (both the initial
  /// build and any [LiveTranscriber.setVadSensitivity] rebuild),
  /// `"recognizer"` (the plain, non-routed path's single model), for
  /// [RoutingProfile.jaSenseVoice]: `"ja"`, `"sensevoice"`, `"lid"`, and
  /// `"punct"` (the Japanese punctuation model, when `start` was given a
  /// `JaPunctuation` — it loads last, after the recognizers). `"punct"`'s
  /// `ms` on `"done"` includes one warm-up inference run right after the
  /// model loads, so the model's first real `restore()` call in this
  /// process is fast: without it, the ~300 ms a cold ONNX Runtime session
  /// pays on its first run would land on the first punctuated final or
  /// refine instead of here.
  final String model;

  /// `"start"` right before the background-isolate build call, `"done"`
  /// right after it returns.
  final String phase;

  /// Wall-clock milliseconds the build took. `null` on `"start"`; always
  /// set on `"done"`.
  final double? ms;
}
