import 'dart:ffi' as ffi;
import 'dart:io' show File, Platform;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'ort_bindings.dart';
import 'ort_library.dart';

/// Thrown when ONNX Runtime reports an error, or when the loaded model does
/// not look like the punctuation model this package expects.
class OrtSessionException implements Exception {
  OrtSessionException(this.message);

  final String message;

  @override
  String toString() => 'OrtSessionException: $message';
}

/// Runs the Japanese punctuation model through ONNX Runtime's C API.
///
/// This is the whole native surface of the punctuation feature: everything
/// above it — tokenizing, thresholding, inserting marks — is plain Dart.
/// The class takes a list of token ids and returns the model's raw logits,
/// two per input position.
///
/// The model is loaded from a file path rather than from bytes on purpose.
/// The punctuation model is 182 MB; reading it into a Dart list first would
/// mean holding that much in the Dart heap and copying it again on the way
/// into the runtime, when ONNX Runtime can memory-map the file itself.
class PunctOrtSession {
  PunctOrtSession._(
    this._api,
    this._env,
    this._session,
    this._memoryInfo,
    this._inputNames,
    this._outputNames,
    this.runtimeVersion,
  );

  /// The model's two inputs and one output, in the order the graph
  /// declares them. Checked against the loaded file at open time so a
  /// wrong model fails with a readable message instead of an ORT error.
  static const List<String> inputNames = <String>['input_ids', 'attention_mask'];
  static const String outputName = 'logits';

  final ffi.Pointer<OrtApi> _api;
  final ffi.Pointer<OrtEnv> _env;
  final ffi.Pointer<OrtSession> _session;
  final ffi.Pointer<OrtMemoryInfo> _memoryInfo;
  final ffi.Pointer<ffi.Pointer<ffi.Char>> _inputNames;
  final ffi.Pointer<ffi.Pointer<ffi.Char>> _outputNames;

  /// The ONNX Runtime version that actually answered, e.g. `1.27.1`.
  final String runtimeVersion;

  bool _disposed = false;

  /// Opens [modelPath] with ONNX Runtime's CPU provider.
  ///
  /// [libraryPath] is passed straight to [OrtLibrary.open]; see there for
  /// per-platform defaults. [intraOpNumThreads] is how many threads a
  /// single operator may use — kept small because this runs next to the
  /// speech recognizer, which wants the cores.
  factory PunctOrtSession.open(
    String modelPath, {
    String? libraryPath,
    int intraOpNumThreads = 2,
  }) {
    if (!File(modelPath).existsSync()) {
      throw OrtSessionException('Model file not found: $modelPath');
    }
    final OrtLibrary lib = OrtLibrary.open(libraryPath: libraryPath);
    final ffi.Pointer<OrtApi> api = lib.api;

    final ffi.Pointer<ffi.Pointer<OrtEnv>> envOut = calloc<ffi.Pointer<OrtEnv>>();
    final ffi.Pointer<ffi.Pointer<OrtSessionOptions>> optionsOut =
        calloc<ffi.Pointer<OrtSessionOptions>>();
    final ffi.Pointer<ffi.Pointer<OrtSession>> sessionOut =
        calloc<ffi.Pointer<OrtSession>>();
    final ffi.Pointer<ffi.Pointer<OrtMemoryInfo>> memoryOut =
        calloc<ffi.Pointer<OrtMemoryInfo>>();
    final ffi.Pointer<Utf8> logId = 'hayamimi_punct'.toNativeUtf8();
    // ORTCHAR_T is wchar_t on Windows and char everywhere else, so the
    // model path has to be encoded differently per platform. Getting this
    // wrong hands ONNX Runtime a mangled path and it reports a missing
    // file, which is a confusing way to learn about an encoding bug.
    final ffi.Pointer<ffi.Void> nativeModelPath = Platform.isWindows
        ? modelPath.toNativeUtf16().cast<ffi.Void>()
        : modelPath.toNativeUtf8().cast<ffi.Void>();

    ffi.Pointer<OrtEnv> env = ffi.nullptr;
    ffi.Pointer<OrtSessionOptions> options = ffi.nullptr;
    ffi.Pointer<OrtSession> session = ffi.nullptr;
    ffi.Pointer<OrtMemoryInfo> memoryInfo = ffi.nullptr;
    try {
      _check(
        api,
        api.ref.CreateEnv
            .asFunction<
              OrtStatusPtr Function(
                int,
                ffi.Pointer<ffi.Char>,
                ffi.Pointer<ffi.Pointer<OrtEnv>>,
              )
            >()(ortLoggingLevelWarning, logId.cast<ffi.Char>(), envOut),
        'CreateEnv',
      );
      env = envOut.value;

      _check(
        api,
        api.ref.CreateSessionOptions
            .asFunction<
              OrtStatusPtr Function(ffi.Pointer<ffi.Pointer<OrtSessionOptions>>)
            >()(optionsOut),
        'CreateSessionOptions',
      );
      options = optionsOut.value;

      _check(
        api,
        api.ref.SetIntraOpNumThreads
            .asFunction<
              OrtStatusPtr Function(ffi.Pointer<OrtSessionOptions>, int)
            >()(options, intraOpNumThreads),
        'SetIntraOpNumThreads',
      );
      _check(
        api,
        api.ref.SetInterOpNumThreads
            .asFunction<
              OrtStatusPtr Function(ffi.Pointer<OrtSessionOptions>, int)
            >()(options, 1),
        'SetInterOpNumThreads',
      );

      // No execution provider is appended: with none registered ONNX
      // Runtime falls back to the CPU provider, which is the only one
      // available in the sherpa_onnx builds anyway.
      _check(
        api,
        api.ref.CreateSession
            .asFunction<
              OrtStatusPtr Function(
                ffi.Pointer<OrtEnv>,
                ffi.Pointer<ffi.Void>,
                ffi.Pointer<OrtSessionOptions>,
                ffi.Pointer<ffi.Pointer<OrtSession>>,
              )
            >()(env, nativeModelPath, options, sessionOut),
        'CreateSession',
      );
      session = sessionOut.value;

      _check(
        api,
        api.ref.CreateCpuMemoryInfo
            .asFunction<
              OrtStatusPtr Function(
                int,
                int,
                ffi.Pointer<ffi.Pointer<OrtMemoryInfo>>,
              )
            >()(ortArenaAllocator, ortMemTypeDefault, memoryOut),
        'CreateCpuMemoryInfo',
      );
      memoryInfo = memoryOut.value;

      _verifyGraphIo(api, session);
    } catch (_) {
      if (memoryInfo != ffi.nullptr) {
        api.ref.ReleaseMemoryInfo
            .asFunction<void Function(ffi.Pointer<OrtMemoryInfo>)>()(memoryInfo);
      }
      if (session != ffi.nullptr) {
        api.ref.ReleaseSession
            .asFunction<void Function(ffi.Pointer<OrtSession>)>()(session);
      }
      if (env != ffi.nullptr) {
        api.ref.ReleaseEnv.asFunction<void Function(ffi.Pointer<OrtEnv>)>()(env);
      }
      rethrow;
    } finally {
      if (options != ffi.nullptr) {
        // The session copied what it needs out of the options.
        api.ref.ReleaseSessionOptions
            .asFunction<void Function(ffi.Pointer<OrtSessionOptions>)>()(
              options,
            );
      }
      calloc.free(envOut);
      calloc.free(optionsOut);
      calloc.free(sessionOut);
      calloc.free(memoryOut);
      malloc.free(logId);
      malloc.free(nativeModelPath);
    }

    // Input and output names are handed to Run() on every call, so they
    // are encoded once and kept for the life of the session.
    final ffi.Pointer<ffi.Pointer<ffi.Char>> inputNamePtrs =
        calloc<ffi.Pointer<ffi.Char>>(inputNames.length);
    for (int i = 0; i < inputNames.length; i++) {
      inputNamePtrs[i] = inputNames[i].toNativeUtf8().cast<ffi.Char>();
    }
    final ffi.Pointer<ffi.Pointer<ffi.Char>> outputNamePtrs =
        calloc<ffi.Pointer<ffi.Char>>(1);
    outputNamePtrs[0] = outputName.toNativeUtf8().cast<ffi.Char>();

    return PunctOrtSession._(
      api,
      env,
      session,
      memoryInfo,
      inputNamePtrs,
      outputNamePtrs,
      lib.versionString,
    );
  }

  /// Runs the model over [inputIds] and returns its logits, laid out as
  /// `[comma, period]` per input position — so position `i` is at
  /// `result[i * 2]` and `result[i * 2 + 1]`, with the same length as
  /// [inputIds].
  ///
  /// The attention mask is all ones: this model is only ever run on one
  /// sequence at a time, so there is nothing to pad.
  Float32List run(List<int> inputIds) {
    if (_disposed) {
      throw StateError('PunctOrtSession was disposed');
    }
    if (inputIds.isEmpty) {
      throw ArgumentError.value(inputIds, 'inputIds', 'must not be empty');
    }
    final int seqLen = inputIds.length;

    final ffi.Pointer<ffi.Int64> ids = calloc<ffi.Int64>(seqLen);
    final ffi.Pointer<ffi.Int64> mask = calloc<ffi.Int64>(seqLen);
    final ffi.Pointer<ffi.Int64> shape = calloc<ffi.Int64>(2);
    final ffi.Pointer<ffi.Pointer<OrtValue>> inputs =
        calloc<ffi.Pointer<OrtValue>>(2);
    final ffi.Pointer<ffi.Pointer<OrtValue>> outputs =
        calloc<ffi.Pointer<OrtValue>>(1);
    for (int i = 0; i < seqLen; i++) {
      ids[i] = inputIds[i];
      mask[i] = 1;
    }
    shape[0] = 1;
    shape[1] = seqLen;

    try {
      _createInt64Tensor(ids, seqLen, shape, inputs, 0);
      _createInt64Tensor(mask, seqLen, shape, inputs, 1);

      _check(
        _api,
        _api.ref.Run
            .asFunction<
              OrtStatusPtr Function(
                ffi.Pointer<OrtSession>,
                ffi.Pointer<OrtRunOptions>,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                ffi.Pointer<ffi.Pointer<OrtValue>>,
                int,
                ffi.Pointer<ffi.Pointer<ffi.Char>>,
                int,
                ffi.Pointer<ffi.Pointer<OrtValue>>,
              )
            >()(
              _session,
              ffi.nullptr,
              _inputNames,
              inputs,
              inputNames.length,
              _outputNames,
              1,
              outputs,
            ),
        'Run',
      );

      return _readLogits(outputs[0], seqLen);
    } finally {
      for (int i = 0; i < 2; i++) {
        if (inputs[i] != ffi.nullptr) {
          _api.ref.ReleaseValue
              .asFunction<void Function(ffi.Pointer<OrtValue>)>()(inputs[i]);
        }
      }
      if (outputs[0] != ffi.nullptr) {
        _api.ref.ReleaseValue
            .asFunction<void Function(ffi.Pointer<OrtValue>)>()(outputs[0]);
      }
      calloc.free(ids);
      calloc.free(mask);
      calloc.free(shape);
      calloc.free(inputs);
      calloc.free(outputs);
    }
  }

  void _createInt64Tensor(
    ffi.Pointer<ffi.Int64> data,
    int length,
    ffi.Pointer<ffi.Int64> shape,
    ffi.Pointer<ffi.Pointer<OrtValue>> out,
    int index,
  ) {
    final ffi.Pointer<ffi.Pointer<OrtValue>> slot = out + index;
    _check(
      _api,
      _api.ref.CreateTensorWithDataAsOrtValue
          .asFunction<
            OrtStatusPtr Function(
              ffi.Pointer<OrtMemoryInfo>,
              ffi.Pointer<ffi.Void>,
              int,
              ffi.Pointer<ffi.Int64>,
              int,
              int,
              ffi.Pointer<ffi.Pointer<OrtValue>>,
            )
          >()(
            _memoryInfo,
            data.cast<ffi.Void>(),
            length * ffi.sizeOf<ffi.Int64>(),
            shape,
            2,
            onnxTensorElementDataTypeInt64,
            slot,
          ),
      'CreateTensorWithDataAsOrtValue',
    );
  }

  Float32List _readLogits(ffi.Pointer<OrtValue> value, int seqLen) {
    final ffi.Pointer<ffi.Pointer<OrtTensorTypeAndShapeInfo>> infoOut =
        calloc<ffi.Pointer<OrtTensorTypeAndShapeInfo>>();
    ffi.Pointer<OrtTensorTypeAndShapeInfo> info = ffi.nullptr;
    try {
      _check(
        _api,
        _api.ref.GetTensorTypeAndShape
            .asFunction<
              OrtStatusPtr Function(
                ffi.Pointer<OrtValue>,
                ffi.Pointer<ffi.Pointer<OrtTensorTypeAndShapeInfo>>,
              )
            >()(value, infoOut),
        'GetTensorTypeAndShape',
      );
      info = infoOut.value;

      final ffi.Pointer<ffi.Int32> elementType = calloc<ffi.Int32>();
      final ffi.Pointer<ffi.Size> rank = calloc<ffi.Size>();
      try {
        _check(
          _api,
          _api.ref.GetTensorElementType
              .asFunction<
                OrtStatusPtr Function(
                  ffi.Pointer<OrtTensorTypeAndShapeInfo>,
                  ffi.Pointer<ffi.Int32>,
                )
              >()(info, elementType),
          'GetTensorElementType',
        );
        if (elementType.value != onnxTensorElementDataTypeFloat) {
          // The fp16 model was exported with keep_io_types, so its inputs
          // and outputs are still int64/float32 even though the weights
          // are float16. A float16 output here would mean a different
          // export, and reading it as float32 would be silent nonsense.
          throw OrtSessionException(
            'Expected the "$outputName" output to be float32 (ONNX element '
            'type $onnxTensorElementDataTypeFloat) but the model produced '
            'element type ${elementType.value}.',
          );
        }

        _check(
          _api,
          _api.ref.GetDimensionsCount
              .asFunction<
                OrtStatusPtr Function(
                  ffi.Pointer<OrtTensorTypeAndShapeInfo>,
                  ffi.Pointer<ffi.Size>,
                )
              >()(info, rank),
          'GetDimensionsCount',
        );
        if (rank.value != 3) {
          throw OrtSessionException(
            'Expected "$outputName" to have 3 dimensions [1, sequence, 2] '
            'but it has ${rank.value}.',
          );
        }

        final ffi.Pointer<ffi.Int64> dims = calloc<ffi.Int64>(3);
        try {
          _check(
            _api,
            _api.ref.GetDimensions
                .asFunction<
                  OrtStatusPtr Function(
                    ffi.Pointer<OrtTensorTypeAndShapeInfo>,
                    ffi.Pointer<ffi.Int64>,
                    int,
                  )
                >()(info, dims, 3),
            'GetDimensions',
          );
          if (dims[0] != 1 || dims[1] != seqLen || dims[2] != 2) {
            throw OrtSessionException(
              'Expected "$outputName" shape [1, $seqLen, 2] but got '
              '[${dims[0]}, ${dims[1]}, ${dims[2]}].',
            );
          }
        } finally {
          calloc.free(dims);
        }
      } finally {
        calloc.free(elementType);
        calloc.free(rank);
      }

      final ffi.Pointer<ffi.Pointer<ffi.Void>> dataOut =
          calloc<ffi.Pointer<ffi.Void>>();
      try {
        _check(
          _api,
          _api.ref.GetTensorMutableData
              .asFunction<
                OrtStatusPtr Function(
                  ffi.Pointer<OrtValue>,
                  ffi.Pointer<ffi.Pointer<ffi.Void>>,
                )
              >()(value, dataOut),
          'GetTensorMutableData',
        );
        final ffi.Pointer<ffi.Float> floats = dataOut.value.cast<ffi.Float>();
        // Copy out: the buffer belongs to the OrtValue, which is released
        // as soon as this call returns.
        return Float32List.fromList(floats.asTypedList(seqLen * 2));
      } finally {
        calloc.free(dataOut);
      }
    } finally {
      if (info != ffi.nullptr) {
        _api.ref.ReleaseTensorTypeAndShapeInfo
            .asFunction<
              void Function(ffi.Pointer<OrtTensorTypeAndShapeInfo>)
            >()(info);
      }
      calloc.free(infoOut);
    }
  }

  /// Releases the session, the environment and the cached native strings.
  /// Safe to call more than once.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _api.ref.ReleaseMemoryInfo
        .asFunction<void Function(ffi.Pointer<OrtMemoryInfo>)>()(_memoryInfo);
    _api.ref.ReleaseSession
        .asFunction<void Function(ffi.Pointer<OrtSession>)>()(_session);
    _api.ref.ReleaseEnv.asFunction<void Function(ffi.Pointer<OrtEnv>)>()(_env);
    for (int i = 0; i < inputNames.length; i++) {
      malloc.free(_inputNames[i]);
    }
    calloc.free(_inputNames);
    malloc.free(_outputNames[0]);
    calloc.free(_outputNames);
  }

  /// Confirms the loaded graph has the inputs and output this code names,
  /// so a mismatched model is reported here instead of as an opaque error
  /// from `Run`.
  static void _verifyGraphIo(
    ffi.Pointer<OrtApi> api,
    ffi.Pointer<OrtSession> session,
  ) {
    final ffi.Pointer<OrtAllocator> allocator = _defaultAllocator(api);
    final List<String> actualInputs = _graphNames(
      api,
      session,
      allocator,
      isInput: true,
    );
    final List<String> actualOutputs = _graphNames(
      api,
      session,
      allocator,
      isInput: false,
    );
    if (!_startsWith(actualInputs, inputNames) ||
        !actualOutputs.contains(outputName)) {
      throw OrtSessionException(
        'This does not look like the Japanese punctuation model: expected '
        'inputs $inputNames and an output named "$outputName", but the '
        'graph declares inputs $actualInputs and outputs $actualOutputs.',
      );
    }
  }

  static bool _startsWith(List<String> actual, List<String> expected) {
    if (actual.length < expected.length) return false;
    for (int i = 0; i < expected.length; i++) {
      if (actual[i] != expected[i]) return false;
    }
    return true;
  }

  static ffi.Pointer<OrtAllocator> _defaultAllocator(ffi.Pointer<OrtApi> api) {
    final ffi.Pointer<ffi.Pointer<OrtAllocator>> out =
        calloc<ffi.Pointer<OrtAllocator>>();
    try {
      _check(
        api,
        api.ref.GetAllocatorWithDefaultOptions
            .asFunction<
              OrtStatusPtr Function(ffi.Pointer<ffi.Pointer<OrtAllocator>>)
            >()(out),
        'GetAllocatorWithDefaultOptions',
      );
      // Process-owned; must not be released.
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  static List<String> _graphNames(
    ffi.Pointer<OrtApi> api,
    ffi.Pointer<OrtSession> session,
    ffi.Pointer<OrtAllocator> allocator, {
    required bool isInput,
  }) {
    final ffi.Pointer<ffi.Size> count = calloc<ffi.Size>();
    final ffi.Pointer<ffi.Pointer<ffi.Char>> nameOut =
        calloc<ffi.Pointer<ffi.Char>>();
    final void Function(ffi.Pointer<OrtAllocator>, ffi.Pointer<ffi.Void>) free =
        allocator.ref.Free
            .asFunction<
              void Function(ffi.Pointer<OrtAllocator>, ffi.Pointer<ffi.Void>)
            >();
    try {
      _check(
        api,
        (isInput ? api.ref.SessionGetInputCount : api.ref.SessionGetOutputCount)
            .asFunction<
              OrtStatusPtr Function(
                ffi.Pointer<OrtSession>,
                ffi.Pointer<ffi.Size>,
              )
            >()(session, count),
        isInput ? 'SessionGetInputCount' : 'SessionGetOutputCount',
      );
      final List<String> names = <String>[];
      for (int i = 0; i < count.value; i++) {
        _check(
          api,
          (isInput ? api.ref.SessionGetInputName : api.ref.SessionGetOutputName)
              .asFunction<
                OrtStatusPtr Function(
                  ffi.Pointer<OrtSession>,
                  int,
                  ffi.Pointer<OrtAllocator>,
                  ffi.Pointer<ffi.Pointer<ffi.Char>>,
                )
              >()(session, i, allocator, nameOut),
          isInput ? 'SessionGetInputName' : 'SessionGetOutputName',
        );
        // The name was allocated by ORT's allocator, so it goes back
        // through the same allocator's Free, not Dart's.
        names.add(nameOut.value.cast<Utf8>().toDartString());
        free(allocator, nameOut.value.cast<ffi.Void>());
        nameOut.value = ffi.nullptr;
      }
      return names;
    } finally {
      calloc.free(count);
      calloc.free(nameOut);
    }
  }

  /// Turns a non-null `OrtStatus*` into an exception, releasing the status.
  static void _check(
    ffi.Pointer<OrtApi> api,
    ffi.Pointer<OrtStatus> status,
    String call,
  ) {
    if (status == ffi.nullptr) return;
    final int code = api.ref.GetErrorCode
        .asFunction<int Function(ffi.Pointer<OrtStatus>)>()(status);
    final ffi.Pointer<ffi.Char> message = api.ref.GetErrorMessage
        .asFunction<ffi.Pointer<ffi.Char> Function(ffi.Pointer<OrtStatus>)>()(
          status,
        );
    final String text = message == ffi.nullptr
        ? '(no message)'
        : message.cast<Utf8>().toDartString();
    api.ref.ReleaseStatus
        .asFunction<void Function(ffi.Pointer<OrtStatus>)>()(status);
    throw OrtSessionException('ONNX Runtime $call failed (code $code): $text');
  }
}
