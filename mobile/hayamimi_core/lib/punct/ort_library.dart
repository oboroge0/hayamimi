import 'dart:ffi' as ffi;
import 'dart:io' show Platform;

import 'ort_bindings.dart';

/// Thrown when the ONNX Runtime shared library, or its entry point, cannot
/// be reached from Dart.
class OrtLibraryException implements Exception {
  OrtLibraryException(this.message);

  final String message;

  @override
  String toString() => 'OrtLibraryException: $message';
}

/// Finds the ONNX Runtime C API that is already loaded in this process.
///
/// The problem this solves: hayamimi's speech recognition runs through
/// `sherpa_onnx`, which bundles its own ONNX Runtime shared library. The
/// punctuation model needs ONNX Runtime too, but adding a second ONNX
/// Runtime package (`onnxruntime`, `flutter_onnxruntime`) would put a
/// second copy of `libonnxruntime.so` in the same app. Two copies of that
/// library in one process is a known crash on Android
/// (k2-fsa/sherpa-onnx#3261). So instead of shipping a runtime, this class
/// looks up ONNX Runtime's C entry point — `OrtGetApiBase` — in the
/// library sherpa_onnx has already loaded, and calls the C API directly
/// through `dart:ffi`.
///
/// `OrtGetApiBase()` returns a small handle whose `GetApi(version)` returns
/// the function table for the API version asked for. That table only ever
/// grows at the end across ONNX Runtime releases, so a table described from
/// an older header keeps working against a newer runtime — which is what
/// lets this package pin one low version and stay compatible as
/// `sherpa_onnx` bumps its bundled runtime.
class OrtLibrary {
  OrtLibrary._(this.library, this.api, this.versionString);

  /// The loaded shared library `OrtGetApiBase` was found in.
  final ffi.DynamicLibrary library;

  /// The ONNX Runtime C API function table.
  final ffi.Pointer<OrtApi> api;

  /// The runtime's own version string, e.g. `1.27.1`. Reported in errors
  /// so a version mismatch is visible rather than guessed at.
  final String versionString;

  /// Pass this as `libraryPath` to look the symbol up in the running
  /// process instead of opening a file. This is what an iOS build would
  /// need if `sherpa_onnx`'s xcframework re-exports `OrtGetApiBase` — see
  /// the class docs on [open] for why that is not the default there.
  static const String processSymbols = ':process:';

  /// The API version requested from the runtime.
  ///
  /// 11 is ONNX Runtime 1.10's API. Every function this package calls has
  /// existed since well before that, and asking for a low version is what
  /// keeps the binding valid against both older and newer runtimes.
  static const int apiVersion = 11;

  static final Map<String, OrtLibrary> _cache = <String, OrtLibrary>{};

  /// Opens ONNX Runtime and returns its C API table.
  ///
  /// [libraryPath] takes precedence when given: an absolute path to the
  /// shared library, or [processSymbols] to search the running process.
  /// Pass it on desktop and in tests, where nothing has put ONNX Runtime
  /// on the loader search path — for example the `onnxruntime.dll` that
  /// the `sherpa_onnx_windows` package ships.
  ///
  /// Without it, the default per platform is:
  ///
  ///  * **Android** — `libonnxruntime.so`, which resolves to the copy
  ///    `sherpa_onnx` already loaded into the app. This is the case the
  ///    design is for, but it has not been run on a device in this change.
  ///  * **iOS** — refused. `sherpa_onnx` links ONNX Runtime as a static
  ///    xcframework and it has not been confirmed that `OrtGetApiBase`
  ///    stays exported from the app binary, so claiming support would be a
  ///    guess. Pass [processSymbols] explicitly to try it.
  ///  * **Windows / macOS / Linux** — `onnxruntime.dll`,
  ///    `libonnxruntime.dylib`, `libonnxruntime.so` by plain name, i.e.
  ///    whatever the operating system's loader search path finds. Nothing
  ///    puts it there by default, so [libraryPath] is normally the right
  ///    answer on these platforms.
  ///
  /// Repeat calls with the same [libraryPath] return the same instance —
  /// the process only ever holds one ONNX Runtime.
  static OrtLibrary open({String? libraryPath}) {
    final String key = libraryPath ?? '';
    final OrtLibrary? cached = _cache[key];
    if (cached != null) return cached;
    final OrtLibrary opened = _open(libraryPath);
    _cache[key] = opened;
    return opened;
  }

  static OrtLibrary _open(String? libraryPath) {
    final ffi.DynamicLibrary library = _resolve(libraryPath);

    final OrtGetApiBaseDart getApiBase;
    try {
      getApiBase = library
          .lookupFunction<OrtGetApiBaseNative, OrtGetApiBaseDart>(
            'OrtGetApiBase',
          );
    } on ArgumentError catch (e) {
      throw OrtLibraryException(
        'The library loaded for ONNX Runtime does not export OrtGetApiBase '
        '($e). On mobile this symbol comes from the ONNX Runtime that '
        'sherpa_onnx bundles; if that build hides its symbols, this package '
        'cannot reach the C API and needs an explicit libraryPath.',
      );
    }

    final ffi.Pointer<OrtApiBase> base = getApiBase();
    if (base == ffi.nullptr) {
      throw OrtLibraryException('OrtGetApiBase() returned null.');
    }

    final ffi.Pointer<ffi.Char> versionPtr = base.ref.GetVersionString
        .asFunction<ffi.Pointer<ffi.Char> Function()>()();
    final String version = _readCString(versionPtr.cast<ffi.Uint8>());

    final ffi.Pointer<OrtApi> api = base.ref.GetApi
        .asFunction<ffi.Pointer<OrtApi> Function(int)>()(apiVersion);
    if (api == ffi.nullptr) {
      throw OrtLibraryException(
        'ONNX Runtime $version does not serve C API version $apiVersion. '
        'That should not happen — the API table is append-only — so treat '
        'this as an incompatible or unexpected runtime build.',
      );
    }

    return OrtLibrary._(library, api, version);
  }

  static ffi.DynamicLibrary _resolve(String? libraryPath) {
    if (libraryPath == processSymbols) {
      return ffi.DynamicLibrary.process();
    }
    if (libraryPath != null) {
      try {
        return ffi.DynamicLibrary.open(libraryPath);
      } on ArgumentError catch (e) {
        throw OrtLibraryException(
          'Could not load the ONNX Runtime library at "$libraryPath": $e',
        );
      }
    }
    if (Platform.isIOS) {
      throw OrtLibraryException(
        'Loading ONNX Runtime on iOS is unverified. sherpa_onnx links it as '
        'an xcframework and it has not been confirmed that OrtGetApiBase '
        'stays exported from the app binary, so this package refuses to '
        'guess. Pass libraryPath: OrtLibrary.processSymbols to try the '
        'running process, or an explicit path to a framework binary.',
      );
    }
    final String name = switch (Platform.operatingSystem) {
      'android' || 'linux' => 'libonnxruntime.so',
      'macos' => 'libonnxruntime.dylib',
      'windows' => 'onnxruntime.dll',
      final String other => throw OrtLibraryException(
        'No default ONNX Runtime library name for platform "$other"; pass '
        'libraryPath explicitly.',
      ),
    };
    try {
      return ffi.DynamicLibrary.open(name);
    } on ArgumentError catch (e) {
      throw OrtLibraryException(
        'Could not load "$name": $e. On Android this library comes from '
        'sherpa_onnx; on desktop nothing puts it on the loader search path, '
        'so pass libraryPath explicitly (for example the onnxruntime.dll '
        'inside the sherpa_onnx_windows package).',
      );
    }
  }

  static String _readCString(ffi.Pointer<ffi.Uint8> p) {
    if (p == ffi.nullptr) return '(unknown)';
    int length = 0;
    while (p[length] != 0) {
      length++;
    }
    return String.fromCharCodes(p.asTypedList(length));
  }
}
