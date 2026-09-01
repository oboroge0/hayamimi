// ignore_for_file: non_constant_identifier_names, camel_case_types
//
// ONNX Runtime C API bindings, trimmed to what the Japanese punctuation
// restorer uses.
//
// Derived from the ffigen-generated bindings in
// https://github.com/gtbluesky/onnxruntime_flutter (MIT, Copyright (c) 2023
// gtbluesky), file `lib/src/bindings/onnxruntime_bindings_generated.dart`,
// which were generated from ONNX Runtime's `onnxruntime_c_api.h` (MIT,
// Copyright (c) Microsoft Corporation). See THIRD_PARTY_NOTICES.md.
//
// Two things were changed against the upstream file, both deliberate:
//
//  1. Only the ~26 `OrtApi` members this package calls keep their real
//     signature. Every other member is declared as a plain
//     `Pointer<Void>` placeholder under its real name. `OrtApi` is a
//     struct of function pointers, so every member is one machine word
//     wide either way and the members that matter still land on the right
//     offsets — the placeholders exist purely to hold their slots. Keeping
//     the real names makes the slot list auditable line-by-line against
//     the ORT header instead of being a run of anonymous padding.
//
//     `OrtApi` is append-only by ONNX Runtime's own ABI promise (new
//     members are only ever added at the end), which is why a table
//     declared from an older header stays correct against a newer runtime,
//     and why truncating it after the last member used is safe: this code
//     only ever reads members through a pointer the runtime handed back,
//     it never allocates an `OrtApi` itself.
//
//  2. `CreateSession`'s `model_path` is `ORTCHAR_T*`, which is `wchar_t*`
//     on Windows and `char*` everywhere else. The upstream bindings were
//     generated on a non-Windows host, so they declare it `Pointer<Char>`,
//     which silently produces garbage paths on Windows. It is declared
//     `Pointer<Void>` here so the caller has to choose the encoding —
//     see `PunctOrtSession`.

import 'dart:ffi' as ffi;

// --- opaque handles -------------------------------------------------------

final class OrtEnv extends ffi.Opaque {}

final class OrtStatus extends ffi.Opaque {}

final class OrtMemoryInfo extends ffi.Opaque {}

final class OrtSession extends ffi.Opaque {}

final class OrtValue extends ffi.Opaque {}

final class OrtRunOptions extends ffi.Opaque {}

final class OrtTensorTypeAndShapeInfo extends ffi.Opaque {}

final class OrtSessionOptions extends ffi.Opaque {}

/// Non-null when an ORT call failed; the caller owns it and must release it.
typedef OrtStatusPtr = ffi.Pointer<OrtStatus>;

// --- enum values used here ------------------------------------------------

/// `OrtLoggingLevel`: only warnings and worse reach stderr.
const int ortLoggingLevelWarning = 2;

/// `ONNXTensorElementDataType`: the two element types this model uses.
const int onnxTensorElementDataTypeFloat = 1;
const int onnxTensorElementDataTypeInt64 = 7;

/// `OrtAllocatorType.OrtArenaAllocator` and `OrtMemType.OrtMemTypeDefault`.
const int ortArenaAllocator = 1;
const int ortMemTypeDefault = 0;

// --- structs --------------------------------------------------------------

/// The allocator ORT hands out for strings it allocates on the caller's
/// behalf (input/output names). `Free` is the only member used here.
final class OrtAllocator extends ffi.Struct {
  @ffi.Uint32()
  external int version;

  external ffi.Pointer<
    ffi.NativeFunction<
      ffi.Pointer<ffi.Void> Function(
        ffi.Pointer<OrtAllocator> this_,
        ffi.Size size,
      )
    >
  >
  Alloc;

  external ffi.Pointer<
    ffi.NativeFunction<
      ffi.Void Function(ffi.Pointer<OrtAllocator> this_, ffi.Pointer<ffi.Void> p)
    >
  >
  Free;

  external ffi.Pointer<
    ffi.NativeFunction<
      ffi.Pointer<OrtMemoryInfo> Function(ffi.Pointer<OrtAllocator> this_)
    >
  >
  Info;
}

/// The entry point `OrtGetApiBase()` returns: a version-negotiating handle,
/// stable across every ONNX Runtime release.
final class OrtApiBase extends ffi.Struct {
  external ffi.Pointer<
    ffi.NativeFunction<ffi.Pointer<OrtApi> Function(ffi.Uint32 version)>
  >
  GetApi;

  external ffi.Pointer<ffi.NativeFunction<ffi.Pointer<ffi.Char> Function()>>
  GetVersionString;
}

/// The ONNX Runtime C API function table. See the file header for what was
/// trimmed and why the trim is safe.
final class OrtApi extends ffi.Struct {
  // 0: unused slot, kept for offset alignment.
  external ffi.Pointer<ffi.Void> CreateStatus;

  external ffi.Pointer<
    ffi.NativeFunction<ffi.Int32 Function(ffi.Pointer<OrtStatus> status)>
  >
  GetErrorCode;

  external ffi.Pointer<
    ffi.NativeFunction<
      ffi.Pointer<ffi.Char> Function(ffi.Pointer<OrtStatus> status)
    >
  >
  GetErrorMessage;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Int32 log_severity_level,
        ffi.Pointer<ffi.Char> logid,
        ffi.Pointer<ffi.Pointer<OrtEnv>> out,
      )
    >
  >
  CreateEnv;

  // 4-6: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> CreateEnvWithCustomLogger;
  external ffi.Pointer<ffi.Void> EnableTelemetryEvents;
  external ffi.Pointer<ffi.Void> DisableTelemetryEvents;

  /// `model_path` is `ORTCHAR_T*`: UTF-16 on Windows, UTF-8 elsewhere.
  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtEnv> env,
        ffi.Pointer<ffi.Void> model_path,
        ffi.Pointer<OrtSessionOptions> options,
        ffi.Pointer<ffi.Pointer<OrtSession>> out,
      )
    >
  >
  CreateSession;

  // 8: unused slot, kept for offset alignment.
  external ffi.Pointer<ffi.Void> CreateSessionFromArray;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtSession> session,
        ffi.Pointer<OrtRunOptions> run_options,
        ffi.Pointer<ffi.Pointer<ffi.Char>> input_names,
        ffi.Pointer<ffi.Pointer<OrtValue>> inputs,
        ffi.Size input_len,
        ffi.Pointer<ffi.Pointer<ffi.Char>> output_names,
        ffi.Size output_names_len,
        ffi.Pointer<ffi.Pointer<OrtValue>> outputs,
      )
    >
  >
  Run;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(ffi.Pointer<ffi.Pointer<OrtSessionOptions>> options)
    >
  >
  CreateSessionOptions;

  // 11-23: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> SetOptimizedModelFilePath;
  external ffi.Pointer<ffi.Void> CloneSessionOptions;
  external ffi.Pointer<ffi.Void> SetSessionExecutionMode;
  external ffi.Pointer<ffi.Void> EnableProfiling;
  external ffi.Pointer<ffi.Void> DisableProfiling;
  external ffi.Pointer<ffi.Void> EnableMemPattern;
  external ffi.Pointer<ffi.Void> DisableMemPattern;
  external ffi.Pointer<ffi.Void> EnableCpuMemArena;
  external ffi.Pointer<ffi.Void> DisableCpuMemArena;
  external ffi.Pointer<ffi.Void> SetSessionLogId;
  external ffi.Pointer<ffi.Void> SetSessionLogVerbosityLevel;
  external ffi.Pointer<ffi.Void> SetSessionLogSeverityLevel;
  external ffi.Pointer<ffi.Void> SetSessionGraphOptimizationLevel;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtSessionOptions> options,
        ffi.Int intra_op_num_threads,
      )
    >
  >
  SetIntraOpNumThreads;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtSessionOptions> options,
        ffi.Int inter_op_num_threads,
      )
    >
  >
  SetInterOpNumThreads;

  // 26-29: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> CreateCustomOpDomain;
  external ffi.Pointer<ffi.Void> CustomOpDomain_Add;
  external ffi.Pointer<ffi.Void> AddCustomOpDomain;
  external ffi.Pointer<ffi.Void> RegisterCustomOpsLibrary;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtSession> session,
        ffi.Pointer<ffi.Size> out,
      )
    >
  >
  SessionGetInputCount;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtSession> session,
        ffi.Pointer<ffi.Size> out,
      )
    >
  >
  SessionGetOutputCount;

  // 32-35: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> SessionGetOverridableInitializerCount;
  external ffi.Pointer<ffi.Void> SessionGetInputTypeInfo;
  external ffi.Pointer<ffi.Void> SessionGetOutputTypeInfo;
  external ffi.Pointer<ffi.Void> SessionGetOverridableInitializerTypeInfo;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtSession> session,
        ffi.Size index,
        ffi.Pointer<OrtAllocator> allocator,
        ffi.Pointer<ffi.Pointer<ffi.Char>> value,
      )
    >
  >
  SessionGetInputName;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtSession> session,
        ffi.Size index,
        ffi.Pointer<OrtAllocator> allocator,
        ffi.Pointer<ffi.Pointer<ffi.Char>> value,
      )
    >
  >
  SessionGetOutputName;

  // 38-48: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> SessionGetOverridableInitializerName;
  external ffi.Pointer<ffi.Void> CreateRunOptions;
  external ffi.Pointer<ffi.Void> RunOptionsSetRunLogVerbosityLevel;
  external ffi.Pointer<ffi.Void> RunOptionsSetRunLogSeverityLevel;
  external ffi.Pointer<ffi.Void> RunOptionsSetRunTag;
  external ffi.Pointer<ffi.Void> RunOptionsGetRunLogVerbosityLevel;
  external ffi.Pointer<ffi.Void> RunOptionsGetRunLogSeverityLevel;
  external ffi.Pointer<ffi.Void> RunOptionsGetRunTag;
  external ffi.Pointer<ffi.Void> RunOptionsSetTerminate;
  external ffi.Pointer<ffi.Void> RunOptionsUnsetTerminate;
  external ffi.Pointer<ffi.Void> CreateTensorAsOrtValue;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtMemoryInfo> info,
        ffi.Pointer<ffi.Void> p_data,
        ffi.Size p_data_len,
        ffi.Pointer<ffi.Int64> shape,
        ffi.Size shape_len,
        ffi.Int32 type,
        ffi.Pointer<ffi.Pointer<OrtValue>> out,
      )
    >
  >
  CreateTensorWithDataAsOrtValue;

  // 50: unused slot, kept for offset alignment.
  external ffi.Pointer<ffi.Void> IsTensor;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtValue> value,
        ffi.Pointer<ffi.Pointer<ffi.Void>> out,
      )
    >
  >
  GetTensorMutableData;

  // 52-59: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> FillStringTensor;
  external ffi.Pointer<ffi.Void> GetStringTensorDataLength;
  external ffi.Pointer<ffi.Void> GetStringTensorContent;
  external ffi.Pointer<ffi.Void> CastTypeInfoToTensorInfo;
  external ffi.Pointer<ffi.Void> GetOnnxTypeFromTypeInfo;
  external ffi.Pointer<ffi.Void> CreateTensorTypeAndShapeInfo;
  external ffi.Pointer<ffi.Void> SetTensorElementType;
  external ffi.Pointer<ffi.Void> SetDimensions;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtTensorTypeAndShapeInfo> info,
        ffi.Pointer<ffi.Int32> out,
      )
    >
  >
  GetTensorElementType;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtTensorTypeAndShapeInfo> info,
        ffi.Pointer<ffi.Size> out,
      )
    >
  >
  GetDimensionsCount;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtTensorTypeAndShapeInfo> info,
        ffi.Pointer<ffi.Int64> dim_values,
        ffi.Size dim_values_length,
      )
    >
  >
  GetDimensions;

  // 63-64: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> GetSymbolicDimensions;
  external ffi.Pointer<ffi.Void> GetTensorShapeElementCount;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Pointer<OrtValue> value,
        ffi.Pointer<ffi.Pointer<OrtTensorTypeAndShapeInfo>> out,
      )
    >
  >
  GetTensorTypeAndShape;

  // 66-68: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> GetTypeInfo;
  external ffi.Pointer<ffi.Void> GetValueType;
  external ffi.Pointer<ffi.Void> CreateMemoryInfo;

  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(
        ffi.Int32 type,
        ffi.Int32 mem_type,
        ffi.Pointer<ffi.Pointer<OrtMemoryInfo>> out,
      )
    >
  >
  CreateCpuMemoryInfo;

  // 70-77: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> CompareMemoryInfo;
  external ffi.Pointer<ffi.Void> MemoryInfoGetName;
  external ffi.Pointer<ffi.Void> MemoryInfoGetId;
  external ffi.Pointer<ffi.Void> MemoryInfoGetMemType;
  external ffi.Pointer<ffi.Void> MemoryInfoGetType;
  external ffi.Pointer<ffi.Void> AllocatorAlloc;
  external ffi.Pointer<ffi.Void> AllocatorFree;
  external ffi.Pointer<ffi.Void> AllocatorGetInfo;

  /// The returned allocator is process-owned; it must not be released.
  external ffi.Pointer<
    ffi.NativeFunction<
      OrtStatusPtr Function(ffi.Pointer<ffi.Pointer<OrtAllocator>> out)
    >
  >
  GetAllocatorWithDefaultOptions;

  // 79-91: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> AddFreeDimensionOverride;
  external ffi.Pointer<ffi.Void> GetValue;
  external ffi.Pointer<ffi.Void> GetValueCount;
  external ffi.Pointer<ffi.Void> CreateValue;
  external ffi.Pointer<ffi.Void> CreateOpaqueValue;
  external ffi.Pointer<ffi.Void> GetOpaqueValue;
  external ffi.Pointer<ffi.Void> KernelInfoGetAttribute_float;
  external ffi.Pointer<ffi.Void> KernelInfoGetAttribute_int64;
  external ffi.Pointer<ffi.Void> KernelInfoGetAttribute_string;
  external ffi.Pointer<ffi.Void> KernelContext_GetInputCount;
  external ffi.Pointer<ffi.Void> KernelContext_GetOutputCount;
  external ffi.Pointer<ffi.Void> KernelContext_GetInput;
  external ffi.Pointer<ffi.Void> KernelContext_GetOutput;

  external ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<OrtEnv> input)>
  >
  ReleaseEnv;

  external ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<OrtStatus> input)>
  >
  ReleaseStatus;

  external ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<OrtMemoryInfo> input)>
  >
  ReleaseMemoryInfo;

  external ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<OrtSession> input)>
  >
  ReleaseSession;

  external ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<OrtValue> input)>
  >
  ReleaseValue;

  // 97-98: unused slots, kept for offset alignment.
  external ffi.Pointer<ffi.Void> ReleaseRunOptions;
  external ffi.Pointer<ffi.Void> ReleaseTypeInfo;

  external ffi.Pointer<
    ffi.NativeFunction<
      ffi.Void Function(ffi.Pointer<OrtTensorTypeAndShapeInfo> input)
    >
  >
  ReleaseTensorTypeAndShapeInfo;

  // The last member this table declares. Everything ONNX Runtime added
  // after it is unused here, so it is left off; see the file header.
  external ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<OrtSessionOptions> input)>
  >
  ReleaseSessionOptions;
}

/// The C signature of the one symbol looked up by name.
typedef OrtGetApiBaseNative = ffi.Pointer<OrtApiBase> Function();
typedef OrtGetApiBaseDart = ffi.Pointer<OrtApiBase> Function();
