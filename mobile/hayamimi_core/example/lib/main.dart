// Minimal embedding example for hayamimi_core: one dependency
// (pubspec.yaml) + this file wires up a live, routed subtitle widget using
// only hayamimi_core's public API -- no code copied from the mobile/ demo
// app. See this directory's README for the embedding steps.
import 'dart:io';

import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/material.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() {
  sherpa_onnx.initBindings();
  runApp(const SubtitleDemoApp());
}

class SubtitleDemoApp extends StatelessWidget {
  const SubtitleDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'hayamimi_core example',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
      home: const SubtitlePage(),
    );
  }
}

/// The subtitle widget itself: starts [HayamimiLive] with multilingual
/// routing and draft (in-progress) subtitles on, and renders its [events]
/// stream as a draft line + a scrolling list of finalized, language-tagged
/// lines.
class SubtitlePage extends StatefulWidget {
  const SubtitlePage({super.key});

  @override
  State<SubtitlePage> createState() => _SubtitlePageState();
}

class _SubtitlePageState extends State<SubtitlePage> {
  // The entire embedding surface: one HayamimiLive instance, fed model
  // paths on start() and read back through its events stream.
  final _live = HayamimiLive();

  String _draftText = '';
  final _finals = <FinalSubtitleEvent>[];
  bool _isRunning = false;
  String? _error;

  // Model download state -- see _downloadModels below. Not started
  // automatically: ModelProfile.jaSenseVoice is ~396 MB, so this only
  // happens when the user taps the button.
  bool _downloading = false;
  ModelDownloadEvent? _downloadEvent;
  String? _downloadError;

  @override
  void initState() {
    super.initState();
    _live.events.listen((event) {
      setState(() {
        switch (event) {
          case PartialSubtitleEvent(:final text):
            _draftText = text;
          case FinalSubtitleEvent():
            _draftText = '';
            _finals.add(event);
          default: // RefineSubtitleEvent/TranslationSubtitleEvent/error
            break;
        }
      });
    });
  }

  @override
  void dispose() {
    _live.dispose();
    super.dispose();
  }

  Future<_ModelPaths> _resolveModelPaths() async {
    // Same on-device layout the mobile/ reference app uses: model files
    // aren't bundled with either app, so they must already be present under
    // the app's own documents directory (adb push / a download step you add
    // yourself) -- see this package's README "Model files" section.
    final docsDir = await getApplicationDocumentsDirectory();
    final sep = Platform.pathSeparator;
    return _ModelPaths(
      modelDir: '${docsDir.path}$sep' 'model',
      vadModelPath: '${docsDir.path}$sep' 'vad$sep' 'silero_vad.onnx',
      senseVoiceModelDir: '${docsDir.path}$sep' 'sense_voice',
      lidModelDir: '${docsDir.path}$sep' 'lid',
    );
  }

  /// Downloads every model file `RoutingProfile.jaSenseVoice` needs (see
  /// [_resolveModelPaths]) into this app's Documents directory, guarded
  /// behind a button tap so the ~396 MB transfer never starts on its own
  /// (see README "Model files"). Safe to tap again later -- `downloadProfile`
  /// re-verifies by checksum and only re-fetches what's missing/corrupt.
  Future<void> _downloadModels() async {
    setState(() {
      _downloading = true;
      _downloadError = null;
      _downloadEvent = null;
    });
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      await downloadProfile(
        ModelProfile.jaSenseVoice,
        docsDir.path,
        onProgress: (event) {
          if (mounted) setState(() => _downloadEvent = event);
        },
      );
    } catch (e) {
      setState(() => _downloadError = e.toString());
    } finally {
      if (mounted) setState(() => _downloading = false);
    }
  }

  Future<void> _toggle() async {
    if (_isRunning) {
      await _live.stop();
      setState(() => _isRunning = false);
      return;
    }
    setState(() => _error = null);
    try {
      final paths = await _resolveModelPaths();
      await _live.start(
        modelDir: paths.modelDir,
        vadModelPath: paths.vadModelPath,
        // Routes each segment to ReazonSpeech (ja) or SenseVoice
        // (en/zh/ko/yue) instead of a single fixed-language model.
        routingProfile: RoutingProfile.jaSenseVoice,
        senseVoiceModelDir: paths.senseVoiceModelDir,
        lidModelDir: paths.lidModelDir,
      );
      setState(() => _isRunning = true);
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('hayamimi_core example')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _ModelDownloadSection(
              downloading: _downloading,
              event: _downloadEvent,
              error: _downloadError,
              onDownload: _downloadModels,
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _live.isDebugStreaming ? null : _toggle,
              icon: Icon(_isRunning ? Icons.stop : Icons.mic),
              label: Text(_isRunning ? 'Stop' : 'Start listening'),
            ),
            if (_error != null)
              Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            const SizedBox(height: 12),
            // Draft ("in-progress") line: grows while the speaker is still
            // talking, replaced by the real final the moment the segment
            // closes -- see HayamimiLive.events' PartialSubtitleEvent.
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _draftText.isEmpty ? '…' : _draftText,
                style: const TextStyle(fontStyle: FontStyle.italic),
              ),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: ListView.builder(
                reverse: true,
                itemCount: _finals.length,
                itemBuilder: (context, i) {
                  final entry = _finals[_finals.length - 1 - i];
                  return ListTile(
                    leading: entry.lang.isEmpty ? null : _LangBadge(entry.lang),
                    title: SelectableText(entry.text),
                  );
                },
              ),
            ),
            if (kDebugMode) _DebugWavStreamSection(live: _live),
          ],
        ),
      ),
    );
  }
}

/// A "Download models" button + progress readout, standing in for
/// whatever onboarding UI a host app would build around
/// [downloadProfile] -- see this package's README "Model files" section
/// for the full snippet this widget is built from.
class _ModelDownloadSection extends StatelessWidget {
  const _ModelDownloadSection({
    required this.downloading,
    required this.event,
    required this.error,
    required this.onDownload,
  });

  final bool downloading;
  final ModelDownloadEvent? event;
  final String? error;
  final VoidCallback onDownload;

  String _label() {
    if (!downloading) return 'Download models (~396 MB)';
    final e = event;
    if (e == null) return 'Starting download…';
    switch (e.phase) {
      case ModelDownloadPhase.skipped:
        return 'Already downloaded (${e.sourceIndex + 1}/${e.sourceCount})';
      case ModelDownloadPhase.downloading:
        final total = e.totalBytes;
        final mb = (e.bytesReceived / 1e6).toStringAsFixed(1);
        final totalMb = total == null ? '?' : (total / 1e6).toStringAsFixed(1);
        return 'Downloading ${e.sourceIndex + 1}/${e.sourceCount}: '
            '$mb / $totalMb MB';
      case ModelDownloadPhase.verifyingDownload:
        return 'Verifying ${e.sourceIndex + 1}/${e.sourceCount}…';
      case ModelDownloadPhase.extracting:
        return 'Extracting ${e.sourceIndex + 1}/${e.sourceCount}…';
      case ModelDownloadPhase.done:
        return 'Done ${e.sourceIndex + 1}/${e.sourceCount}';
    }
  }

  @override
  Widget build(BuildContext context) {
    final progress = event?.totalBytes == null
        ? null
        : event!.bytesReceived / event!.totalBytes!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        OutlinedButton(
          onPressed: downloading ? null : onDownload,
          child: Text(_label()),
        ),
        if (downloading &&
            event?.phase == ModelDownloadPhase.downloading)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: LinearProgressIndicator(value: progress),
          ),
        if (error != null)
          Text(error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
      ],
    );
  }
}

class _LangBadge extends StatelessWidget {
  const _LangBadge(this.lang);
  final String lang;

  @override
  Widget build(BuildContext context) {
    return CircleAvatar(
      radius: 14,
      child: Text(lang.toUpperCase(), style: const TextStyle(fontSize: 10)),
    );
  }
}

/// Debug-only (kDebugMode) verification aid: an emulator has no usable
/// microphone, so this streams a pushed test wav through the exact same
/// pipeline `_toggle`'s `HayamimiLive.start` uses, via
/// [HayamimiLive.startDebugWavStream] -- not part of the "real" embedding
/// surface a host app needs (see README).
class _DebugWavStreamSection extends StatefulWidget {
  const _DebugWavStreamSection({required this.live});
  final HayamimiLive live;

  @override
  State<_DebugWavStreamSection> createState() => _DebugWavStreamSectionState();
}

class _DebugWavStreamSectionState extends State<_DebugWavStreamSection> {
  String? _error;

  Future<void> _stream(String filename) async {
    setState(() => _error = null);
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      final sep = Platform.pathSeparator;
      await widget.live.startDebugWavStream(
        modelDir: '${docsDir.path}$sep' 'model',
        vadModelPath: '${docsDir.path}$sep' 'vad$sep' 'silero_vad.onnx',
        wavPath: '${docsDir.path}$sep$filename',
        routingProfile: RoutingProfile.jaSenseVoice,
        senseVoiceModelDir: '${docsDir.path}$sep' 'sense_voice',
        lidModelDir: '${docsDir.path}$sep' 'lid',
      );
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    final streaming = widget.live.isDebugStreaming;
    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Wrap(
        spacing: 8,
        children: [
          OutlinedButton(
            onPressed: streaming ? null : () => _stream('ja_test.wav'),
            child: const Text('Debug: stream ja_test.wav'),
          ),
          OutlinedButton(
            onPressed: streaming ? null : () => _stream('en_test.wav'),
            child: const Text('Debug: stream en_test.wav'),
          ),
          if (_error != null) Text(_error!),
        ],
      ),
    );
  }
}

class _ModelPaths {
  const _ModelPaths({
    required this.modelDir,
    required this.vadModelPath,
    required this.senseVoiceModelDir,
    required this.lidModelDir,
  });

  final String modelDir;
  final String vadModelPath;
  final String senseVoiceModelDir;
  final String lidModelDir;
}
