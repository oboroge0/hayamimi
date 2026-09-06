import 'dart:io';

import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/material.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import 'live/live_page.dart';
import 'remote/remote_page.dart';

void main() {
  sherpa_onnx.initBindings();
  runApp(const HayamimiMobileApp());
}

class HayamimiMobileApp extends StatelessWidget {
  const HayamimiMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'hayamimi',
      theme: ThemeData(primarySwatch: Colors.indigo, useMaterial3: true),
      home: const HomeTabs(),
    );
  }
}

/// Top-level tab switcher between the RTF bench screen and the live mic
/// transcription screen.
class HomeTabs extends StatelessWidget {
  const HomeTabs({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('hayamimi'),
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Bench', icon: Icon(Icons.speed)),
              Tab(text: 'Live', icon: Icon(Icons.mic)),
              Tab(text: 'Remote', icon: Icon(Icons.cast)),
            ],
          ),
        ),
        body: const TabBarView(
          children: [BenchPage(), LivePage(), RemotePage()],
        ),
      ),
    );
  }
}

class BenchPage extends StatefulWidget {
  const BenchPage({super.key});

  @override
  State<BenchPage> createState() => _BenchPageState();
}

class _BenchPageState extends State<BenchPage> {
  final _modelDirController = TextEditingController();
  final _wavPathController = TextEditingController();

  // Manifest batch-eval controls (kDebugMode only — see _ManifestEvalSection).
  final _manifestModelDirController = TextEditingController();
  final _manifestPathController = TextEditingController();
  final _manifestWavDirController = TextEditingController();
  final _manifestOutputController = TextEditingController();
  bool _manifestRunning = false;
  String? _manifestError;
  String? _manifestSummary;

  // Routed (multilingual) manifest batch-eval controls (kDebugMode only):
  // same idea as the manifest eval above, but through RoutingProfile
  // .jaSenseVoice (ja + SenseVoice en/zh/ko/yue) instead of one recognizer,
  // so a mixed-language manifest's routing accuracy can be measured. See
  // docs/design/mobile_quantization.md "Multi-language routing on mobile".
  final _routedReazonDirController = TextEditingController();
  final _routedSenseVoiceDirController = TextEditingController();
  final _routedLidDirController = TextEditingController();
  final _routedManifestPathController = TextEditingController();
  final _routedWavDirController = TextEditingController();
  final _routedOutputController = TextEditingController();
  bool _routedRunning = false;
  String? _routedError;
  String? _routedSummary;

  ModelKind _modelKind = ModelKind.zipformerTransducer;
  bool _isRunning = false;
  String? _errorText;
  BenchResult? _result;

  @override
  void initState() {
    super.initState();
    _prefillDefaultPaths();
  }

  Future<void> _prefillDefaultPaths() async {
    // Convenience defaults: the app's own documents directory, so a user
    // (or `adb push`) can drop `model/` and `test.wav` there without typing
    // a full path by hand. Any path is still editable.
    final docsDir = await getApplicationDocumentsDirectory();
    if (!mounted) return;
    final sep = Platform.pathSeparator;
    setState(() {
      _modelDirController.text = '${docsDir.path}$sep' 'model';
      _wavPathController.text = '${docsDir.path}$sep' 'test.wav';
      _manifestModelDirController.text = '${docsDir.path}$sep' 'model';
      _manifestPathController.text =
          '${docsDir.path}${sep}eval_real${sep}manifest.json';
      _manifestWavDirController.text = '${docsDir.path}$sep' 'eval_real';
      _manifestOutputController.text =
          '${docsDir.path}${sep}manifest_eval_result.json';
      _routedReazonDirController.text = '${docsDir.path}$sep' 'model';
      _routedSenseVoiceDirController.text =
          '${docsDir.path}$sep' 'sense_voice';
      _routedLidDirController.text = '${docsDir.path}$sep' 'lid';
      _routedManifestPathController.text =
          '${docsDir.path}${sep}eval_real_zhko${sep}manifest.json';
      _routedWavDirController.text = '${docsDir.path}$sep' 'eval_real_zhko';
      _routedOutputController.text =
          '${docsDir.path}${sep}routed_manifest_eval_result.json';
    });
  }

  @override
  void dispose() {
    _modelDirController.dispose();
    _wavPathController.dispose();
    _manifestModelDirController.dispose();
    _manifestPathController.dispose();
    _manifestWavDirController.dispose();
    _manifestOutputController.dispose();
    _routedReazonDirController.dispose();
    _routedSenseVoiceDirController.dispose();
    _routedLidDirController.dispose();
    _routedManifestPathController.dispose();
    _routedWavDirController.dispose();
    _routedOutputController.dispose();
    super.dispose();
  }

  Future<void> _runManifestEval() async {
    setState(() {
      _manifestRunning = true;
      _manifestError = null;
      _manifestSummary = null;
    });

    try {
      final results = await ManifestEvalRunner.run(
        modelDir: _manifestModelDirController.text.trim(),
        manifestPath: _manifestPathController.text.trim(),
        wavDir: _manifestWavDirController.text.trim(),
      );

      final outputPath = _manifestOutputController.text.trim();
      await File(outputPath).writeAsString(ManifestEvalRunner.toJson(results));

      final avgRtf = results.isEmpty
          ? 0.0
          : results.map((r) => r.rtf).reduce((a, b) => a + b) /
                results.length;
      setState(() {
        _manifestSummary =
            'Decoded ${results.length} clips. '
            'Mean RTF (this device, informational only): '
            '${avgRtf.toStringAsFixed(3)}.\n'
            'Wrote results to: $outputPath\n'
            '(pull with: adb pull $outputPath, then score with '
            'scripts/eval_accuracy.py\'s cer_ja)';
      });
    } catch (e) {
      setState(() => _manifestError = e.toString());
    } finally {
      setState(() => _manifestRunning = false);
    }
  }

  Future<void> _runRoutedManifestEval() async {
    setState(() {
      _routedRunning = true;
      _routedError = null;
      _routedSummary = null;
    });

    try {
      final results = await ManifestEvalRunner.runRouted(
        reazonModelDir: _routedReazonDirController.text.trim(),
        senseVoiceModelDir: _routedSenseVoiceDirController.text.trim(),
        lidModelDir: _routedLidDirController.text.trim(),
        manifestPath: _routedManifestPathController.text.trim(),
        wavDir: _routedWavDirController.text.trim(),
      );

      final outputPath = _routedOutputController.text.trim();
      await File(outputPath).writeAsString(ManifestEvalRunner.toJson(results));

      final langCorrect = results
          .where((r) => r.detectedLang == r.lang)
          .length;
      final langAccuracy = results.isEmpty
          ? 0.0
          : langCorrect / results.length * 100;
      setState(() {
        _routedSummary =
            'Decoded ${results.length} clips through ja+SenseVoice '
            'routing.\n'
            'Language routing accuracy: $langCorrect/${results.length} '
            '(${langAccuracy.toStringAsFixed(1)}%)\n'
            'Wrote results to: $outputPath\n'
            '(pull with: adb pull $outputPath, then score CER per-language '
            'with scripts/eval_accuracy.py)';
      });
    } catch (e) {
      setState(() => _routedError = e.toString());
    } finally {
      setState(() => _routedRunning = false);
    }
  }

  Future<void> _runBench() async {
    setState(() {
      _isRunning = true;
      _errorText = null;
      _result = null;
    });

    try {
      final result = await BenchRunner.run(
        modelKind: _modelKind,
        modelDir: _modelDirController.text.trim(),
        wavPath: _wavPathController.text.trim(),
      );
      setState(() => _result = result);
    } catch (e) {
      setState(() => _errorText = e.toString());
    } finally {
      setState(() => _isRunning = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          DropdownButtonFormField<ModelKind>(
            initialValue: _modelKind,
            decoration: const InputDecoration(
              labelText: 'Model type',
              border: OutlineInputBorder(),
            ),
            items: [
              for (final kind in ModelKind.values)
                DropdownMenuItem(value: kind, child: Text(kind.label)),
            ],
            onChanged: (kind) => setState(() => _modelKind = kind!),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _modelDirController,
            decoration: const InputDecoration(
              labelText: 'Model directory (encoder/decoder/joiner/tokens.txt)',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _wavPathController,
            decoration: const InputDecoration(
              labelText: 'WAV file path (16kHz mono)',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          FilledButton(
            onPressed: _isRunning ? null : _runBench,
            child: _isRunning
                ? const SizedBox(
                    height: 20,
                    width: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Text('Run'),
          ),
          const SizedBox(height: 24),
          if (_errorText != null)
            Text(
              _errorText!,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          if (_result != null) _ResultCard(result: _result!),
          if (kDebugMode) ...[
            const SizedBox(height: 32),
            const Divider(),
            const SizedBox(height: 8),
            Text(
              'Manifest batch eval (debug only)',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const Text(
              'Decodes every clip in a manifest.json (see '
              'testdata/eval_real) through one recognizer and writes a '
              'JSON results file for offline CER scoring on the PC. Speed '
              'numbers here are informational only — not comparable across '
              'devices.',
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _manifestModelDirController,
              decoration: const InputDecoration(
                labelText: 'Model directory',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _manifestPathController,
              decoration: const InputDecoration(
                labelText: 'manifest.json path',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _manifestWavDirController,
              decoration: const InputDecoration(
                labelText: 'WAV directory (manifest "wav" fields join here)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _manifestOutputController,
              decoration: const InputDecoration(
                labelText: 'Output JSON path',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _manifestRunning ? null : _runManifestEval,
              child: _manifestRunning
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Run manifest eval'),
            ),
            const SizedBox(height: 16),
            if (_manifestError != null)
              Text(
                _manifestError!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            if (_manifestSummary != null) Text(_manifestSummary!),
            const SizedBox(height: 32),
            const Divider(),
            const SizedBox(height: 8),
            Text(
              'Routed multilingual manifest eval (debug only)',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const Text(
              'Decodes every clip in a manifest.json through ja+SenseVoice '
              'routing (RoutingProfile.jaSenseVoice): each clip is routed '
              'independently to ReazonSpeech (ja) or SenseVoice '
              '(en/zh/ko/yue) by the same dual-LID policy the live pipeline '
              'uses. Reports language-routing accuracy against the '
              'manifest\'s ground-truth "lang" field; writes a JSON results '
              'file (with a per-clip "detected_lang") for CER scoring on '
              'the PC.',
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _routedReazonDirController,
              decoration: const InputDecoration(
                labelText: 'ja model directory (ReazonSpeech)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _routedSenseVoiceDirController,
              decoration: const InputDecoration(
                labelText: 'SenseVoice model directory',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _routedLidDirController,
              decoration: const InputDecoration(
                labelText: 'whisper-tiny LID model directory',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _routedManifestPathController,
              decoration: const InputDecoration(
                labelText: 'manifest.json path (multilingual)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _routedWavDirController,
              decoration: const InputDecoration(
                labelText: 'WAV directory',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _routedOutputController,
              decoration: const InputDecoration(
                labelText: 'Output JSON path',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _routedRunning ? null : _runRoutedManifestEval,
              child: _routedRunning
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Run routed manifest eval'),
            ),
            const SizedBox(height: 16),
            if (_routedError != null)
              Text(
                _routedError!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            if (_routedSummary != null) Text(_routedSummary!),
          ],
        ],
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});

  final BenchResult result;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'RTF: ${result.rtf.toStringAsFixed(3)}',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              'Audio duration: ${result.audioDurationSeconds.toStringAsFixed(2)} s',
            ),
            Text(
              'Processing time: ${result.processingDurationSeconds.toStringAsFixed(2)} s',
            ),
            const SizedBox(height: 12),
            const Text(
              'Output text:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            SelectableText(result.text.isEmpty ? '(empty)' : result.text),
          ],
        ),
      ),
    );
  }
}
