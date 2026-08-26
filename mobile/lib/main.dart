import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import 'bench/bench_result.dart';
import 'bench/bench_runner.dart';
import 'bench/model_kind.dart';

void main() {
  sherpa_onnx.initBindings();
  runApp(const HayamimiMobileApp());
}

class HayamimiMobileApp extends StatelessWidget {
  const HayamimiMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'hayamimi RTF bench',
      theme: ThemeData(primarySwatch: Colors.indigo, useMaterial3: true),
      home: const BenchPage(),
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
    setState(() {
      _modelDirController.text = '${docsDir.path}${Platform.pathSeparator}model';
      _wavPathController.text = '${docsDir.path}${Platform.pathSeparator}test.wav';
    });
  }

  @override
  void dispose() {
    _modelDirController.dispose();
    _wavPathController.dispose();
    super.dispose();
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
    return Scaffold(
      appBar: AppBar(title: const Text('hayamimi RTF bench')),
      body: SingleChildScrollView(
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
          ],
        ),
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
            Text('Audio duration: ${result.audioDurationSeconds.toStringAsFixed(2)} s'),
            Text('Processing time: ${result.processingDurationSeconds.toStringAsFixed(2)} s'),
            const SizedBox(height: 12),
            const Text('Output text:', style: TextStyle(fontWeight: FontWeight.bold)),
            SelectableText(result.text.isEmpty ? '(empty)' : result.text),
          ],
        ),
      ),
    );
  }
}
