import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../bench/model_kind.dart';
import 'live_transcript_entry.dart';
import 'live_transcriber.dart';

/// Live mic transcription screen: start/stop capture, see the model in use,
/// and watch finalized transcript lines accumulate as speech is detected.
class LivePage extends StatefulWidget {
  const LivePage({super.key});

  @override
  State<LivePage> createState() => _LivePageState();
}

class _LivePageState extends State<LivePage> {
  final _modelDirController = TextEditingController();
  final _vadModelPathController = TextEditingController();
  final _scrollController = ScrollController();

  final _transcriber = LiveTranscriber();
  StreamSubscription<LiveTranscriptEntry>? _entriesSubscription;
  StreamSubscription<bool>? _decodingSubscription;

  final ModelKind _modelKind = ModelKind.zipformerTransducer;
  final List<LiveTranscriptEntry> _entries = [];
  bool _isStarting = false;
  bool _isDecoding = false;
  String? _errorText;

  @override
  void initState() {
    super.initState();
    _prefillDefaultPaths();
    _entriesSubscription = _transcriber.entries.listen(_onEntry);
    _decodingSubscription = _transcriber.decoding.listen((decoding) {
      if (!mounted) return;
      setState(() => _isDecoding = decoding);
    });
  }

  Future<void> _prefillDefaultPaths() async {
    final docsDir = await getApplicationDocumentsDirectory();
    if (!mounted) return;
    final sep = Platform.pathSeparator;
    setState(() {
      _modelDirController.text =
          '${docsDir.path}$sep'
          'model';
      _vadModelPathController.text =
          '${docsDir.path}$sep'
          'vad$sep'
          'silero_vad.onnx';
    });
  }

  void _onEntry(LiveTranscriptEntry entry) {
    if (!mounted) return;
    setState(() => _entries.add(entry));
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _toggle() async {
    if (_transcriber.isRunning) {
      await _transcriber.stop();
      if (!mounted) return;
      setState(() {});
      return;
    }

    setState(() {
      _isStarting = true;
      _errorText = null;
    });
    try {
      await _transcriber.start(
        modelKind: _modelKind,
        modelDir: _modelDirController.text.trim(),
        vadModelPath: _vadModelPathController.text.trim(),
      );
    } catch (e) {
      setState(() => _errorText = e.toString());
    } finally {
      if (mounted) {
        setState(() => _isStarting = false);
      }
    }
  }

  @override
  void dispose() {
    _entriesSubscription?.cancel();
    _decodingSubscription?.cancel();
    _transcriber.dispose();
    _modelDirController.dispose();
    _vadModelPathController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isRunning = _transcriber.isRunning;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Model: ${_modelKind.label}',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _modelDirController,
            enabled: !isRunning,
            decoration: const InputDecoration(
              labelText: 'Model directory',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _vadModelPathController,
            enabled: !isRunning,
            decoration: const InputDecoration(
              labelText: 'VAD model path (silero_vad.onnx)',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _isStarting ? null : _toggle,
            icon: _isStarting
                ? const SizedBox(
                    height: 16,
                    width: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(isRunning ? Icons.stop : Icons.mic),
            label: Text(isRunning ? 'Stop' : 'Start listening'),
            style: FilledButton.styleFrom(
              backgroundColor: isRunning
                  ? Theme.of(context).colorScheme.error
                  : null,
            ),
          ),
          const SizedBox(height: 8),
          if (isRunning)
            Row(
              children: [
                Icon(Icons.circle, size: 10, color: Colors.red.shade400),
                const SizedBox(width: 6),
                Text(_isDecoding ? 'Decoding...' : 'Listening...'),
              ],
            ),
          if (_errorText != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                _errorText!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ),
          const SizedBox(height: 16),
          Expanded(
            child: _entries.isEmpty
                ? const Center(child: Text('Transcript will appear here.'))
                : ListView.separated(
                    controller: _scrollController,
                    itemCount: _entries.length,
                    separatorBuilder: (_, _) => const Divider(height: 1),
                    itemBuilder: (context, index) {
                      final entry = _entries[index];
                      return ListTile(
                        title: SelectableText(entry.text),
                        subtitle: Text(_formatTimestamp(entry.timestamp)),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

String _formatTimestamp(DateTime timestamp) {
  String twoDigits(int value) => value.toString().padLeft(2, '0');
  return '${twoDigits(timestamp.hour)}:${twoDigits(timestamp.minute)}:${twoDigits(timestamp.second)}';
}
