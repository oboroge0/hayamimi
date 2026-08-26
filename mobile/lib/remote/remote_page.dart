import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'remote_connection_state.dart';
import 'remote_event.dart';
import 'remote_transcriber.dart';

/// One line shown in the finals list: a [RemoteFinalEvent] plus the local
/// wall-clock time it arrived, and any [RemoteTranslationEvent] that
/// followed it (the server sends translation as a separate event tied to
/// "whatever the most recent final was").
class _FinalEntry {
  _FinalEntry(this.event) : receivedAt = DateTime.now();

  final RemoteFinalEvent event;
  final DateTime receivedAt;
  String? translation;
  String? translationLang;
}

/// Remote-mode screen: stream this phone's mic to a hayamimi server running
/// `--input ws --serve` and show the subtitle events streamed back, so the
/// PC does the actual recognition (its 5-tier language routing included)
/// and the phone is a thin capture + display client.
class RemotePage extends StatefulWidget {
  const RemotePage({super.key});

  @override
  State<RemotePage> createState() => _RemotePageState();
}

class _RemotePageState extends State<RemotePage> {
  // Default matches realtime_transcribe.py's --ws-port default (8766) for
  // the /ingest WebSocket endpoint — distinct from --serve's HTTP
  // dashboard/SSE port (8833). 10.0.2.2 is the Android emulator's alias
  // for the host machine's localhost.
  final _urlController = TextEditingController(text: 'ws://10.0.2.2:8766/ingest');
  final _wavPathController = TextEditingController();
  final _scrollController = ScrollController();

  final _transcriber = RemoteTranscriber();
  StreamSubscription<RemoteEvent>? _eventsSubscription;
  StreamSubscription<RemoteConnectionState>? _stateSubscription;

  RemoteConnectionState _connectionState = RemoteConnectionState.disconnected;
  String _partialText = '';
  final List<_FinalEntry> _finals = [];
  String? _errorText;
  bool _isConnecting = false;
  bool _isSendingTestWav = false;

  @override
  void initState() {
    super.initState();
    _prefillTestWavPath();
    _eventsSubscription = _transcriber.events.listen(_onEvent);
    _stateSubscription = _transcriber.connectionState.listen((state) {
      if (!mounted) return;
      setState(() => _connectionState = state);
    });
  }

  Future<void> _prefillTestWavPath() async {
    final docsDir = await getApplicationDocumentsDirectory();
    if (!mounted) return;
    setState(() {
      _wavPathController.text = '${docsDir.path}${Platform.pathSeparator}ja_test.wav';
    });
  }

  void _onEvent(RemoteEvent event) {
    if (!mounted) return;
    switch (event) {
      case RemotePartialEvent(:final text):
        setState(() => _partialText = text);
      case RemoteFinalEvent():
        setState(() {
          _partialText = '';
          _finals.add(_FinalEntry(event));
        });
        _scrollToBottom();
      case RemoteTranslationEvent(:final lang, :final text):
        if (_finals.isNotEmpty) {
          setState(() {
            _finals.last.translation = text;
            _finals.last.translationLang = lang;
          });
        }
      case RemoteRefineEvent(:final text, :final lang, :final speaker):
        // Refines replace the most recent matching-speaker text in place of
        // showing a duplicate line; simplest correct behavior for a thin
        // client is to just append it as its own labeled entry.
        setState(() {
          _finals.add(
            _FinalEntry(
              RemoteFinalEvent(text: '[清書] $text', lang: lang, speaker: speaker),
            ),
          );
        });
        _scrollToBottom();
      case RemoteErrorEvent(:final message):
        setState(() => _errorText = message);
      case RemoteReadyEvent():
      case RemoteSessionStartEvent():
      case RemoteUnknownEvent():
        break;
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _toggleConnection() async {
    if (_transcriber.isConnected ||
        _connectionState == RemoteConnectionState.reconnecting) {
      await _transcriber.disconnect();
      return;
    }

    setState(() {
      _isConnecting = true;
      _errorText = null;
    });
    try {
      await _transcriber.connect(_urlController.text.trim());
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorText = e.toString());
    } finally {
      if (mounted) {
        setState(() => _isConnecting = false);
      }
    }
  }

  Future<void> _sendTestWav() async {
    setState(() {
      _isSendingTestWav = true;
      _errorText = null;
    });
    try {
      await _transcriber.sendTestWavFile(
        _urlController.text.trim(),
        _wavPathController.text.trim(),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorText = e.toString());
    } finally {
      if (mounted) {
        setState(() => _isSendingTestWav = false);
      }
    }
  }

  @override
  void dispose() {
    _eventsSubscription?.cancel();
    _stateSubscription?.cancel();
    _transcriber.dispose();
    _urlController.dispose();
    _wavPathController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isConnectedOrConnecting =
        _connectionState == RemoteConnectionState.connected ||
        _connectionState == RemoteConnectionState.reconnecting;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'PC上のhayamimi (--input ws --serve) にマイク音声を送り、'
            '返ってくる字幕を表示します。実機ではPCのLAN IPを指定してください '
            '(例: ws://192.168.1.10:8766/ingest)。',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _urlController,
            enabled: !isConnectedOrConnecting,
            decoration: const InputDecoration(
              labelText: 'サーバーURL (ws://host:port/ingest)',
              border: OutlineInputBorder(),
              helperText: 'エミュレータからホストPCへは 10.0.2.2 でアクセスできます',
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: _isConnecting ? null : _toggleConnection,
            icon: _isConnecting
                ? const SizedBox(
                    height: 16,
                    width: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(isConnectedOrConnecting ? Icons.link_off : Icons.link),
            label: Text(_connectButtonLabel()),
            style: FilledButton.styleFrom(
              backgroundColor: isConnectedOrConnecting
                  ? Theme.of(context).colorScheme.error
                  : null,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Icon(Icons.circle, size: 10, color: _statusColor()),
              const SizedBox(width: 6),
              Text(_statusLabel()),
            ],
          ),
          if (kDebugMode) ...[
            const SizedBox(height: 12),
            _DebugWavCard(
              pathController: _wavPathController,
              isSending: _isSendingTestWav,
              onSend: _isSendingTestWav ? null : _sendTestWav,
            ),
          ],
          if (_errorText != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                _errorText!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ),
          const SizedBox(height: 12),
          if (_partialText.isNotEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                _partialText,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
          const SizedBox(height: 8),
          Expanded(
            child: _finals.isEmpty
                ? const Center(child: Text('字幕はここに表示されます'))
                : ListView.separated(
                    controller: _scrollController,
                    itemCount: _finals.length,
                    separatorBuilder: (_, _) => const Divider(height: 1),
                    itemBuilder: (context, index) => _FinalTile(entry: _finals[index]),
                  ),
          ),
        ],
      ),
    );
  }

  String _connectButtonLabel() {
    switch (_connectionState) {
      case RemoteConnectionState.connected:
        return '切断';
      case RemoteConnectionState.reconnecting:
        return '再接続中... (タップで中止)';
      case RemoteConnectionState.connecting:
      case RemoteConnectionState.disconnected:
        return '接続';
    }
  }

  String _statusLabel() {
    switch (_connectionState) {
      case RemoteConnectionState.connected:
        return '接続中';
      case RemoteConnectionState.connecting:
        return '接続しています...';
      case RemoteConnectionState.reconnecting:
        return '接続が切れました。再接続を試みています...';
      case RemoteConnectionState.disconnected:
        return '未接続';
    }
  }

  Color _statusColor() {
    switch (_connectionState) {
      case RemoteConnectionState.connected:
        return Colors.green;
      case RemoteConnectionState.connecting:
        return Colors.amber;
      case RemoteConnectionState.reconnecting:
        return Colors.orange;
      case RemoteConnectionState.disconnected:
        return Colors.grey;
    }
  }
}

class _FinalTile extends StatelessWidget {
  const _FinalTile({required this.entry});

  final _FinalEntry entry;

  @override
  Widget build(BuildContext context) {
    final event = entry.event;
    return ListTile(
      title: SelectableText(event.text),
      subtitle: entry.translation == null
          ? null
          : SelectableText('→${entry.translationLang}: ${entry.translation}'),
      leading: event.lang.isEmpty
          ? null
          : CircleAvatar(
              radius: 14,
              child: Text(
                event.lang,
                style: const TextStyle(fontSize: 10),
              ),
            ),
      trailing: Text(
        [
          if (event.latencyMs != null) '${event.latencyMs!.round()}ms',
          _formatTimestamp(entry.receivedAt),
        ].join('\n'),
        style: Theme.of(context).textTheme.bodySmall,
        textAlign: TextAlign.right,
      ),
    );
  }
}

String _formatTimestamp(DateTime timestamp) {
  String twoDigits(int value) => value.toString().padLeft(2, '0');
  return '${twoDigits(timestamp.hour)}:${twoDigits(timestamp.minute)}:${twoDigits(timestamp.second)}';
}

/// Debug-only card (see [kDebugMode]) that streams a pushed `.wav` file
/// over its own `/ingest` connection, for exercising the pipeline on an
/// emulator where there's no usable microphone.
class _DebugWavCard extends StatelessWidget {
  const _DebugWavCard({
    required this.pathController,
    required this.isSending,
    required this.onSend,
  });

  final TextEditingController pathController;
  final bool isSending;
  final VoidCallback? onSend;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'デバッグ: テストwav送信 (adb push した16bit PCM wav)',
              style: Theme.of(context).textTheme.labelMedium,
            ),
            const SizedBox(height: 8),
            TextField(
              controller: pathController,
              decoration: const InputDecoration(
                labelText: 'wavファイルパス',
                border: OutlineInputBorder(),
                isDense: true,
              ),
            ),
            const SizedBox(height: 8),
            OutlinedButton.icon(
              onPressed: onSend,
              icon: isSending
                  ? const SizedBox(
                      height: 16,
                      width: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.upload_file),
              label: Text(isSending ? '送信中...' : 'テストwavを送信'),
            ),
          ],
        ),
      ),
    );
  }
}
