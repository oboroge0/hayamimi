import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:path_provider/path_provider.dart';

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
  final _senseVoiceModelDirController = TextEditingController();
  final _lidModelDirController = TextEditingController();
  final _scrollController = ScrollController();

  final _transcriber = LiveTranscriber();
  StreamSubscription<LiveTranscriptEntry>? _entriesSubscription;
  StreamSubscription<bool>? _decodingSubscription;
  StreamSubscription<LiveTranscriptEntry>? _refineEntriesSubscription;
  StreamSubscription<LiveTranscriptEntry>? _draftsSubscription;

  // Draft ("発話中の暫定字幕"): the most recent in-progress decode while a
  // VAD segment is still open -- see hayamimi_core's live/draft_pass.dart.
  // Never stored in a list like _entries/_refineEntries; just the latest
  // one, cleared the moment a real final for that segment arrives.
  LiveTranscriptEntry? _draftEntry;

  final ModelKind _modelKind = ModelKind.zipformerTransducer;
  final List<LiveTranscriptEntry> _entries = [];
  bool _isStarting = false;
  bool _isDecoding = false;
  String? _errorText;

  // Multilingual routing (see docs/MOBILE.md "Multi-language routing on
  // mobile"): off by default (single ja model, unchanged prior behavior).
  // Switching this needs senseVoiceModelDir/lidModelDir filled in too.
  RoutingProfile _routingProfile = RoutingProfile.jaOnly;

  // Two-pass "refine" (清書): see lib/live/refine_pass.dart for why the
  // trigger conditions and defaults are what they are.
  final List<LiveTranscriptEntry> _refineEntries = [];
  bool _autoRefineEnabled = false;
  bool _isRefining = false;

  // Debug-only "wavから清書テスト" path (kDebugMode only, see build() below):
  // runs LiveTranscriber.runDebugWavRefineTest against a pushed wav file, so
  // the refine pass's audio-combining logic can be exercised on an
  // emulator, which has no usable microphone for a real live session.
  final _debugWavPathController = TextEditingController();
  bool _isRunningDebugWavTest = false;
  String? _debugWavError;
  DebugRefineTestResult? _debugWavResult;

  // Debug-only "wavをリアルタイムペースで流す" path (kDebugMode only): streams a
  // wav file through LiveTranscriber.startDebugWavStream at (roughly) real
  // time pace, so the draft pass (and the fast-final/refine passes) can be
  // exercised end to end on an emulator, which has no usable microphone.
  final _debugWavStreamPathController = TextEditingController();
  String? _debugWavStreamError;

  // "Other app integration": an in-app HTTP server that mirrors the
  // desktop hayamimi subtitle feed (scripts/subtitle_server.py) so an OBS
  // browser source or browser on the same LAN can subscribe to this
  // phone's live transcript. See lib/server/.
  final _broadcastServer = SubtitleBroadcastServer();
  bool _isBroadcastEnabled = false;
  bool _isBroadcastStarting = false;
  String? _broadcastError;
  String? _lanAddress;

  /// Fallback language tag for broadcast events when a segment carries no
  /// routed language (a plain [RoutingProfile.jaOnly] session, which only
  /// ever runs the ja model) — see [FinalSubtitleEvent.lang] and
  /// [_onEntry], which prefers [LiveTranscriptEntry.lang] when set.
  static const _broadcastLang = 'ja';

  @override
  void initState() {
    super.initState();
    _prefillDefaultPaths();
    _entriesSubscription = _transcriber.entries.listen(_onEntry);
    _decodingSubscription = _transcriber.decoding.listen((decoding) {
      if (!mounted) return;
      setState(() => _isDecoding = decoding);
    });
    _refineEntriesSubscription = _transcriber.refineEntries.listen(
      _onRefineEntry,
    );
    _draftsSubscription = _transcriber.drafts.listen(_onDraft);
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
      _senseVoiceModelDirController.text =
          '${docsDir.path}$sep' 'sense_voice';
      _lidModelDirController.text = '${docsDir.path}$sep' 'lid';
      _debugWavPathController.text = '${docsDir.path}${sep}test.wav';
      _debugWavStreamPathController.text = '${docsDir.path}${sep}test.wav';
    });
  }

  void _onDraft(LiveTranscriptEntry entry) {
    if (!mounted) return;
    setState(() => _draftEntry = entry);
    if (_broadcastServer.isRunning) {
      _broadcastServer.broadcast(PartialSubtitleEvent(entry.text));
    }
  }

  void _onEntry(LiveTranscriptEntry entry) {
    if (!mounted) return;
    // The final supersedes whatever draft was showing for this segment.
    setState(() {
      _draftEntry = null;
      _entries.add(entry);
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    });
    if (_broadcastServer.isRunning) {
      _broadcastServer.broadcast(
        FinalSubtitleEvent(
          text: entry.text,
          // Use the segment's routed language when available (a
          // [RoutingProfile.jaSenseVoice] session) — falling back to the
          // fixed ja default would mislabel every non-ja routed segment.
          lang: entry.lang ?? _broadcastLang,
          latencyMs: entry.latencyMs,
        ),
      );
    }
  }

  void _onRefineEntry(LiveTranscriptEntry entry) {
    if (!mounted) return;
    setState(() => _refineEntries.add(entry));
  }

  Future<void> _triggerRefine() async {
    setState(() => _isRefining = true);
    try {
      await _transcriber.refineNow();
    } finally {
      if (mounted) {
        setState(() => _isRefining = false);
      }
    }
  }

  void _toggleAutoRefine(bool enabled) {
    setState(() => _autoRefineEnabled = enabled);
    _transcriber.autoRefineEnabled = enabled;
  }

  Future<void> _runDebugWavRefineTest() async {
    setState(() {
      _isRunningDebugWavTest = true;
      _debugWavError = null;
      _debugWavResult = null;
    });
    try {
      final result = await LiveTranscriber.runDebugWavRefineTest(
        modelDir: _modelDirController.text.trim(),
        wavPath: _debugWavPathController.text.trim(),
        routingProfile: _routingProfile,
        senseVoiceModelDir: _routingProfile.dualConfirmed
            ? _senseVoiceModelDirController.text.trim()
            : null,
        lidModelDir: _routingProfile.dualConfirmed
            ? _lidModelDirController.text.trim()
            : null,
      );
      if (!mounted) return;
      setState(() => _debugWavResult = result);
      // Also feed the results through the same entry/broadcast pipeline a
      // real live session uses (_onEntry/_onRefineEntry) — this is the only
      // way to exercise the transcript list and the 配信サーバー broadcast
      // end to end on an emulator, which has no usable microphone (see the
      // class doc on `runDebugWavRefineTest`).
      final now = DateTime.now();
      if (result.segment1Text.isNotEmpty) {
        _onEntry(
          LiveTranscriptEntry(
            text: result.segment1Text,
            timestamp: now,
            lang: result.segment1Lang,
          ),
        );
      }
      if (result.segment2Text.isNotEmpty) {
        _onEntry(
          LiveTranscriptEntry(
            text: result.segment2Text,
            timestamp: now,
            lang: result.segment2Lang,
          ),
        );
      }
      if (result.refineText.isNotEmpty) {
        _onRefineEntry(
          LiveTranscriptEntry(
            text: result.refineText,
            timestamp: now,
            lang: result.refineLang,
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _debugWavError = e.toString());
    } finally {
      if (mounted) {
        setState(() => _isRunningDebugWavTest = false);
      }
    }
  }

  Future<void> _startDebugWavStream() async {
    setState(() => _debugWavStreamError = null);
    try {
      // Awaits until the whole file has streamed through -- setState calls
      // that happen in the meantime (via _onEntry/_onDraft/_onRefineEntry,
      // each firing as the pipeline produces output) keep the "streaming"
      // button state and the transcript lists live while this runs.
      await _transcriber.startDebugWavStream(
        modelKind: _modelKind,
        modelDir: _modelDirController.text.trim(),
        vadModelPath: _vadModelPathController.text.trim(),
        wavPath: _debugWavStreamPathController.text.trim(),
        routingProfile: _routingProfile,
        senseVoiceModelDir: _routingProfile.dualConfirmed
            ? _senseVoiceModelDirController.text.trim()
            : null,
        lidModelDir: _routingProfile.dualConfirmed
            ? _lidModelDirController.text.trim()
            : null,
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _debugWavStreamError = e.toString());
    } finally {
      if (mounted) {
        setState(() {}); // isDebugStreaming is back to false
      }
    }
  }

  Future<void> _stopDebugWavStream() async {
    await _transcriber.stopDebugWavStream();
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _toggleBroadcast() async {
    if (_broadcastServer.isRunning) {
      await _broadcastServer.stop();
      if (!mounted) return;
      setState(() {
        _isBroadcastEnabled = false;
        _lanAddress = null;
      });
      return;
    }

    setState(() {
      _isBroadcastStarting = true;
      _broadcastError = null;
    });
    try {
      await _broadcastServer.start();
      final lanAddress = await currentLanAddress();
      if (!mounted) return;
      setState(() {
        _isBroadcastEnabled = true;
        _lanAddress = lanAddress;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _broadcastError = e.toString());
    } finally {
      if (mounted) {
        setState(() => _isBroadcastStarting = false);
      }
    }
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
        routingProfile: _routingProfile,
        senseVoiceModelDir: _routingProfile.dualConfirmed
            ? _senseVoiceModelDirController.text.trim()
            : null,
        lidModelDir: _routingProfile.dualConfirmed
            ? _lidModelDirController.text.trim()
            : null,
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
    _refineEntriesSubscription?.cancel();
    _draftsSubscription?.cancel();
    _transcriber.dispose();
    _broadcastServer.stop();
    _modelDirController.dispose();
    _vadModelPathController.dispose();
    _senseVoiceModelDirController.dispose();
    _lidModelDirController.dispose();
    _debugWavPathController.dispose();
    _debugWavStreamPathController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isRunning = _transcriber.isRunning;
    return Padding(
      padding: const EdgeInsets.all(16),
      // SingleChildScrollView (rather than Expanded lists filling all
      // remaining space) so the page degrades to scrolling instead of
      // overflowing once the debug-only refine card is showing too — the
      // fixed controls above the transcript/清書 lists can already get tall
      // enough to not leave room for both at a comfortable size.
      child: SingleChildScrollView(
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
            const SizedBox(height: 12),
            DropdownButtonFormField<RoutingProfile>(
              initialValue: _routingProfile,
              decoration: const InputDecoration(
                labelText: 'Language routing',
                border: OutlineInputBorder(),
              ),
              items: [
                for (final profile in RoutingProfile.values)
                  DropdownMenuItem(value: profile, child: Text(profile.label)),
              ],
              onChanged: isRunning
                  ? null
                  : (profile) =>
                        setState(() => _routingProfile = profile!),
            ),
            if (_routingProfile.dualConfirmed) ...[
              const SizedBox(height: 12),
              TextField(
                controller: _senseVoiceModelDirController,
                enabled: !isRunning,
                decoration: const InputDecoration(
                  labelText: 'SenseVoice model directory',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _lidModelDirController,
                enabled: !isRunning,
                decoration: const InputDecoration(
                  labelText: 'whisper-tiny LID model directory',
                  border: OutlineInputBorder(),
                ),
              ),
            ],
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
            const SizedBox(height: 12),
            _BroadcastServerCard(
              isEnabled: _isBroadcastEnabled,
              isStarting: _isBroadcastStarting,
              lanAddress: _lanAddress,
              port: _broadcastServer.boundPort ?? _broadcastServer.port,
              errorText: _broadcastError,
              onToggle: _isBroadcastStarting ? null : _toggleBroadcast,
            ),
            const SizedBox(height: 8),
            _RefinePassCard(
              isRunning: isRunning,
              isRefining: _isRefining,
              autoRefineEnabled: _autoRefineEnabled,
              bufferedSeconds: _transcriber.refineBufferedSeconds,
              onRefineNow: isRunning && !_isRefining ? _triggerRefine : null,
              onToggleAuto: isRunning ? _toggleAutoRefine : null,
            ),
            if (kDebugMode) ...[
              const SizedBox(height: 12),
              _DebugWavRefineCard(
                pathController: _debugWavPathController,
                isRunning: _isRunningDebugWavTest,
                errorText: _debugWavError,
                result: _debugWavResult,
                onRun: _isRunningDebugWavTest ? null : _runDebugWavRefineTest,
              ),
              const SizedBox(height: 12),
              _DebugWavStreamCard(
                pathController: _debugWavStreamPathController,
                isStreaming: _transcriber.isDebugStreaming,
                errorText: _debugWavStreamError,
                onStart: _transcriber.isDebugStreaming
                    ? null
                    : _startDebugWavStream,
                onStop: _transcriber.isDebugStreaming
                    ? _stopDebugWavStream
                    : null,
              ),
            ],
            const SizedBox(height: 16),
            // Draft ("発話中の暫定字幕"): shown above the finalized list, PC
            // dashboard-style ("いま聞き取り中"), so the user sees text grow
            // while still speaking instead of only after a pause.
            if (isRunning || _transcriber.isDebugStreaming || _draftEntry != null)
              _DraftStrip(entry: _draftEntry),
            const SizedBox(height: 8),
            Text('文字起こし', style: Theme.of(context).textTheme.labelLarge),
            const SizedBox(height: 4),
            SizedBox(
              height: 240,
              child: _entries.isEmpty
                  ? const Center(child: Text('Transcript will appear here.'))
                  : ListView.separated(
                      controller: _scrollController,
                      itemCount: _entries.length,
                      separatorBuilder: (_, _) => const Divider(height: 1),
                      itemBuilder: (context, index) {
                        final entry = _entries[index];
                        return ListTile(
                          leading: entry.lang == null
                              ? null
                              : _LangBadge(lang: entry.lang!),
                          title: SelectableText(entry.text),
                          subtitle: Text(_formatTimestamp(entry.timestamp)),
                        );
                      },
                    ),
            ),
            const Divider(height: 16),
            Text('清書', style: Theme.of(context).textTheme.labelLarge),
            const SizedBox(height: 4),
            SizedBox(
              height: 200,
              child: _refineEntries.isEmpty
                  ? const Center(child: Text('清書結果はここに表示されます。'))
                  : ListView.separated(
                      itemCount: _refineEntries.length,
                      separatorBuilder: (_, _) => const Divider(height: 1),
                      itemBuilder: (context, index) {
                        final entry = _refineEntries[index];
                        return ListTile(
                          leading: entry.lang == null
                              ? null
                              : _LangBadge(lang: entry.lang!),
                          title: SelectableText(entry.text),
                          subtitle: Text(_formatTimestamp(entry.timestamp)),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Small pill showing a transcript entry's routed language (e.g. "JA",
/// "EN") — only shown for a [RoutingProfile.jaSenseVoice] session, since a
/// plain single-model session already knows its one language from context.
class _LangBadge extends StatelessWidget {
  const _LangBadge({required this.lang});

  final String lang;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 8),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.secondaryContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        lang.toUpperCase(),
        style: Theme.of(context).textTheme.labelSmall?.copyWith(
          color: Theme.of(context).colorScheme.onSecondaryContainer,
        ),
      ),
    );
  }
}

String _formatTimestamp(DateTime timestamp) {
  String twoDigits(int value) => value.toString().padLeft(2, '0');
  return '${twoDigits(timestamp.hour)}:${twoDigits(timestamp.minute)}:${twoDigits(timestamp.second)}';
}

/// Toggle + status card for the "other app integration" broadcast server:
/// lets OBS or a browser on the same LAN subscribe to this phone's live
/// transcript at `http://<lan ip>:<port>/`.
class _BroadcastServerCard extends StatelessWidget {
  const _BroadcastServerCard({
    required this.isEnabled,
    required this.isStarting,
    required this.lanAddress,
    required this.port,
    required this.errorText,
    required this.onToggle,
  });

  final bool isEnabled;
  final bool isStarting;
  final String? lanAddress;
  final int port;
  final String? errorText;
  final VoidCallback? onToggle;

  @override
  Widget build(BuildContext context) {
    final url = lanAddress == null ? null : 'http://$lanAddress:$port/';
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('配信サーバー'),
              subtitle: const Text('同じLAN内のOBS/ブラウザに字幕を配信します（画面ON中のみ）'),
              value: isEnabled,
              onChanged: isStarting || onToggle == null
                  ? null
                  : (_) => onToggle!(),
              secondary: isStarting
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.podcasts),
            ),
            if (isEnabled)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: url == null
                    ? const Text('LAN上のIPアドレスが見つかりません（Wi-Fi未接続？）')
                    : SelectableText(url),
              ),
            if (errorText != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  errorText!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// Manual "清書" button + "自動清書" toggle for the two-pass refine feature.
/// See `lib/live/refine_pass.dart` for why auto mode defaults off and what
/// triggers it, and `LiveTranscriber.refineNow`/`autoRefineEnabled` for the
/// glue this drives.
class _RefinePassCard extends StatelessWidget {
  const _RefinePassCard({
    required this.isRunning,
    required this.isRefining,
    required this.autoRefineEnabled,
    required this.bufferedSeconds,
    required this.onRefineNow,
    required this.onToggleAuto,
  });

  final bool isRunning;
  final bool isRefining;
  final bool autoRefineEnabled;
  final double bufferedSeconds;
  final VoidCallback? onRefineNow;
  final ValueChanged<bool>? onToggleAuto;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: onRefineNow,
                    icon: isRefining
                        ? const SizedBox(
                            height: 16,
                            width: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.auto_fix_high),
                    label: Text(
                      isRunning
                          ? '清書 (${bufferedSeconds.toStringAsFixed(1)}s分)'
                          : '清書',
                    ),
                  ),
                ),
              ],
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: const Text('自動清書'),
              subtitle: const Text('無音が続いたら自動で清書します（既定オフ：バッテリー・発熱に配慮）'),
              value: autoRefineEnabled,
              onChanged: onToggleAuto,
            ),
          ],
        ),
      ),
    );
  }
}

/// Debug-only (see `kDebugMode`) card that exercises the refine pass's
/// audio-combining logic against a pushed test wav, splitting it into two
/// halves to stand in for two VAD segments — see
/// `LiveTranscriber.runDebugWavRefineTest`. Exists because an emulator has
/// no usable microphone, so there's no other way to exercise this path
/// there.
class _DebugWavRefineCard extends StatelessWidget {
  const _DebugWavRefineCard({
    required this.pathController,
    required this.isRunning,
    required this.errorText,
    required this.result,
    required this.onRun,
  });

  final TextEditingController pathController;
  final bool isRunning;
  final String? errorText;
  final DebugRefineTestResult? result;
  final VoidCallback? onRun;

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
              'デバッグ: wavから清書テスト（モデルは上のModel directoryを使用）',
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
              onPressed: onRun,
              icon: isRunning
                  ? const SizedBox(
                      height: 16,
                      width: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.science),
              label: Text(isRunning ? '実行中...' : 'wavから清書テスト'),
            ),
            if (errorText != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  errorText!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              ),
            if (result != null) ...[
              const SizedBox(height: 8),
              _DebugResultLine(
                label: '個別1',
                text: result!.segment1Text,
                lang: result!.segment1Lang,
              ),
              _DebugResultLine(
                label: '個別2',
                text: result!.segment2Text,
                lang: result!.segment2Lang,
              ),
              const Divider(height: 16),
              _DebugResultLine(
                label: '清書 (結合)',
                text: result!.refineText,
                lang: result!.refineLang,
                bold: true,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// One labeled result row in [_DebugWavRefineCard]: the decoded text, with
/// a [_LangBadge] alongside it when the test ran with routing enabled
/// (`lang` non-null).
class _DebugResultLine extends StatelessWidget {
  const _DebugResultLine({
    required this.label,
    required this.text,
    this.lang,
    this.bold = false,
  });

  final String label;
  final String text;
  final String? lang;
  final bool bold;

  @override
  Widget build(BuildContext context) {
    final style = bold ? const TextStyle(fontWeight: FontWeight.bold) : null;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Text(
              '$label: ${text.isEmpty ? '(empty)' : text}',
              style: style,
            ),
          ),
          if (lang != null) ...[
            const SizedBox(width: 8),
            _LangBadge(lang: lang!),
          ],
        ],
      ),
    );
  }
}

/// "発話中の暫定字幕" strip shown above the finalized transcript list while a
/// session (or the debug wav stream) is running: the most recent draft
/// decode, dimmed/italic like the PC dashboard's "いま聞き取り中" panel, or a
/// waiting placeholder before any draft has arrived yet.
class _DraftStrip extends StatelessWidget {
  const _DraftStrip({required this.entry});

  final LiveTranscriptEntry? entry;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hasDraft = entry != null && entry!.text.isNotEmpty;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (hasDraft && entry!.lang != null) ...[
            _LangBadge(lang: entry!.lang!),
            const SizedBox(width: 8),
          ],
          Expanded(
            child: Text(
              hasDraft ? entry!.text : 'マイクの音声を待っています…',
              style: theme.textTheme.bodyLarge?.copyWith(
                fontStyle: FontStyle.italic,
                color: hasDraft
                    ? theme.colorScheme.onSurfaceVariant
                    : theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.6),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Debug-only "wavをリアルタイムペースで流す" card: streams a wav file through
/// LiveTranscriber.startDebugWavStream so the draft pass (and the fast-final
/// /refine passes) can be exercised end to end on an emulator, which has no
/// usable microphone -- see LiveTranscriber.startDebugWavStream's doc.
class _DebugWavStreamCard extends StatelessWidget {
  const _DebugWavStreamCard({
    required this.pathController,
    required this.isStreaming,
    required this.errorText,
    required this.onStart,
    required this.onStop,
  });

  final TextEditingController pathController;
  final bool isStreaming;
  final String? errorText;
  final VoidCallback? onStart;
  final VoidCallback? onStop;

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
              'デバッグ: wavをリアルタイムペースで流す（ドラフト検証用）',
              style: Theme.of(context).textTheme.labelMedium,
            ),
            const SizedBox(height: 8),
            TextField(
              controller: pathController,
              enabled: !isStreaming,
              decoration: const InputDecoration(
                labelText: 'wavファイルパス（16kHzモノラル）',
                border: OutlineInputBorder(),
                isDense: true,
              ),
            ),
            const SizedBox(height: 8),
            OutlinedButton.icon(
              onPressed: isStreaming ? onStop : onStart,
              icon: isStreaming
                  ? const SizedBox(
                      height: 16,
                      width: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.play_arrow),
              label: Text(isStreaming ? '停止' : 'リアルタイム再生で流す'),
            ),
            if (errorText != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  errorText!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
