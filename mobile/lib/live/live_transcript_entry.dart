/// One finalized line of live transcription: the recognized text for a
/// single VAD-bounded speech segment, plus when it was produced.
class LiveTranscriptEntry {
  const LiveTranscriptEntry({required this.text, required this.timestamp});

  final String text;
  final DateTime timestamp;
}
