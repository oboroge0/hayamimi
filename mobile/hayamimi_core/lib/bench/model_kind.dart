/// Supported offline ASR model families for the RTF bench.
///
/// Only [zipformerTransducer] is wired up end-to-end today. The other
/// entries exist so the UI and [BenchRunner] can be extended without
/// reshaping the enum later (SenseVoice / Paraformer / CTC, etc.).
enum ModelKind {
  zipformerTransducer('Zipformer (transducer)'),
  senseVoice('SenseVoice (not yet implemented)'),
  paraformer('Paraformer (not yet implemented)'),
  ctc('CTC (not yet implemented)');

  const ModelKind(this.label);

  final String label;

  bool get isImplemented => this == ModelKind.zipformerTransducer;
}
