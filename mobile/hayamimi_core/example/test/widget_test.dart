// Smoke test: the example app builds and shows its start/stop control
// without needing native model files or a mic (no HayamimiLive.start()
// call happens just from building the widget tree).
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:hayamimi_core_example/main.dart';

void main() {
  testWidgets('SubtitleDemoApp shows the start button', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const SubtitleDemoApp());

    expect(find.text('Start listening'), findsOneWidget);
    expect(find.byIcon(Icons.mic), findsOneWidget);
  });
}
