import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_mobile/server/lan_address.dart';

void main() {
  group('pickLanAddress', () {
    test('returns null when there are no addresses', () {
      expect(pickLanAddress([]), isNull);
    });

    test('skips loopback addresses', () {
      expect(pickLanAddress(['127.0.0.1']), isNull);
    });

    test('skips link-local addresses', () {
      expect(pickLanAddress(['169.254.1.2']), isNull);
    });

    test('picks a normal LAN address', () {
      expect(pickLanAddress(['192.168.1.42']), '192.168.1.42');
    });

    test('skips loopback/link-local and picks the first usable address', () {
      final result = pickLanAddress([
        '127.0.0.1',
        '169.254.3.4',
        '10.0.0.5',
        '192.168.1.42',
      ]);
      expect(result, '10.0.0.5');
    });
  });
}
