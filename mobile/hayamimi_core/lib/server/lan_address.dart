import 'dart:io';

/// Picks the most likely LAN-reachable IPv4 address out of a device's
/// interface addresses, so the UI can show a URL another device on the same
/// network can open.
///
/// Pure selection logic, split out from [currentLanAddress] so it's unit
/// testable without a real network stack. Loopback (`127.x`) and link-local
/// (`169.254.x`, seen when Wi-Fi is connecting or absent) addresses are
/// never useful here and are skipped.
String? pickLanAddress(List<String> ipv4Addresses) {
  for (final address in ipv4Addresses) {
    if (!_isLoopbackOrLinkLocal(address)) {
      return address;
    }
  }
  return null;
}

bool _isLoopbackOrLinkLocal(String address) {
  return address.startsWith('127.') || address.startsWith('169.254.');
}

/// Wires [pickLanAddress] to the real network interfaces on this device.
/// Returns `null` if no usable LAN address is found (e.g. no Wi-Fi).
Future<String?> currentLanAddress() async {
  final interfaces = await NetworkInterface.list(
    type: InternetAddressType.IPv4,
    includeLoopback: false,
    includeLinkLocal: false,
  );
  final addresses = [
    for (final iface in interfaces)
      for (final addr in iface.addresses) addr.address,
  ];
  return pickLanAddress(addresses);
}
