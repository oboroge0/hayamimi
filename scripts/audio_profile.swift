import CoreAudio
import Foundation

enum AudioProfileError: Error, CustomStringConvertible {
    case coreAudio(String, OSStatus)
    case deviceNotFound(String)
    case invalidArguments

    var description: String {
        switch self {
        case let .coreAudio(operation, status):
            return "\(operation) に失敗しました（Core Audio: \(status)）"
        case let .deviceNotFound(name):
            return "音声装置が見つかりません: \(name)"
        case .invalidArguments:
            return "使い方: audio_profile create PHYSICAL BLACKHOLE PROFILE_NAME PROFILE_UID | destroy PROFILE_UID | list"
        }
    }
}

func check(_ status: OSStatus, _ operation: String) throws {
    guard status == noErr else {
        throw AudioProfileError.coreAudio(operation, status)
    }
}

func allDevices() throws -> [AudioObjectID] {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    try check(
        AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size
        ),
        "音声装置一覧のサイズ取得"
    )
    var devices = [AudioObjectID](
        repeating: kAudioObjectUnknown,
        count: Int(size) / MemoryLayout<AudioObjectID>.size
    )
    let status = devices.withUnsafeMutableBytes { buffer in
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size,
            buffer.baseAddress!
        )
    }
    try check(status, "音声装置一覧の取得")
    return devices
}

func stringProperty(_ objectID: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var value: Unmanaged<CFString>? = nil
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    try check(
        AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value),
        "音声装置プロパティの取得"
    )
    return value?.takeUnretainedValue() as String? ?? ""
}

func findDevice(named name: String) throws -> (AudioObjectID, String) {
    for device in try allDevices() {
        let deviceName = try stringProperty(device, kAudioObjectPropertyName)
        if deviceName == name {
            let uid = try stringProperty(device, kAudioDevicePropertyDeviceUID)
            return (device, uid)
        }
    }
    throw AudioProfileError.deviceNotFound(name)
}

func findDevice(uid wantedUID: String) throws -> AudioObjectID? {
    for device in try allDevices() {
        if try stringProperty(device, kAudioDevicePropertyDeviceUID) == wantedUID {
            return device
        }
    }
    return nil
}

func destroy(uid: String) throws {
    guard let device = try findDevice(uid: uid) else { return }
    try check(AudioHardwareDestroyAggregateDevice(device), "一時会議出力の削除")
    Thread.sleep(forTimeInterval: 0.25)
}

func create(physicalName: String, blackHoleName: String, profileName: String, profileUID: String) throws {
    if try findDevice(uid: profileUID) != nil {
        try destroy(uid: profileUID)
    }

    let (_, physicalUID) = try findDevice(named: physicalName)
    let (_, blackHoleUID) = try findDevice(named: blackHoleName)
    let subdevices: [[String: Any]] = [
        [kAudioSubDeviceUIDKey: physicalUID],
        [
            kAudioSubDeviceUIDKey: blackHoleUID,
            kAudioSubDeviceDriftCompensationKey: 1,
            kAudioSubDeviceDriftCompensationQualityKey: 127,
        ],
    ]
    let description: [String: Any] = [
        kAudioAggregateDeviceNameKey: profileName,
        kAudioAggregateDeviceUIDKey: profileUID,
        kAudioAggregateDeviceSubDeviceListKey: subdevices,
        kAudioAggregateDeviceMainSubDeviceKey: physicalUID,
        kAudioAggregateDeviceIsPrivateKey: 0,
        kAudioAggregateDeviceIsStackedKey: 1,
    ]
    var aggregateID = AudioObjectID(kAudioObjectUnknown)
    try check(
        AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregateID),
        "一時会議出力の作成"
    )
    Thread.sleep(forTimeInterval: 0.5)
    print(profileName)
}

do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard let command = arguments.first else {
        throw AudioProfileError.invalidArguments
    }
    switch command {
    case "create" where arguments.count == 5:
        try create(
            physicalName: arguments[1],
            blackHoleName: arguments[2],
            profileName: arguments[3],
            profileUID: arguments[4]
        )
    case "destroy" where arguments.count == 2:
        try destroy(uid: arguments[1])
    case "list" where arguments.count == 1:
        for device in try allDevices() {
            let name = try stringProperty(device, kAudioObjectPropertyName)
            let uid = try stringProperty(device, kAudioDevicePropertyDeviceUID)
            print("\(name)\t\(uid)")
        }
    default:
        throw AudioProfileError.invalidArguments
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
