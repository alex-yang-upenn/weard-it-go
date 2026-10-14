//
//  StreamSettings.swift
//  weard-it-go-ios
//

import Foundation

/// User-tunable knobs for the glasses → server pipeline, persisted in UserDefaults.
///
/// Two stages: the glasses stream at `sourceResolution` / `sourceFrameRate` over
/// Bluetooth, then the phone throttles to `outputFrameRate`, scales to `maxLongEdge`
/// and publishes to LiveKit capped at `maxBitrateKbps`.
nonisolated struct StreamSettings: Sendable, Equatable {
    enum SourceResolution: String, CaseIterable, Sendable, Identifiable {
        case low, medium, high
        var id: String { rawValue }
        var label: String {
            switch self {
            case .low: "Low (360×640)"
            case .medium: "Medium (504×896)"
            case .high: "High (720×1280)"
            }
        }
    }

    /// Frame rates the DAT SDK accepts for the glasses stream.
    static let sourceFrameRates: [UInt] = [2, 7, 15, 24, 30]
    static let outputFrameRates: [Double] = [2, 5, 10, 15, 24, 30]
    static let longEdges: [Int] = [224, 320, 480, 640, 1280]
    static let bitratesKbps: [Int] = [250, 500, 800, 1500, 3000]

    /// The backend that issues LiveKit tokens, e.g. http://192.168.1.20:8765
    var backendURL: String = ""
    var sourceResolution: SourceResolution = .low
    var sourceFrameRate: UInt = 15
    var outputFrameRate: Double = 10
    var maxLongEdge: Int = 480
    var maxBitrateKbps: Int = 800

    enum Key {
        static let backendURL = "backendURL"
        static let sourceResolution = "sourceResolution"
        static let sourceFrameRate = "sourceFrameRate"
        static let outputFrameRate = "outputFrameRate"
        static let maxLongEdge = "maxLongEdge"
        static let maxBitrateKbps = "maxBitrateKbps"
    }

    static func load(from defaults: UserDefaults = .standard) -> StreamSettings {
        var s = StreamSettings()
        if let v = defaults.string(forKey: Key.backendURL) { s.backendURL = v }
        if let v = defaults.string(forKey: Key.sourceResolution), let r = SourceResolution(rawValue: v) { s.sourceResolution = r }
        if let v = defaults.object(forKey: Key.sourceFrameRate) as? Int { s.sourceFrameRate = UInt(v) }
        if let v = defaults.object(forKey: Key.outputFrameRate) as? Double { s.outputFrameRate = v }
        if let v = defaults.object(forKey: Key.maxLongEdge) as? Int { s.maxLongEdge = v }
        if let v = defaults.object(forKey: Key.maxBitrateKbps) as? Int { s.maxBitrateKbps = v }
        return s
    }
}
