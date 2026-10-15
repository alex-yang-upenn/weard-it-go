//
//  MockGlassesController.swift
//  weard-it-go-ios
//

#if DEBUG

import Foundation
import MWDATMockDevice
import Observation

/// Debug-only simulated Ray-Ban Meta glasses, so the pipeline can be exercised
/// without hardware or Meta AI registration.
///
/// The feed is `Documents/mock_feed.mp4` (HEVC) if present — handy on the
/// simulator, which has no camera — otherwise the phone's back camera.
@Observable
@MainActor
final class MockGlassesController {
    private(set) var isActive = false
    private(set) var errorMessage: String?

    @ObservationIgnored private var device: MockGlasses?

    func toggle() {
        isActive ? disable() : enable()
    }

    /// Launch with `-mockAutoStart YES` to turn on mock glasses and start
    /// streaming without tapping through the UI (for simulator testing).
    static var autoStartRequested: Bool {
        UserDefaults.standard.bool(forKey: "mockAutoStart")
    }

    func autoStart(_ controller: GlassesStreamController) async {
        if !isActive { enable() }
        // The mock device shows up asynchronously after power-on/unfold.
        for _ in 0..<50 where controller.activeDeviceName == nil {
            try? await Task.sleep(for: .milliseconds(100))
        }
        await controller.start()
    }

    private func enable() {
        let kit = MockDeviceKit.shared
        // Starts registered with permissions granted.
        kit.enable()
        do {
            let glasses = try kit.pairGlasses(model: .rayBanMeta)
            glasses.powerOn()
            glasses.unfold()
            glasses.don()
            let video = URL.documentsDirectory.appending(path: "mock_feed.mp4")
            if FileManager.default.fileExists(atPath: video.path) {
                glasses.services.camera.setCameraFeed(fileURL: video)
            } else {
                glasses.services.camera.setCameraFeed(cameraFacing: .back)
            }
            device = glasses
            isActive = true
            errorMessage = nil
        } catch {
            kit.disable()
            errorMessage = error.localizedDescription
        }
    }

    private func disable() {
        MockDeviceKit.shared.disable()
        device = nil
        isActive = false
    }
}

#endif
