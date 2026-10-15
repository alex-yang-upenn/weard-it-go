//
//  GlassesStreamController.swift
//  weard-it-go-ios
//

import CoreImage
import CoreMedia
import Foundation
import LiveKit
import MWDATCamera
import MWDATCore
import Observation
import UIKit
import UserNotifications

/// Owns the whole pipeline: Meta AI registration, the glasses `DeviceSession` and
/// camera `Stream`, the downsampler, the LiveKit publisher, and alerts coming
/// back from the backend.
///
/// Starting is event-driven: `start()` joins the LiveKit room and creates the
/// glasses session; the camera is attached when the session reports `.started`.
@Observable
@MainActor
final class GlassesStreamController {
    // MARK: - Glasses state (mirrors the SDK)

    private(set) var registrationState: RegistrationState
    private(set) var activeDeviceName: String?
    private(set) var sessionState: DeviceSessionState = .idle
    private(set) var streamState: MWDATCamera.StreamState = .stopped

    // MARK: - LiveKit state

    private(set) var roomState: LiveKit.ConnectionState = .disconnected
    private(set) var isPublished = false
    /// A recent frame exactly as published (refreshed about twice a second).
    private(set) var preview: UIImage?
    private(set) var stats = Stats()

    struct Stats: Equatable {
        var glassesFPS: Double = 0
        var publishFPS: Double = 0
        var framesPublished = 0
    }

    // MARK: - Alerts from the backend

    private(set) var alerts: [AlertEvent] = []

    // MARK: - Errors

    var errorMessage: String?
    /// The DAT app on the glasses is too old; the alert offers to open Meta AI's update page.
    var needsGlassesAppUpdate = false

    // MARK: - Derived

    var isRegistered: Bool { registrationState == .registered }
    /// The user asked to stream and hasn't stopped (the SDK may still be connecting).
    private(set) var isRunning = false

    var streamStateText: String {
        switch streamState {
        case .streaming: "streaming"
        case .starting: "starting"
        case .waitingForDevice: "waiting for device"
        case .stopping: "stopping"
        case .stopped: "stopped"
        case .paused: "paused (tap glasses to resume)"
        }
    }

    var roomStateText: String {
        switch roomState {
        case .disconnected: "disconnected"
        case .connecting: "connecting…"
        case .reconnecting: "reconnecting…"
        case .connected: isPublished ? "live" : "connected"
        case .disconnecting: "disconnecting…"
        }
    }

    // MARK: - Private

    @ObservationIgnored private let wearables: WearablesInterface
    @ObservationIgnored private let deviceSelector: AutoDeviceSelector
    @ObservationIgnored private var deviceSession: DeviceSession?
    @ObservationIgnored private var camera: MWDATCamera.Camera?
    @ObservationIgnored private var settings = StreamSettings()
    @ObservationIgnored private var publisher: LiveKitPublisher?
    @ObservationIgnored private let identity = "phone-" + UUID().uuidString.prefix(8)
    @ObservationIgnored private let counters = Counters()
    @ObservationIgnored private let sessionTokens = ListenerTokenBag()
    @ObservationIgnored private let streamTokens = ListenerTokenBag()
    @ObservationIgnored private var monitorTasks: [Task<Void, Never>] = []
    @ObservationIgnored private var publisherEventsTask: Task<Void, Never>?
    @ObservationIgnored private let haptics = UINotificationFeedbackGenerator()

    init(wearables: WearablesInterface = Wearables.shared) {
        self.wearables = wearables
        self.registrationState = wearables.registrationState
        self.deviceSelector = AutoDeviceSelector(wearables: wearables)
        startMonitoring()
    }

    // MARK: - Registration

    func register() async {
        do {
            try await wearables.startRegistration()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func unregister() async {
        stop()
        do {
            try await wearables.startUnregistration()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func handleOpenURL(_ url: URL) {
        guard
            let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
            components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
        else { return }
        Task {
            do {
                _ = try await wearables.handleUrl(url)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: - Start / stop

    func start() async {
        guard !isRunning else { return }
        settings = StreamSettings.load()
        guard !settings.backendURL.isEmpty else {
            errorMessage = "Set your backend's address in Settings first, e.g. http://192.168.1.20:8765"
            return
        }
        guard let backendURL = URL(string: settings.backendURL),
              let scheme = backendURL.scheme?.lowercased(), scheme == "http" || scheme == "https"
        else {
            errorMessage = "Backend URL must start with http:// or https://"
            return
        }

        // Asking for camera access switches to the Meta AI app and back.
        do {
            if try await wearables.checkPermissionStatus(.camera) != .granted {
                guard try await wearables.requestPermission(.camera) == .granted else {
                    errorMessage = "Camera permission was denied in Meta AI."
                    return
                }
            }
        } catch {
            errorMessage = error.localizedDescription
            return
        }

        isRunning = true
        stats = Stats()
        preview = nil
        isPublished = false
        counters.reset()
        requestNotificationPermission()

        // Join the room first so the video track exists before frames arrive.
        let publisher = LiveKitPublisher()
        self.publisher = publisher
        observePublisher(publisher)
        do {
            try await publisher.connect(backendURL: backendURL, identity: identity, settings: settings)
        } catch {
            errorMessage = "Couldn't reach the backend at \(settings.backendURL): \(error.localizedDescription)"
            stop()
            return
        }
        guard isRunning else { return }

        do throws(DeviceSessionError) {
            let session = try wearables.createSession(deviceSelector: deviceSelector)
            deviceSession = session
            // Subscribe before start() so no transition is missed.
            observeSession(session)
            sessionState = .starting
            try session.start()
        } catch {
            handleSessionError(error)
            stop()
        }
    }

    func stop() {
        isRunning = false
        camera?.stop()
        deviceSession?.stop()
        if let publisher {
            self.publisher = nil
            Task {
                await publisher.disconnect()
                publisherEventsTask?.cancel()
                isPublished = false
                roomState = .disconnected
            }
        }
    }

    func clearAlerts() {
        alerts.removeAll()
    }

    // MARK: - Session

    private func observeSession(_ session: DeviceSession) {
        session.statePublisher.listen { [weak self] state in
            guard let self else { return }
            Task { @MainActor in self.handleSessionState(state) }
        }.store(in: sessionTokens)
        session.errorPublisher.listen { [weak self] error in
            guard let self else { return }
            Task { @MainActor in self.handleSessionError(error) }
        }.store(in: sessionTokens)
    }

    private func handleSessionError(_ error: DeviceSessionError) {
        if case .datAppOnTheGlassesUpdateRequired = error {
            needsGlassesAppUpdate = true
        }
        errorMessage = error.localizedDescription
    }

    /// Opens Meta AI on the page that updates the app running on the glasses.
    func openGlassesAppUpdate() async {
        needsGlassesAppUpdate = false
        errorMessage = nil
        do {
            try await wearables.openDATGlassesAppUpdate()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func handleSessionState(_ state: DeviceSessionState) {
        sessionState = state
        switch state {
        case .started:
            if isRunning, camera == nil, let deviceSession {
                attachCamera(to: deviceSession)
            }
        case .stopped:
            sessionTokens.clear()
            deviceSession = nil
            if isRunning {
                // Glasses folded / disconnected: the session is terminal, so stop
                // cleanly and let the user start again.
                stop()
            }
        default:
            break
        }
    }

    // MARK: - Camera stream

    private func attachCamera(to session: DeviceSession) {
        let resolution: StreamingResolution = switch settings.sourceResolution {
        case .low: .low
        case .medium: .medium
        case .high: .high
        }
        let config = StreamConfiguration(
            videoCodec: .raw,
            resolution: resolution,
            frameRate: settings.sourceFrameRate
        )
        do {
            guard let newCamera = try session.addCamera(config: config) else {
                errorMessage = "Couldn't add the glasses camera. Try again."
                return
            }
            camera = newCamera
            observeStream(newCamera.stream)
            streamState = .starting
            newCamera.stream.start()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func observeStream(_ stream: MWDATCamera.Stream) {
        guard let publisher else { return }
        stream.statePublisher.listen { [weak self] state in
            guard let self else { return }
            Task { @MainActor in self.handleStreamState(state) }
        }.store(in: streamTokens)
        stream.errorPublisher.listen { [weak self] error in
            guard let self else { return }
            Task { @MainActor in self.errorMessage = error.localizedDescription }
        }.store(in: streamTokens)
        Self.makeFrameListener(
            stream: stream,
            downsampler: FrameDownsampler(outputFrameRate: settings.outputFrameRate, maxLongEdge: settings.maxLongEdge),
            publisher: publisher,
            counters: counters
        ) { [weak self] preview in
            guard let self else { return }
            Task { @MainActor in
                guard UIApplication.shared.applicationState == .active else { return }
                self.preview = UIImage(cgImage: preview)
            }
        }.store(in: streamTokens)
    }

    /// Built in a nonisolated context so the frame callback runs on the SDK's
    /// delivery thread instead of hopping to the main actor.
    private nonisolated static func makeFrameListener(
        stream: MWDATCamera.Stream,
        downsampler: FrameDownsampler,
        publisher: LiveKitPublisher,
        counters: Counters,
        onPreview: @escaping @Sendable (CGImage) -> Void
    ) -> AnyListenerToken {
        stream.videoFramePublisher.listen { frame in
            counters.recordIncoming()
            downsampler.offer(makeImage: {
                if let pixelBuffer = CMSampleBufferGetImageBuffer(frame.sampleBuffer) {
                    return CIImage(cvPixelBuffer: pixelBuffer)
                }
                return frame.makeUIImage().flatMap { CIImage(image: $0) }
            }) { downsampled in
                publisher.capture(downsampled)
                if let preview = downsampled.preview {
                    onPreview(preview)
                }
            }
        }
    }

    private func handleStreamState(_ state: MWDATCamera.StreamState) {
        streamState = state
        if state == .stopped {
            streamTokens.clear()
            // Detach so a later addCamera() can register a new one.
            camera?.stop()
            camera = nil
        }
    }

    // MARK: - Publisher events

    private func observePublisher(_ publisher: LiveKitPublisher) {
        publisherEventsTask?.cancel()
        publisherEventsTask = Task { [weak self, counters] in
            for await event in publisher.events {
                guard let self else { return }
                switch event {
                case .connectionState(let state):
                    roomState = state
                case .published:
                    isPublished = true
                case .framePublished:
                    counters.recordPublished()
                    stats.framesPublished += 1
                case .alert(let alert):
                    receive(alert)
                }
            }
        }
    }

    private func receive(_ alert: AlertEvent) {
        alerts.insert(alert, at: 0)
        if alerts.count > 50 { alerts.removeLast(alerts.count - 50) }
        haptics.notificationOccurred(.warning)

        let content = UNMutableNotificationContent()
        content.title = alert.title
        content.body = alert.detail ?? ""
        content.sound = .default
        let request = UNNotificationRequest(identifier: alert.id.uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    private func requestNotificationPermission() {
        Task {
            _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound])
        }
    }

    // MARK: - Monitoring

    private func startMonitoring() {
        monitorTasks.append(Task { [weak self] in
            guard let wearables = self?.wearables else { return }
            for await state in wearables.registrationStateStream() {
                self?.registrationState = state
            }
        })
        monitorTasks.append(Task { [weak self] in
            guard let selector = self?.deviceSelector else { return }
            for await deviceID in selector.activeDeviceStream() {
                guard let self else { return }
                activeDeviceName = deviceID.map { id in
                    self.wearables.deviceForIdentifier(id)?.nameOrId() ?? id
                }
            }
        })
        // Roll counters into per-second rates for the UI.
        monitorTasks.append(Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard let self else { return }
                let snapshot = counters.drain()
                stats.glassesFPS = Double(snapshot.incoming)
                stats.publishFPS = Double(snapshot.published)
            }
        })
    }
}

/// Thread-safe per-second counters, written from the frame thread.
nonisolated final class Counters: @unchecked Sendable {
    struct Snapshot { var incoming = 0, published = 0 }

    private let lock = NSLock()
    private var current = Snapshot()

    func recordIncoming() {
        lock.withLock { current.incoming += 1 }
    }

    func recordPublished() {
        lock.withLock { current.published += 1 }
    }

    func drain() -> Snapshot {
        lock.withLock {
            defer { current = Snapshot() }
            return current
        }
    }

    func reset() {
        _ = drain()
    }
}
