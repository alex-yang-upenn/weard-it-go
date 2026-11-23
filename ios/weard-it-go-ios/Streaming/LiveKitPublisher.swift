//
//  LiveKitPublisher.swift
//  weard-it-go-ios
//

import CoreVideo
import Foundation
import LiveKit

/// Publishes downsampled glasses video and GPS telemetry to a LiveKit room.
///
/// The backend's orchestrator hands out the room: `POST <backend>/sessions {"user_id": …}`
/// returns `{"session_id": …, "livekit_url": "ws://…:7880", "token": …}`. The backend's
/// ingest agent joins every room on its own. LiveKit handles reconnects, bandwidth
/// adaptation and NAT traversal.
nonisolated final class LiveKitPublisher: NSObject, @unchecked Sendable {
    enum Event: Sendable {
        case connectionState(LiveKit.ConnectionState)
        case published
        case framePublished
    }

    let events: AsyncStream<Event>
    private let continuation: AsyncStream<Event>.Continuation

    private let room = Room()
    private let lock = NSLock()
    private var track: LocalVideoTrack?
    private var publishStarted = false
    private var encoding = VideoEncoding(maxBitrate: 800_000, maxFps: 10)

    override init() {
        (events, continuation) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(64))
        super.init()
        room.add(delegate: self)
    }

    deinit {
        continuation.finish()
    }

    // MARK: - Connect / disconnect

    private struct CreateSessionRequest: Encodable {
        let user_id: String
    }

    private struct CreateSessionResponse: Decodable {
        let session_id: String
        let livekit_url: String
        let token: String
    }

    func connect(backendURL: URL, userID: String, settings: StreamSettings) async throws {
        var request = URLRequest(url: backendURL.appending(path: "sessions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(CreateSessionRequest(user_id: userID))
        request.timeoutInterval = 10

        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        let session = try JSONDecoder().decode(CreateSessionResponse.self, from: data)

        try await room.connect(url: session.livekit_url, token: session.token)

        let fps = Int(settings.outputFrameRate.rounded(.up))
        let newTrack = await LocalVideoTrack.createBufferTrack(
            name: "glasses",
            source: .camera,
            options: BufferCaptureOptions(fps: fps)
        )
        lock.withLock {
            track = newTrack
            publishStarted = false
            encoding = VideoEncoding(maxBitrate: settings.maxBitrateKbps * 1000, maxFps: fps)
        }
    }

    func disconnect() async {
        lock.withLock {
            track = nil
            publishStarted = false
        }
        await room.disconnect()
    }

    // MARK: - Telemetry

    /// Sends one JSON telemetry sample (e.g. GPS) on the data topic the backend's ingest listens to.
    func sendTelemetry(_ telemetryJSON: Data) async throws {
        try await room.localParticipant.publish(
            data: telemetryJSON,
            options: DataPublishOptions(topic: "telemetry", reliable: true)
        )
    }

    // MARK: - Frames

    /// Feeds one frame to the video track. The track is published after the
    /// first frame, since LiveKit needs the dimensions up front.
    func capture(_ frame: DownsampledFrame) {
        lock.lock()
        let currentTrack = track
        let shouldPublish = currentTrack != nil && !publishStarted
        if shouldPublish { publishStarted = true }
        let encoding = encoding
        lock.unlock()

        guard let track = currentTrack, let capturer = track.capturer as? BufferCapturer else { return }
        capturer.capture(frame.pixelBuffer, timeStampNs: frame.timeStampNs)
        continuation.yield(.framePublished)

        if shouldPublish {
            Task {
                do {
                    let options = VideoPublishOptions(encoding: encoding, simulcast: false)
                    try await room.localParticipant.publish(videoTrack: track, options: options)
                    continuation.yield(.published)
                } catch {
                    // Let the next frame retry.
                    lock.withLock { publishStarted = false }
                }
            }
        }
    }
}

extension LiveKitPublisher: RoomDelegate {
    func room(_: Room, didUpdateConnectionState connectionState: LiveKit.ConnectionState, from _: LiveKit.ConnectionState) {
        continuation.yield(.connectionState(connectionState))
    }
}
