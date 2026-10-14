//
//  LiveKitPublisher.swift
//  weard-it-go-ios
//

import CoreVideo
import Foundation
import LiveKit

/// Something the backend's model noticed, delivered over LiveKit RPC ("alert").
nonisolated struct AlertEvent: Identifiable, Sendable, Decodable {
    let id = UUID()
    let title: String
    let detail: String?
    let receivedAt = Date()

    private enum CodingKeys: String, CodingKey { case title, detail }
}

/// Publishes downsampled glasses video to a LiveKit room and receives alerts
/// back from the backend agent.
///
/// The backend hands out credentials: `GET <backend>/token?identity=…` returns
/// `{"url": "ws://…:7880", "token": "…", "room": "…"}`. LiveKit itself handles
/// reconnects, bandwidth adaptation and NAT traversal.
nonisolated final class LiveKitPublisher: NSObject, @unchecked Sendable {
    enum Event: Sendable {
        case connectionState(LiveKit.ConnectionState)
        case published
        case framePublished
        case alert(AlertEvent)
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

    private struct TokenResponse: Decodable {
        let url: String
        let token: String
    }

    func connect(backendURL: URL, identity: String, settings: StreamSettings) async throws {
        var components = URLComponents(url: backendURL.appending(path: "token"), resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "identity", value: identity)]
        guard let tokenURL = components?.url else { throw URLError(.badURL) }

        let (data, response) = try await URLSession.shared.data(from: tokenURL)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        let credentials = try JSONDecoder().decode(TokenResponse.self, from: data)

        try await room.connect(url: credentials.url, token: credentials.token)

        try await room.registerRpcMethod("alert") { [continuation] invocation in
            if let event = try? JSONDecoder().decode(AlertEvent.self, from: Data(invocation.payload.utf8)) {
                continuation.yield(.alert(event))
            }
            return "ok"
        }

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
