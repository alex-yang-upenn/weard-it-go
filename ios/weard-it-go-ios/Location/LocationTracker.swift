import CoreLocation
import Foundation
import Observation

/// One name for this phone, shared by the video stream and location uploads,
/// so the backend can match scenes to locations. New each app launch.
enum DeviceIdentity {
    static let phone = "phone-" + UUID().uuidString.prefix(8)
}

/// Streams the phone's GPS position while the app is in use and uploads a
/// sample to the backend (`POST <backend>/location`) whenever you've moved
/// `minDistance` meters or `maxInterval` seconds have passed since the last one.
/// iOS stops sending updates while you stand still, so nothing is uploaded
/// then; the backend keeps using the last position.
/// Iterating `liveUpdates()` shows the "Allow location?" prompt the first time.
@MainActor
@Observable
final class LocationTracker {
    static let minDistance: CLLocationDistance = 15
    static let maxInterval: TimeInterval = 30

    private(set) var location: CLLocation?
    private(set) var statusText = "Off"
    private(set) var lastSentAt: Date?
    private(set) var sentCount = 0
    private(set) var sendError: String?

    private var task: Task<Void, Never>?
    private var lastSent: CLLocation?
    private var isSending = false

    var isRunning: Bool { task != nil }

    func start() {
        guard task == nil else { return }
        statusText = "Waiting for fix…"
        task = Task { [weak self] in
            do {
                for try await update in CLLocationUpdate.liveUpdates(.otherNavigation) {
                    guard let self else { return }
                    if update.authorizationDenied || update.authorizationDeniedGlobally {
                        self.statusText = "Permission denied — enable in Settings"
                    } else if let location = update.location {
                        self.location = location
                        self.statusText = update.stationary ? "Live (standing still)" : "Live"
                        self.sendIfDue(location)
                    }
                }
            } catch {
                self?.statusText = "Error: \(error.localizedDescription)"
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        lastSent = nil
        statusText = "Off"
    }

    // MARK: - Upload

    private func sendIfDue(_ location: CLLocation) {
        // A negative accuracy means the fix is invalid.
        guard location.horizontalAccuracy >= 0, !isSending else { return }
        if let lastSent {
            let moved = location.distance(from: lastSent) >= Self.minDistance
            let stale = location.timestamp.timeIntervalSince(lastSent.timestamp) >= Self.maxInterval
            guard moved || stale else { return }
        }

        let settings = StreamSettings.load()
        guard let base = URL(string: settings.backendURL), base.scheme?.hasPrefix("http") == true else {
            sendError = "Backend not set — open Settings"
            return
        }

        isSending = true
        let sample = LocationSample(location, identity: DeviceIdentity.phone)
        Task {
            defer { isSending = false }
            do {
                var request = URLRequest(url: base.appending(path: "location"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONEncoder().encode(sample)
                request.timeoutInterval = 10
                let (_, response) = try await URLSession.shared.data(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
                // Only count it as sent once the server has it, so a failure retries on the next update.
                lastSent = location
                lastSentAt = .now
                sentCount += 1
                sendError = nil
            } catch {
                sendError = error.localizedDescription
            }
        }
    }
}

/// The JSON body of `POST /location`. Field names match the scene JSONs (`ts_ms`, `identity`).
nonisolated struct LocationSample: Codable, Sendable {
    let identity: String
    let lat: Double
    let lon: Double
    let accuracy_m: Double
    let ts_ms: Int64

    init(_ location: CLLocation, identity: String) {
        self.identity = identity
        lat = location.coordinate.latitude
        lon = location.coordinate.longitude
        accuracy_m = location.horizontalAccuracy
        ts_ms = Int64(location.timestamp.timeIntervalSince1970 * 1000)
    }
}
