import CoreLocation
import Foundation
import Observation

/// The `user_id` this phone sends to the backend. Made once per install and kept,
/// so searches (e.g. `test_iphone/main.py --user-id <this> search ...`) find this phone's frames.
enum UserIdentity {
    private static let key = "userID"

    static var userID: String {
        if let existing = UserDefaults.standard.string(forKey: key) { return existing }
        let created = "phone-" + UUID().uuidString.prefix(8).lowercased()
        UserDefaults.standard.set(created, forKey: key)
        return created
    }
}

/// Follows the phone's GPS position while the app is in use. The stream controller
/// forwards each new position to the backend as LiveKit telemetry while streaming.
/// iOS stops sending updates while you stand still; the backend keeps using the last position.
/// Iterating `liveUpdates()` shows the "Allow location?" prompt the first time.
@MainActor
@Observable
final class LocationTracker {
    private(set) var location: CLLocation?
    private(set) var statusText = "Off"

    private var task: Task<Void, Never>?

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
                    } else if let location = update.location, location.horizontalAccuracy >= 0 {
                        self.location = location
                        self.statusText = update.stationary ? "Live (standing still)" : "Live"
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
        statusText = "Off"
    }
}

/// The JSON telemetry sample sent on the LiveKit data topic `telemetry`.
nonisolated struct GpsTelemetry: Encodable, Sendable {
    let latitude: Double
    let longitude: Double
    let accuracy_m: Double
    let phone_time_ms: Int64

    init(_ location: CLLocation) {
        latitude = location.coordinate.latitude
        longitude = location.coordinate.longitude
        accuracy_m = location.horizontalAccuracy
        phone_time_ms = Int64(location.timestamp.timeIntervalSince1970 * 1000)
    }
}
