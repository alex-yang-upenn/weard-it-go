//
//  weard_it_go_iosApp.swift
//  weard-it-go-ios
//
//  Created by Daniel Zhao on 10/9/26.
//

import MWDATCore
import SwiftUI
import UserNotifications

@main
struct weard_it_go_iosApp: App {
    @UIApplicationDelegateAdaptor private var appDelegate: AppDelegate
    @State private var controller: GlassesStreamController

    init() {
        do {
            try Wearables.configure()
        } catch {
            assertionFailure("Failed to configure Wearables SDK: \(error)")
        }
        _controller = State(wrappedValue: GlassesStreamController())
    }

    var body: some Scene {
        WindowGroup {
            ContentView(controller: controller)
                // Meta AI calls back here after registration and permission prompts.
                .onOpenURL { controller.handleOpenURL($0) }
        }
    }
}

/// Shows backend alerts as banners even while the app is open.
final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        return true
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }
}
