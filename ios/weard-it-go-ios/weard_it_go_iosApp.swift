//
//  weard_it_go_iosApp.swift
//  weard-it-go-ios
//
//  Created by Daniel Zhao on 10/9/26.
//

import MWDATCore
import SwiftUI

@main
struct weard_it_go_iosApp: App {
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
