//
//  ContentView.swift
//  weard-it-go-ios
//
//  Created by Daniel Zhao on 10/9/26.
//

import MWDATCore
import SwiftUI

struct ContentView: View {
    @Bindable var controller: GlassesStreamController
    @AppStorage(StreamSettings.Key.backendURL) private var backendURL = StreamSettings().backendURL
    @State private var showSettings = false
    #if DEBUG
    @State private var mockGlasses = MockGlassesController()
    #endif

    var body: some View {
        NavigationStack {
            List {
                glassesSection
                if controller.isRegistered {
                    streamSection
                    alertsSection
                    previewSection
                }
            }
            .navigationTitle("Wear'd It Go")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Settings", systemImage: "slider.horizontal.3") { showSettings = true }
                }
                #if DEBUG
                ToolbarItem(placement: .topBarLeading) {
                    Button(mockGlasses.isActive ? "Mock On" : "Mock Off", systemImage: "ladybug") {
                        mockGlasses.toggle()
                    }
                    .tint(mockGlasses.isActive ? .orange : nil)
                    .disabled(controller.isRunning)
                }
                #endif
            }
            #if DEBUG
            .task {
                if MockGlassesController.autoStartRequested {
                    await mockGlasses.autoStart(controller)
                }
            }
            #endif
            .sheet(isPresented: $showSettings) {
                SettingsView(isStreaming: controller.isRunning)
            }
            .alert(
                "Something went wrong",
                isPresented: Binding(
                    get: { controller.errorMessage != nil },
                    set: { if !$0 { controller.errorMessage = nil } }
                )
            ) {
                if controller.needsGlassesAppUpdate {
                    Button("Update in Meta AI") {
                        Task { await controller.openGlassesAppUpdate() }
                    }
                }
                Button("OK", role: .cancel) {
                    controller.errorMessage = nil
                    controller.needsGlassesAppUpdate = false
                }
            } message: {
                Text(controller.errorMessage ?? "")
            }
        }
    }

    // MARK: - Sections

    private var glassesSection: some View {
        Section("Glasses") {
            LabeledContent("Meta AI", value: registrationText)
            LabeledContent("Device", value: controller.activeDeviceName ?? "None nearby")
            if controller.isRegistered {
                Button("Disconnect from Meta AI", role: .destructive) {
                    Task { await controller.unregister() }
                }
            } else {
                Button("Connect with Meta AI") {
                    Task { await controller.register() }
                }
                .disabled(controller.registrationState == .registering)
            }
        }
    }

    private var streamSection: some View {
        Section("Stream") {
            if controller.isRunning {
                Button("Stop Streaming", systemImage: "stop.fill", role: .destructive) {
                    controller.stop()
                }
            } else {
                Button("Start Streaming", systemImage: "dot.radiowaves.left.and.right") {
                    Task { await controller.start() }
                }
                .disabled(controller.activeDeviceName == nil)
            }
            LabeledContent("Session", value: controller.sessionState.description)
            LabeledContent("Camera", value: controller.streamStateText)
            LabeledContent("Backend") {
                VStack(alignment: .trailing) {
                    Text(controller.roomStateText)
                    Text(backendURL.isEmpty ? "Not set — open Settings" : backendURL)
                        .font(.caption)
                        .foregroundStyle(backendURL.isEmpty ? .red : .secondary)
                }
            }
            LabeledContent("Glasses → phone", value: String(format: "%.0f fps", controller.stats.glassesFPS))
            LabeledContent("Phone → LiveKit", value: String(format: "%.0f fps", controller.stats.publishFPS))
            LabeledContent("Frames published", value: "\(controller.stats.framesPublished)")
        }
    }

    @ViewBuilder
    private var alertsSection: some View {
        if !controller.alerts.isEmpty {
            Section {
                ForEach(controller.alerts) { alert in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text(alert.title).font(.headline)
                            Spacer()
                            Text(alert.receivedAt, style: .time)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if let detail = alert.detail, !detail.isEmpty {
                            Text(detail).font(.subheadline).foregroundStyle(.secondary)
                        }
                    }
                }
            } header: {
                HStack {
                    Text("Alerts from backend")
                    Spacer()
                    Button("Clear") { controller.clearAlerts() }
                        .font(.caption)
                }
            }
        }
    }

    @ViewBuilder
    private var previewSection: some View {
        if let image = controller.preview {
            Section {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 360)
                    .listRowInsets(EdgeInsets())
            } header: {
                Text("Publishing")
            } footer: {
                Text("\(Int(image.size.width))×\(Int(image.size.height))")
            }
        }
    }

    // MARK: - Labels

    private var registrationText: String {
        switch controller.registrationState {
        case .registered: "Connected"
        case .registering: "Connecting…"
        case .available: "Not connected"
        case .unavailable: "Unavailable"
        @unknown default: "Unknown"
        }
    }
}

struct SettingsView: View {
    let isStreaming: Bool
    @Environment(\.dismiss) private var dismiss

    @AppStorage(StreamSettings.Key.backendURL) private var backendURL = StreamSettings().backendURL
    @AppStorage(StreamSettings.Key.sourceResolution) private var sourceResolution = StreamSettings().sourceResolution.rawValue
    @AppStorage(StreamSettings.Key.sourceFrameRate) private var sourceFrameRate = Int(StreamSettings().sourceFrameRate)
    @AppStorage(StreamSettings.Key.outputFrameRate) private var outputFrameRate = StreamSettings().outputFrameRate
    @AppStorage(StreamSettings.Key.maxLongEdge) private var maxLongEdge = StreamSettings().maxLongEdge
    @AppStorage(StreamSettings.Key.maxBitrateKbps) private var maxBitrateKbps = StreamSettings().maxBitrateKbps

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://192.168.1.20:8765", text: $backendURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Backend")
                } footer: {
                    Text("Where backend/live_agent.py runs. It hands the app a LiveKit token and sends alerts back.")
                }

                Section {
                    Picker("Resolution", selection: $sourceResolution) {
                        ForEach(StreamSettings.SourceResolution.allCases) { Text($0.label).tag($0.rawValue) }
                    }
                    Picker("Frame rate", selection: $sourceFrameRate) {
                        ForEach(StreamSettings.sourceFrameRates, id: \.self) { Text("\($0) fps").tag(Int($0)) }
                    }
                } header: {
                    Text("Glasses camera")
                } footer: {
                    Text("Lower settings get less Bluetooth compression, so each frame looks better.")
                }

                Section("Sent to LiveKit") {
                    Picker("Frame rate", selection: $outputFrameRate) {
                        ForEach(StreamSettings.outputFrameRates, id: \.self) { Text("\($0.formatted()) fps").tag($0) }
                    }
                    Picker("Max long edge", selection: $maxLongEdge) {
                        ForEach(StreamSettings.longEdges, id: \.self) { Text("\($0) px").tag($0) }
                    }
                    Picker("Max bitrate", selection: $maxBitrateKbps) {
                        ForEach(StreamSettings.bitratesKbps, id: \.self) { Text("\($0) kbps").tag($0) }
                    }
                }

                if isStreaming {
                    Section {
                        Text("Changes apply the next time you start streaming.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    ContentView(controller: GlassesStreamController())
}
