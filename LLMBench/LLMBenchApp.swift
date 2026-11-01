import SwiftUI

@main
struct LLMBenchApp: App {
    init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
