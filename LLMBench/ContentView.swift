import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var bench = Bench()
    @State private var picking = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    bandwidthSection
                    q4Section
                    llamaSection
                }
                .padding()
            }
            .navigationTitle("LLMBench")
            .fileImporter(isPresented: $picking, allowedContentTypes: [.data], allowsMultipleSelection: false) {
                bench.importModel($0)
            }
        }
    }

    private var header: some View {
        let d = bench.device
        return VStack(alignment: .leading, spacing: 4) {
            row("Device", d.modelIdentifier)
            row("iOS", d.iosVersion)
            row("GPU", d.gpuName)
            row("Thermal state", d.thermalState)
            row("Low Power Mode", d.lowPowerMode ? "on" : "off")
            row("Battery", d.batteryLevelPercent < 0 ? "unknown" : String(format: "%.0f%% (%@)", d.batteryLevelPercent, d.batteryState))
            Button("Refresh") { bench.refreshDevice() }
                .buttonStyle(.bordered)
        }
        .font(.footnote.monospaced())
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack { Text(k).foregroundStyle(.secondary); Spacer(); Text(v) }
    }

    private func section<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func lines(_ l: [String]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(l.enumerated()), id: \.offset) { Text($0.element) }
        }
        .font(.caption.monospaced())
    }

    private func share(_ url: URL?) -> some View {
        Group {
            if let url {
                ShareLink("Share results", item: url)
            } else {
                Text("Share results (run the test first)").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var bandwidthSection: some View {
        section("1. Memory bandwidth") {
            Button(bench.bwRunning ? "Running..." : "Run") { bench.runBandwidth() }
                .buttonStyle(.borderedProminent)
                .disabled(bench.bwRunning)
            lines(bench.bwLines)
            share(bench.bwURL)
        }
    }

    private var q4Section: some View {
        section("2. Q4_0 matrix-vector (LFM3 shapes)") {
            Button(bench.q4Running ? "Running..." : "Run") { bench.runQ4() }
                .buttonStyle(.borderedProminent)
                .disabled(bench.q4Running)
            lines(bench.q4Lines)
            share(bench.q4URL)
        }
    }

    private var llamaSection: some View {
        section("3. llama.cpp baseline (LFM3-1.7B Q4_0)") {
            Button(bench.modelName ?? "Pick GGUF file") { picking = true }
                .buttonStyle(.bordered)
                .disabled(bench.llamaBusy)
            HStack {
                Button("Run burst (100 tokens)") { bench.runBurst() }
                    .buttonStyle(.borderedProminent)
                    .disabled(bench.llamaBusy || bench.modelName == nil)
                Button("Run sustained (10 min)") { bench.runSustained() }
                    .buttonStyle(.borderedProminent)
                    .disabled(bench.llamaBusy || bench.modelName == nil)
            }
            if bench.llamaBusy {
                Button("Stop") { bench.stopLlama() }.buttonStyle(.bordered)
            }
            lines(bench.llamaLines)
            if !bench.samples.isEmpty {
                Text("Sustained log (every 10 s)").font(.subheadline)
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(bench.samples) { s in
                        Text(String(format: "%4.0fs  %5.1f tok/s  %@  batt %.0f%%  %.0f MB",
                                    s.elapsedSeconds, s.decodeTokensPerSecond, s.thermalState,
                                    s.batteryPercent, s.footprintMb))
                    }
                }
                .font(.caption2.monospaced())
            }
            Text("Burst results").font(.caption).foregroundStyle(.secondary)
            share(bench.burstURL)
            Text("Sustained results").font(.caption).foregroundStyle(.secondary)
            share(bench.sustainedURL)
        }
    }
}
