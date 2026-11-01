import SwiftUI
import UniformTypeIdentifiers

/// Holds UI state and runs the tests on background tasks. Results are shown as text lines and
/// written to JSON in Documents (visible in the Files app under LLMBench).
@MainActor
final class Bench: ObservableObject {
    @Published var device = DeviceInfo.current()

    @Published var bwRunning = false
    @Published var bwLines: [String] = []
    @Published var bwURL: URL?

    @Published var q4Running = false
    @Published var q4Lines: [String] = []
    @Published var q4URL: URL?

    @Published var modelName: String?
    @Published var llamaBusy = false
    @Published var llamaLines: [String] = []
    @Published var samples: [SustainedSample] = []
    @Published var burstURL: URL?
    @Published var sustainedURL: URL?

    private var modelURL: URL?
    private var session: LlamaSession?
    private var sessionTask: Task<Void, Never>?

    func refreshDevice() {
        device = DeviceInfo.current()
    }

    // ---------- test 1 ----------

    func runBandwidth() {
        bwRunning = true
        bwLines = []
        bwURL = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let before = DeviceInfo.current()
        let started = isoNow()
        Task.detached(priority: .userInitiated) { [weak self] in
            var result: BandwidthResult?
            var err: String?
            do {
                let b = try MetalBench()
                result = try b.bandwidth(sizesMB: [64, 128, 256, 512], runs: 20) { line in
                    Task { @MainActor in self?.bwLines.append(line) }
                }
            } catch {
                err = error.localizedDescription
            }
            let report = Report(test: "bandwidth", startedAt: started, deviceBefore: before,
                                deviceAfter: DeviceInfo.current(), result: result, error: err)
            let url = writeReport(report, name: "bandwidth")
            await MainActor.run {
                if let err { self?.bwLines.append("ERROR: \(err)") }
                if let r = result { self?.bwLines.append("Peak median: \(String(format: "%.1f", r.peakMedianGbps)) GB/s") }
                if url == nil { self?.bwLines.append("Could not write the JSON file") }
                self?.bwURL = url
                self?.bwRunning = false
                self?.refreshDevice()
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    // ---------- test 2 ----------

    func runQ4() {
        q4Running = true
        q4Lines = []
        q4URL = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let before = DeviceInfo.current()
        let started = isoNow()
        Task.detached(priority: .userInitiated) { [weak self] in
            var result: Q4Result?
            var err: String?
            do {
                let b = try MetalBench()
                result = try b.q4Matvec(runs: 15) { line in
                    Task { @MainActor in self?.q4Lines.append(line) }
                }
            } catch {
                err = error.localizedDescription
            }
            let report = Report(test: "q4_0_matvec", startedAt: started, deviceBefore: before,
                                deviceAfter: DeviceInfo.current(), result: result, error: err)
            let url = writeReport(report, name: "q4_0_matvec")
            await MainActor.run {
                if let err { self?.q4Lines.append("ERROR: \(err)") }
                if let r = result {
                    self?.q4Lines.append("Estimated matvec time per token: \(String(format: "%.0f", r.estimatedMatvecUsPerToken)) us (estimate, see JSON)")
                }
                if url == nil { self?.q4Lines.append("Could not write the JSON file") }
                self?.q4URL = url
                self?.q4Running = false
                self?.refreshDevice()
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    // ---------- test 3 ----------

    func importModel(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let e):
            llamaLines = ["Import failed: \(e.localizedDescription)"]
        case .success(let urls):
            guard let url = urls.first else { return }
            modelURL?.stopAccessingSecurityScopedResource()
            if !url.startAccessingSecurityScopedResource() {
                llamaLines = ["Could not get access to the file. Is it downloaded to the phone?"]
                return
            }
            modelURL = url
            modelName = url.lastPathComponent
            session = nil // load again for the new file
            llamaLines = ["Selected \(url.lastPathComponent)"]
        }
    }

    private func loadedSession() async throws -> LlamaSession {
        if let s = session { return s }
        guard let url = modelURL else { throw BenchError("Pick the GGUF file first") }
        llamaLines.append("Loading model...")
        let path = url.path
        let s = try await Task.detached(priority: .userInitiated) {
            try LlamaSession(path: path, requestedCtx: 32768)
        }.value
        session = s
        llamaLines.append("Loaded in \(String(format: "%.1f", s.common.loadSeconds)) s, n_ctx \(s.common.nCtxUsed), prompt \(s.common.promptTokens) tokens")
        if s.common.nCtxUsed != s.common.nCtxRequested {
            llamaLines.append("WARNING: n_ctx \(s.common.nCtxRequested) was refused, used \(s.common.nCtxUsed)")
        }
        return s
    }

    func runBurst() {
        guard !llamaBusy else { return }
        llamaBusy = true
        llamaLines = []
        burstURL = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let before = DeviceInfo.current()
        let started = isoNow()
        sessionTask = Task {
            var result: BurstResult?
            var err: String?
            do {
                let s = try await loadedSession()
                result = try await Task.detached(priority: .userInitiated) { [weak self] in
                    try s.burst(decodeTokens: 100, repeats: 3) { line in
                        Task { @MainActor in self?.llamaLines.append(line) }
                    }
                }.value
            } catch {
                err = error.localizedDescription
            }
            let report = Report(test: "llama_burst", startedAt: started, deviceBefore: before,
                                deviceAfter: DeviceInfo.current(), result: result, error: err)
            burstURL = writeReport(report, name: "llama-burst")
            if let err { llamaLines.append("ERROR: \(err)") }
            if let r = result {
                llamaLines.append("Median prefill \(String(format: "%.1f", r.medianPrefillTokensPerSecond)) tok/s, decode \(String(format: "%.1f", r.medianDecodeTokensPerSecond)) tok/s, peak memory \(String(format: "%.0f", r.peakFootprintMb)) MB")
            }
            llamaBusy = false
            refreshDevice()
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    func runSustained() {
        guard !llamaBusy else { return }
        llamaBusy = true
        llamaLines = ["Keep the app open and the screen on for 10 minutes."]
        samples = []
        sustainedURL = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let before = DeviceInfo.current()
        let started = isoNow()
        sessionTask = Task {
            var result: SustainedResult?
            var err: String?
            do {
                let s = try await loadedSession()
                s.cancelled = false
                result = try await Task.detached(priority: .userInitiated) { [weak self] in
                    try s.sustained(seconds: 600, window: 10) { sample in
                        Task { @MainActor in self?.samples.append(sample) }
                    }
                }.value
            } catch {
                err = error.localizedDescription
            }
            let report = Report(test: "llama_sustained", startedAt: started, deviceBefore: before,
                                deviceAfter: DeviceInfo.current(), result: result, error: err)
            sustainedURL = writeReport(report, name: "llama-sustained")
            if let err { llamaLines.append("ERROR: \(err)") }
            if let r = result {
                llamaLines.append("Ran \(String(format: "%.0f", r.ranSeconds)) s. First window \(String(format: "%.1f", r.firstWindowTokensPerSecond)) tok/s, last window \(String(format: "%.1f", r.lastWindowTokensPerSecond)) tok/s, peak memory \(String(format: "%.0f", r.peakFootprintMb)) MB")
            }
            llamaBusy = false
            refreshDevice()
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    func stopLlama() {
        session?.cancelled = true
    }
}
