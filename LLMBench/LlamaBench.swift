import Foundation
import llama

// llama.cpp release pinned in CI (see .github/workflows/build.yml). LFM3 support: PR #31252, merged 2026-10-27.
let llamaCppRelease = "b12024"
let llamaCppCommit = "ddf48f85c933c1f438212a02eb21e7b001d2d37d"

let fixedPrompt = """
You are a helpful assistant running on a phone. Explain, in plain language and in detail, how a \
transformer language model turns a prompt into text, one token at a time. Cover tokenization, \
embeddings, attention, the key-value cache, the feed-forward layers, and how the next token is \
chosen. Then describe why generating text on a phone is limited by memory bandwidth. Keep writing \
until you are stopped.
"""

struct LlamaCommon: Encodable {
    let modelFile: String
    let modelSizeMb: Double
    let llamaCppRelease: String
    let llamaCppCommit: String
    let nCtxRequested: Int
    let nCtxUsed: Int
    let nGpuLayersRequested: Int
    let nBatch: Int
    let promptTokens: Int
    let loadSeconds: Double
    let notes: String
}

struct BurstRun: Encodable {
    let prefillSeconds: Double
    let prefillTokensPerSecond: Double
    let decodeTokens: Int
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let thermalStateAfter: String
    let batteryPercentAfter: Double
}

struct BurstResult: Encodable {
    let common: LlamaCommon
    let runs: [BurstRun]
    let medianPrefillTokensPerSecond: Double
    let medianDecodeTokensPerSecond: Double
    let peakFootprintMb: Double
}

struct SustainedSample: Encodable, Identifiable {
    var id: Int { Int(elapsedSeconds) }
    let elapsedSeconds: Double
    let decodeTokensPerSecond: Double
    let tokensInWindow: Int
    let thermalState: String
    let batteryPercent: Double
    let footprintMb: Double
}

struct SustainedResult: Encodable {
    let common: LlamaCommon
    let targetSeconds: Int
    let ranSeconds: Double
    let stoppedEarly: Bool
    let contextResets: Int
    let samples: [SustainedSample]
    let firstWindowTokensPerSecond: Double
    let lastWindowTokensPerSecond: Double
    let peakFootprintMb: Double
}

/// One loaded model plus context. Create it on a background thread, call it from one thread at a time.
final class LlamaSession: @unchecked Sendable {
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var vocab: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var promptTokens: [llama_token] = []
    private var position = 0
    private let nBatch = 512
    private(set) var common: LlamaCommon!
    var cancelled = false

    init(path: String, requestedCtx: Int) throws {
        let t0 = Date()
        llama_backend_init()
        var mp = llama_model_default_params()
        mp.n_gpu_layers = 999 // all layers
        guard let m = llama_model_load_from_file(path, mp) else {
            throw BenchError("llama.cpp could not load the model file. The file may not be a valid GGUF, or the build may not support its architecture.")
        }
        model = m
        vocab = llama_model_get_vocab(m)

        // Try the requested context first. If iOS refuses the memory, fall back and record what was used.
        var used = 0
        for n in [requestedCtx, 16384, 8192, 4096] where n <= requestedCtx {
            var cp = llama_context_default_params()
            cp.n_ctx = UInt32(n)
            cp.n_batch = UInt32(nBatch)
            cp.n_ubatch = UInt32(nBatch)
            if let c = llama_init_from_model(m, cp) {
                ctx = c
                used = n
                break
            }
        }
        guard ctx != nil else { throw BenchError("llama.cpp could not create a context (tried n_ctx \(requestedCtx) down to 4096)") }

        let sp = llama_sampler_chain_default_params()
        sampler = llama_sampler_chain_init(sp)
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())

        promptTokens = tokenize(fixedPrompt)
        if promptTokens.isEmpty { throw BenchError("Tokenizing the fixed prompt produced no tokens") }

        let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?.doubleValue ?? 0
        common = LlamaCommon(
            modelFile: (path as NSString).lastPathComponent,
            modelSizeMb: size / 1_048_576,
            llamaCppRelease: llamaCppRelease, llamaCppCommit: llamaCppCommit,
            nCtxRequested: requestedCtx, nCtxUsed: used,
            nGpuLayersRequested: 999, nBatch: nBatch,
            promptTokens: promptTokens.count,
            loadSeconds: Date().timeIntervalSince(t0),
            notes: "Greedy sampling. End-of-generation tokens do not stop the run, so the token count is fixed. llama.cpp default flags otherwise (flash attention auto, f16 KV cache).")
    }

    deinit {
        if let s = sampler { llama_sampler_free(s) }
        if let c = ctx { llama_free(c) }
        if let m = model { llama_model_free(m) }
    }

    private func tokenize(_ text: String) -> [llama_token] {
        var buf = [llama_token](repeating: 0, count: text.utf8.count + 16)
        var n = llama_tokenize(vocab, text, Int32(text.utf8.count), &buf, Int32(buf.count), true, true)
        if n < 0 {
            buf = [llama_token](repeating: 0, count: Int(-n))
            n = llama_tokenize(vocab, text, Int32(text.utf8.count), &buf, Int32(buf.count), true, true)
        }
        return n > 0 ? Array(buf[0..<Int(n)]) : []
    }

    private func decode(_ tokens: [llama_token]) throws {
        var t = tokens
        let rc = t.withUnsafeMutableBufferPointer { p in
            llama_decode(ctx, llama_batch_get_one(p.baseAddress, Int32(p.count)))
        }
        if rc != 0 { throw BenchError("llama_decode returned \(rc)") }
        position += tokens.count
    }

    /// Clears the KV cache and processes the fixed prompt. Returns (seconds, first sampled token).
    private func prefill() throws -> (Double, llama_token) {
        llama_memory_clear(llama_get_memory(ctx), true)
        position = 0
        let t0 = ProcessInfo.processInfo.systemUptime
        var i = 0
        while i < promptTokens.count {
            let j = min(i + nBatch, promptTokens.count)
            try decode(Array(promptTokens[i..<j]))
            i = j
        }
        llama_synchronize(ctx)
        let first = llama_sampler_sample(sampler, ctx, -1)
        return (ProcessInfo.processInfo.systemUptime - t0, first)
    }

    func burst(decodeTokens: Int, repeats: Int, progress: (String) -> Void) throws -> BurstResult {
        var runs: [BurstRun] = []
        var peak = physFootprintMB()
        for r in 1...repeats {
            let (pf, first) = try prefill()
            var tok = first
            let t0 = ProcessInfo.processInfo.systemUptime
            for _ in 0..<decodeTokens {
                try decode([tok])
                tok = llama_sampler_sample(sampler, ctx, -1)
            }
            let dt = ProcessInfo.processInfo.systemUptime - t0
            peak = max(peak, physFootprintMB())
            let v = liveVitals()
            let run = BurstRun(prefillSeconds: pf, prefillTokensPerSecond: Double(promptTokens.count) / pf,
                               decodeTokens: decodeTokens, decodeSeconds: dt,
                               decodeTokensPerSecond: Double(decodeTokens) / dt,
                               thermalStateAfter: v.thermal, batteryPercentAfter: v.batteryPercent)
            runs.append(run)
            progress("run \(r): prefill \(String(format: "%.1f", run.prefillTokensPerSecond)) tok/s, decode \(String(format: "%.1f", run.decodeTokensPerSecond)) tok/s")
        }
        return BurstResult(common: common, runs: runs,
                           medianPrefillTokensPerSecond: median(runs.map { $0.prefillTokensPerSecond }),
                           medianDecodeTokensPerSecond: median(runs.map { $0.decodeTokensPerSecond }),
                           peakFootprintMb: peak)
    }

    /// Generates for `seconds`, logging one sample per `window` seconds of wall time.
    func sustained(seconds: Int, window: Double, onSample: (SustainedSample) -> Void) throws -> SustainedResult {
        var samples: [SustainedSample] = []
        var resets = 0
        var peak = physFootprintMB()
        var (_, tok) = try prefill()
        let start = ProcessInfo.processInfo.systemUptime
        var windowStart = start
        var windowDecodeTime = 0.0
        var windowTokens = 0
        var now = start
        while now - start < Double(seconds) && !cancelled {
            if position >= Int(llama_n_ctx(ctx)) - 1 {
                // Context is full: clear it and process the prompt again. This time is not counted as decode time.
                (_, tok) = try prefill()
                resets += 1
            }
            let t0 = ProcessInfo.processInfo.systemUptime
            try decode([tok])
            tok = llama_sampler_sample(sampler, ctx, -1)
            now = ProcessInfo.processInfo.systemUptime
            windowDecodeTime += now - t0
            windowTokens += 1
            if windowTokens % 16 == 0 { peak = max(peak, physFootprintMB()) }
            if now - windowStart >= window {
                let foot = physFootprintMB()
                peak = max(peak, foot)
                let v = liveVitals()
                let s = SustainedSample(elapsedSeconds: (now - start).rounded(),
                                        decodeTokensPerSecond: windowDecodeTime > 0 ? Double(windowTokens) / windowDecodeTime : 0,
                                        tokensInWindow: windowTokens,
                                        thermalState: v.thermal, batteryPercent: v.batteryPercent, footprintMb: foot)
                samples.append(s)
                onSample(s)
                windowStart = now
                windowDecodeTime = 0
                windowTokens = 0
            }
        }
        return SustainedResult(common: common, targetSeconds: seconds, ranSeconds: now - start,
                               stoppedEarly: cancelled, contextResets: resets, samples: samples,
                               firstWindowTokensPerSecond: samples.first?.decodeTokensPerSecond ?? 0,
                               lastWindowTokensPerSecond: samples.last?.decodeTokensPerSecond ?? 0,
                               peakFootprintMb: peak)
    }
}
