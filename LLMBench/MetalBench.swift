import Foundation
import Metal

// ---------- Test 1: bandwidth ----------

struct BandwidthSize: Encodable {
    let bufferMb: Int
    let runs: Int
    let bestGbps: Double
    let medianGbps: Double
    let bestUs: Double
    let medianUs: Double
}

struct BandwidthResult: Encodable {
    let method: String
    let gbDefinition: String
    let sizes: [BandwidthSize]
    let skippedSizesMb: [Int]
    let peakMedianGbps: Double
}

// ---------- Test 2: Q4_0 matvec ----------

struct Q4Shape {
    let name: String
    let rows: Int          // output size M
    let cols: Int          // input size K
    let countPerToken: Int // how many times this shape runs per generated token
}

struct Q4ShapeResult: Encodable {
    let name: String
    let rows: Int
    let cols: Int
    let countPerToken: Int
    let weightBytes: Int
    let weightCopiesCycled: Int
    let runs: Int
    let medianUsPerCall: Double
    let bestUsPerCall: Double
    let effectiveGbps: Double
}

struct Q4Correctness: Encodable {
    let rows: Int
    let cols: Int
    let maxAbsError: Double
    let maxAbsReference: Double
    let relativeToMax: Double
    let passed: Bool
}

struct Q4Result: Encodable {
    let shapeSource: String
    let shapeNotes: String
    let method: String
    let correctness: [Q4Correctness]
    let shapes: [Q4ShapeResult]
    let estimatedMatvecUsPerToken: Double
    let estimateNote: String
}

final class MetalBench {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let streamPSO: MTLComputePipelineState
    let q4PSO: MTLComputePipelineState

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw BenchError("No Metal device") }
        guard let q = d.makeCommandQueue() else { throw BenchError("Could not create command queue") }
        guard let lib = d.makeDefaultLibrary() else { throw BenchError("default.metallib is missing from the app bundle") }
        guard let f1 = lib.makeFunction(name: "stream_read"), let f2 = lib.makeFunction(name: "q4_0_matvec") else {
            throw BenchError("Kernel function not found in the Metal library")
        }
        device = d
        queue = q
        streamPSO = try d.makeComputePipelineState(function: f1)
        q4PSO = try d.makeComputePipelineState(function: f2)
        if q4PSO.threadExecutionWidth != 32 {
            throw BenchError("threadExecutionWidth is \(q4PSO.threadExecutionWidth), the Q4_0 kernel assumes 32")
        }
    }

    /// Encodes and runs one command buffer. Returns GPU seconds (GPUEndTime - GPUStartTime).
    private func timedCommandBuffer(_ encode: (MTLComputeCommandEncoder) -> Void) throws -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw BenchError("Could not create command buffer")
        }
        encode(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if cb.status != .completed {
            throw BenchError("Command buffer failed: \(cb.error?.localizedDescription ?? "status \(cb.status.rawValue)")")
        }
        let t = cb.gpuEndTime - cb.gpuStartTime
        if !(t > 0) { throw BenchError("GPU timestamps were not valid (gpuEndTime - gpuStartTime = \(t))") }
        return t
    }

    // ---------- bandwidth ----------

    func bandwidth(sizesMB: [Int], runs: Int, progress: (String) -> Void) throws -> BandwidthResult {
        var out: [BandwidthSize] = []
        var skipped: [Int] = []
        let gridThreads = 1 << 20
        guard let outBuf = device.makeBuffer(length: gridThreads * 4, options: .storageModePrivate) else {
            throw BenchError("Could not allocate the output buffer")
        }
        for mb in sizesMB {
            let bytes = mb * 1_048_576
            guard let src = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                skipped.append(mb)
                progress("\(mb) MB: allocation failed, skipped")
                continue
            }
            memset(src.contents(), 0x3C, bytes)
            var n4 = UInt32(bytes / 16)
            let tgSize = min(256, streamPSO.maxTotalThreadsPerThreadgroup)
            var times: [Double] = []
            for i in 0..<(runs + 3) {
                let t = try timedCommandBuffer { enc in
                    enc.setComputePipelineState(streamPSO)
                    enc.setBuffer(src, offset: 0, index: 0)
                    enc.setBuffer(outBuf, offset: 0, index: 1)
                    enc.setBytes(&n4, length: 4, index: 2)
                    enc.dispatchThreads(MTLSize(width: gridThreads, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                }
                if i >= 3 { times.append(t) } // first 3 runs are warm-up
            }
            let gb = Double(bytes) / 1e9
            let best = times.min() ?? 0
            let med = median(times)
            out.append(BandwidthSize(bufferMb: mb, runs: runs,
                                     bestGbps: gb / best, medianGbps: gb / med,
                                     bestUs: best * 1e6, medianUs: med * 1e6))
            progress("\(mb) MB: best \(String(format: "%.1f", gb / best)) GB/s, median \(String(format: "%.1f", gb / med)) GB/s")
        }
        if out.isEmpty { throw BenchError("No buffer size could be allocated") }
        return BandwidthResult(
            method: "stream_read kernel, \(gridThreads) threads each summing a strided slice of float4, GPUEndTime-GPUStartTime per command buffer, 3 warm-up runs then \(runs) timed runs per size",
            gbDefinition: "1 GB = 1e9 bytes; bytes counted = buffer size (each byte read once)",
            sizes: out, skippedSizesMb: skipped,
            peakMedianGbps: out.map { $0.medianGbps }.max() ?? 0)
    }

    // ---------- Q4_0 matvec ----------

    /// LFM3-1.7B layer shapes from config.json (hidden 2048, intermediate 7680, vocab 128000,
    /// 16 heads x 128, 4 KV heads, 24 layers: 13 conv + 11 attention) and modeling_lfm3.py.
    static let lfm3Shapes: [Q4Shape] = [
        Q4Shape(name: "conv in_proj (hidden -> 3*hidden)", rows: 6144, cols: 2048, countPerToken: 13),
        Q4Shape(name: "conv out_proj (hidden -> hidden)", rows: 2048, cols: 2048, countPerToken: 13),
        Q4Shape(name: "attn q_proj (hidden -> 16*128)", rows: 2048, cols: 2048, countPerToken: 11),
        Q4Shape(name: "attn k_proj (hidden -> 4*128)", rows: 512, cols: 2048, countPerToken: 11),
        Q4Shape(name: "attn v_proj (hidden -> 4*128)", rows: 512, cols: 2048, countPerToken: 11),
        Q4Shape(name: "attn o_proj (16*128 -> hidden)", rows: 2048, cols: 2048, countPerToken: 11),
        Q4Shape(name: "mlp w1 (hidden -> intermediate)", rows: 7680, cols: 2048, countPerToken: 24),
        Q4Shape(name: "mlp w3 (hidden -> intermediate)", rows: 7680, cols: 2048, countPerToken: 24),
        Q4Shape(name: "mlp w2 (intermediate -> hidden)", rows: 2048, cols: 7680, countPerToken: 24),
        Q4Shape(name: "lm_head (hidden -> vocab)", rows: 128000, cols: 2048, countPerToken: 1),
    ]

    private struct Rng {
        var s: UInt64 = 0x9E37_79B9_7F4A_7C15
        mutating func next() -> UInt64 {
            s ^= s << 13
            s ^= s >> 7
            s ^= s << 17
            return s
        }
    }

    /// Fills `p` with `rows * blocksPerRow` random Q4_0 blocks (18 bytes each, scale in 0.01...0.03).
    private func fillQ4(_ p: UnsafeMutableRawPointer, blocks: Int, rng: inout Rng) {
        for b in 0..<blocks {
            let base = p + b * 18
            let scale = Float16(0.01 + 0.02 * Double(rng.next() % 1000) / 1000)
            base.storeBytes(of: scale.bitPattern, as: UInt16.self)
            let lo = rng.next(), hi = rng.next()
            (base + 2).storeBytes(of: lo, as: UInt64.self)       // 18-byte blocks are only 2-byte aligned,
            (base + 10).storeBytes(of: hi, as: UInt64.self)      // so store unaligned via the raw pointer
        }
    }

    private func randomX(_ k: Int, rng: inout Rng) -> [Float] {
        (0..<k).map { _ in Float(Double(rng.next() % 2001) / 1000.0 - 1.0) }
    }

    private func dispatchMatvec(_ enc: MTLComputeCommandEncoder, w: MTLBuffer, x: MTLBuffer, y: MTLBuffer, rows: Int, bpr: Int) {
        var r = UInt32(rows), b = UInt32(bpr)
        enc.setComputePipelineState(q4PSO)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&b, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func correctness(rows: Int, cols: Int) throws -> Q4Correctness {
        var rng = Rng(s: 0x1234_5678_9ABC_DEF1)
        let bpr = cols / 32
        let wBytes = rows * bpr * 18
        guard let w = device.makeBuffer(length: wBytes, options: .storageModeShared),
              let xb = device.makeBuffer(length: cols * 4, options: .storageModeShared),
              let yb = device.makeBuffer(length: rows * 4, options: .storageModeShared) else {
            throw BenchError("Could not allocate correctness buffers")
        }
        fillQ4(w.contents(), blocks: rows * bpr, rng: &rng)
        let x = randomX(cols, rng: &rng)
        memcpy(xb.contents(), x, cols * 4)
        _ = try timedCommandBuffer { enc in dispatchMatvec(enc, w: w, x: xb, y: yb, rows: rows, bpr: bpr) }
        let gpu = yb.contents().bindMemory(to: Float.self, capacity: rows)
        let raw = UnsafeRawPointer(w.contents())
        var maxErr = 0.0, maxRef = 0.0
        for r in 0..<rows {
            var acc = 0.0
            for b in 0..<bpr {
                let base = raw + (r * bpr + b) * 18
                let d = Double(Float16(bitPattern: base.loadUnaligned(as: UInt16.self)))
                var s = 0.0
                for j in 0..<16 {
                    let q = base.load(fromByteOffset: 2 + j, as: UInt8.self)
                    s += Double(Int(q & 0x0F) - 8) * Double(x[b * 32 + j])
                    s += Double(Int(q >> 4) - 8) * Double(x[b * 32 + j + 16])
                }
                acc += d * s
            }
            maxErr = max(maxErr, abs(acc - Double(gpu[r])))
            maxRef = max(maxRef, abs(acc))
        }
        let rel = maxRef > 0 ? maxErr / maxRef : 0
        return Q4Correctness(rows: rows, cols: cols, maxAbsError: maxErr, maxAbsReference: maxRef,
                             relativeToMax: rel, passed: rel < 1e-3)
    }

    func q4Matvec(runs: Int, progress: (String) -> Void) throws -> Q4Result {
        var checks: [Q4Correctness] = []
        for (r, k) in [(64, 512), (50, 2048)] {
            let chk = try correctness(rows: r, cols: k)
            progress("check \(chk.rows)x\(chk.cols): max error \(chk.maxAbsError), relative to max \(chk.relativeToMax), \(chk.passed ? "PASS" : "FAIL")")
            checks.append(chk)
        }
        var results: [Q4ShapeResult] = []
        var rng = Rng()
        for shape in MetalBench.lfm3Shapes {
            let bpr = shape.cols / 32
            let wBytes = shape.rows * bpr * 18
            // Cycle through enough distinct weight copies (about 192 MB) that the weights do not stay in cache,
            // as they would not during real decoding of a 1 GB model.
            let copies = min(64, max(2, Int((192e6 / Double(wBytes)).rounded(.up))))
            guard let first = device.makeBuffer(length: wBytes, options: .storageModeShared),
                  let xb = device.makeBuffer(length: shape.cols * 4, options: .storageModeShared),
                  let yb = device.makeBuffer(length: shape.rows * 4, options: .storageModePrivate) else {
                throw BenchError("Could not allocate buffers for \(shape.name)")
            }
            fillQ4(first.contents(), blocks: shape.rows * bpr, rng: &rng)
            var ws = [first]
            for _ in 1..<copies {
                guard let c = device.makeBuffer(length: wBytes, options: .storageModeShared) else {
                    throw BenchError("Could not allocate weight copy for \(shape.name)")
                }
                memcpy(c.contents(), first.contents(), wBytes)
                ws.append(c)
            }
            let x = randomX(shape.cols, rng: &rng)
            memcpy(xb.contents(), x, shape.cols * 4)
            var perCall: [Double] = []
            for i in 0..<(runs + 2) {
                let t = try timedCommandBuffer { enc in
                    for w in ws { dispatchMatvec(enc, w: w, x: xb, y: yb, rows: shape.rows, bpr: bpr) }
                }
                if i >= 2 { perCall.append(t / Double(copies) * 1e6) } // first 2 runs are warm-up
            }
            let med = median(perCall)
            let best = perCall.min() ?? 0
            results.append(Q4ShapeResult(
                name: shape.name, rows: shape.rows, cols: shape.cols, countPerToken: shape.countPerToken,
                weightBytes: wBytes, weightCopiesCycled: copies, runs: runs,
                medianUsPerCall: med, bestUsPerCall: best,
                effectiveGbps: Double(wBytes) / (med * 1e-6) / 1e9))
            progress("\(shape.name) \(shape.rows)x\(shape.cols): median \(String(format: "%.1f", med)) us, \(String(format: "%.1f", Double(wBytes) / (med * 1e-6) / 1e9)) GB/s")
        }
        let est = results.reduce(0.0) { $0 + $1.medianUsPerCall * Double($1.countPerToken) }
        return Q4Result(
            shapeSource: "https://huggingface.co/LiquidAI/LFM3-1.7B config.json (hidden_size 2048, intermediate_size 7680, vocab_size 128000, num_attention_heads 16, num_key_value_heads 4, head_dim 128, 24 layers: 13 conv + 11 attention, tie_word_embeddings true) and modeling_lfm3.py (Lfm3ShortConv in_proj hidden->3*hidden, out_proj hidden->hidden; Lfm3MLP w1/w3/w2; Lfm3Attention q/k/v/o)",
            shapeNotes: "config.json does not list the conv or attention projection sizes. They are derived from modeling_lfm3.py. The attention gate_proj (hidden -> 16) is too small to matter and is not timed. Real GGUF files may store some tensors (for example the output head) in a different quantization than Q4_0.",
            method: "q4_0_matvec kernel, one simdgroup per output row, 4 rows per threadgroup. Per shape: one command buffer holds N serial dispatches, each reading a different copy of the weights (N copies, about 192 MB total). Time per call = (GPUEndTime - GPUStartTime) / N. 2 warm-up command buffers then \(runs) timed. effective_gbps = weight_bytes / median time per call / 1e9 (weight bytes only, input and output vectors not counted).",
            correctness: checks, shapes: results,
            estimatedMatvecUsPerToken: est,
            estimateNote: "ESTIMATE, not a measurement: sum of median_us_per_call x count_per_token over the listed shapes. Excludes attention, convolution, norms, sampling and all dispatch gaps between ops.")
    }
}
