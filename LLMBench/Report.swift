import Foundation
import UIKit
import Metal
import Darwin

// UIDevice and friends are main-actor APIs. The benchmarks run on background threads.
func onMain<T>(_ f: @MainActor () -> T) -> T {
    if Thread.isMainThread {
        return MainActor.assumeIsolated(f)
    }
    return DispatchQueue.main.sync { MainActor.assumeIsolated(f) }
}

func thermalName(_ s: ProcessInfo.ThermalState) -> String {
    switch s {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// Thermal state and battery level, read live. Battery is -1 when iOS does not report it.
func liveVitals() -> (thermal: String, batteryPercent: Double) {
    let thermal = thermalName(ProcessInfo.processInfo.thermalState)
    let level = onMain { UIDevice.current.batteryLevel }
    return (thermal, level < 0 ? -1 : Double(level) * 100)
}

struct DeviceInfo: Codable {
    var modelIdentifier: String
    var iosVersion: String
    var gpuName: String
    var thermalState: String
    var lowPowerMode: Bool
    var batteryLevelPercent: Double
    var batteryState: String

    static func current() -> DeviceInfo {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &buf, &size, nil, 0)
        let vitals = liveVitals()
        let battery: String = onMain {
            switch UIDevice.current.batteryState {
            case .unknown: return "unknown"
            case .unplugged: return "unplugged"
            case .charging: return "charging"
            case .full: return "full"
            @unknown default: return "unknown"
            }
        }
        let version: String = onMain { UIDevice.current.systemVersion }
        return DeviceInfo(
            modelIdentifier: String(cString: buf),
            iosVersion: version,
            gpuName: MTLCreateSystemDefaultDevice()?.name ?? "no Metal device",
            thermalState: vitals.thermal,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            batteryLevelPercent: vitals.batteryPercent,
            batteryState: battery)
    }
}

/// Stable envelope for every exported JSON file. Bump schemaVersion if a field changes meaning.
struct Report<R: Encodable>: Encodable {
    let schemaVersion = 1
    let test: String
    let appVersion = "0.1.0"
    let startedAt: String
    let deviceBefore: DeviceInfo
    let deviceAfter: DeviceInfo
    let result: R?
    let error: String?
}

func isoNow() -> String {
    ISO8601DateFormatter().string(from: Date())
}

/// Writes the report into the app's Documents folder (visible in the Files app) and returns its URL.
func writeReport<R: Encodable>(_ report: Report<R>, name: String) -> URL? {
    let enc = JSONEncoder()
    enc.keyEncodingStrategy = .convertToSnakeCase
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    guard let data = try? enc.encode(report) else { return nil }
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let stamp = report.startedAt.replacingOccurrences(of: ":", with: "-")
    let url = docs.appendingPathComponent("\(name)-\(stamp).json")
    do {
        try data.write(to: url)
        return url
    } catch {
        return nil
    }
}

func median(_ v: [Double]) -> Double {
    let s = v.sorted()
    if s.isEmpty { return 0 }
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

/// Resident memory footprint of this process in MB (what iOS uses for memory limits). -1 if unavailable.
func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}

struct BenchError: LocalizedError {
    let message: String
    init(_ m: String) { message = m }
    var errorDescription: String? { message }
}
