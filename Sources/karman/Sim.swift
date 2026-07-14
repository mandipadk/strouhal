import Foundation
import Metal
import CryptoKit

enum Precision: String {
    case fp32, fp16s
    var ddfBytes: Int { self == .fp32 ? 4 : 2 }
}

struct Params {
    var nx: UInt32, ny: UInt32, nz: UInt32
    var parity: UInt32
    var omega: Float
    var omegaMinus: Float
    var ulid: Float
    var fx: Float
    var fy: Float = 0, fz: Float = 0
    var pad0: UInt32 = 0, pad1: UInt32 = 0
}

struct InitParams {
    var nx: UInt32, ny: UInt32, nz: UInt32, mode: UInt32
    var amplitude: Float
    var tau: Float = 1.0
    var pad1: Float = 0, pad2: Float = 0
}

enum Cell: UInt8 {
    case fluid = 0
    case solid = 1
    case lid = 2 // solid + moving (+x)
}

struct Pipelines {
    let step: MTLComputePipelineState
    let moments: MTLComputePipelineState
    let initField: MTLComputePipelineState
}

final class GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private var pipelines: [Precision: Pipelines] = [:]
    private let source: String

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw KarmanError.noDevice
        }
        self.device = device
        self.queue = queue
        guard let url = Bundle.module.url(forResource: "Kernels", withExtension: "metal") else {
            throw KarmanError.message("Kernels.metal resource missing")
        }
        self.source = try String(contentsOf: url, encoding: .utf8)
    }

    func pipelines(for precision: Precision) throws -> Pipelines {
        if let p = pipelines[precision] { return p }
        let options = MTLCompileOptions()
        // The determinism contract: no fast-math transforms, precise library
        // functions. FMA contraction within a statement remains (deterministic
        // per build; pinned cross-device later via explicit fma if needed).
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        options.preprocessorMacros = ["FPXX": (precision == .fp32 ? "float" : "half") as NSString]
        let library = try device.makeLibrary(source: source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw KarmanError.message("kernel \(name) not found")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        let p = Pipelines(step: try pipeline("step"),
                          moments: try pipeline("momentsEven"),
                          initField: try pipeline("initField"))
        pipelines[precision] = p
        return p
    }
}

enum KarmanError: Error, CustomStringConvertible {
    case noDevice
    case message(String)
    var description: String {
        switch self {
        case .noDevice: return "no Metal device"
        case .message(let m): return m
        }
    }
}

/// Lock-guarded mutable state shared with command-buffer completion handlers.
final class RunState: @unchecked Sendable {
    private let lock = NSLock()
    private var _gpuSeconds: Double = 0
    private var _firstErrorCode: UInt?
    func record(seconds: Double, errorCode: UInt?) {
        lock.lock()
        _gpuSeconds += seconds
        if _firstErrorCode == nil, let code = errorCode { _firstErrorCode = code }
        lock.unlock()
    }
    var gpuSeconds: Double { lock.lock(); defer { lock.unlock() }; return _gpuSeconds }
    var firstErrorCode: UInt? { lock.lock(); defer { lock.unlock() }; return _firstErrorCode }
}

final class Simulation {
    let gpu: GPU
    let precision: Precision
    let pipes: Pipelines
    let nx: Int, ny: Int, nz: Int
    var cells: Int { nx * ny * nz }
    let fBuf: MTLBuffer
    let flagBuf: MTLBuffer
    let solidMaskBuf: MTLBuffer
    let lidMaskBuf: MTLBuffer
    let momentsBuf: MTLBuffer
    var omega: Float
    var omegaMinus: Float
    var ulidTarget: Float
    var rampSteps: Int
    var force: SIMD3<Float>
    private(set) var stepsDone: Int = 0
    private let runState = RunState()
    var gpuSeconds: Double { runState.gpuSeconds }

    // D3Q19 direction table mirroring Kernels.metal (order is load-bearing).
    static let cx: [Int] = [0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0]
    static let cy: [Int] = [0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1]
    static let cz: [Int] = [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1]

    /// tau -> (omega+, omega-) via the TRT magic parameter Lambda.
    /// Lambda = 1/4 is the stability optimum; 3/16 makes halfway bounce-back
    /// walls viscosity-exact. omegaMinus == omega+ recovers SRT identically.
    static func trtOmegas(tau: Double, lambda: Double?) -> (Float, Float) {
        let wp = 1.0 / tau
        guard let lambda else { return (Float(wp), Float(wp)) } // SRT
        let tm = 0.5 + lambda / (tau - 0.5)
        return (Float(wp), Float(1.0 / tm))
    }

    init(gpu: GPU, precision: Precision = .fp32, nx: Int, ny: Int, nz: Int,
         omega: Float, omegaMinus: Float? = nil,
         ulid: Float = 0, rampSteps: Int = 0, force: SIMD3<Float> = .zero,
         flags: (Int, Int, Int) -> Cell) throws {
        self.gpu = gpu
        self.precision = precision
        self.pipes = try gpu.pipelines(for: precision)
        self.nx = nx; self.ny = ny; self.nz = nz
        self.omega = omega
        self.omegaMinus = omegaMinus ?? omega
        self.ulidTarget = ulid
        self.rampSteps = rampSteps
        self.force = force
        let n = nx * ny * nz

        guard let f = gpu.device.makeBuffer(length: 19 * n * precision.ddfBytes, options: .storageModeShared),
              let fl = gpu.device.makeBuffer(length: n, options: .storageModeShared),
              let sm = gpu.device.makeBuffer(length: n * 4, options: .storageModeShared),
              let lm = gpu.device.makeBuffer(length: n * 4, options: .storageModeShared),
              let mo = gpu.device.makeBuffer(length: n * 16, options: .storageModeShared) else {
            throw KarmanError.message("buffer allocation failed")
        }
        fBuf = f; flagBuf = fl; solidMaskBuf = sm; lidMaskBuf = lm; momentsBuf = mo
        memset(fBuf.contents(), 0, fBuf.length) // shifted equilibrium at rest is exactly 0

        // Flags and per-cell neighbor masks (bit i-1: neighbor at n + c_i).
        let flagPtr = flagBuf.contents().bindMemory(to: UInt8.self, capacity: n)
        var cellFlags = [Cell](repeating: .fluid, count: n)
        for z in 0..<nz { for y in 0..<ny { for x in 0..<nx {
            let c = flags(x, y, z)
            cellFlags[(z * ny + y) * nx + x] = c
            flagPtr[(z * ny + y) * nx + x] = c == .fluid ? 0 : 1
        }}}
        let sPtr = solidMaskBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        let lPtr = lidMaskBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        for z in 0..<nz { for y in 0..<ny { for x in 0..<nx {
            let idx = (z * ny + y) * nx + x
            var sMask: UInt32 = 0, lMask: UInt32 = 0
            if cellFlags[idx] == .fluid {
                for i in 1..<19 {
                    let xn = (x + Self.cx[i] + nx) % nx
                    let yn = (y + Self.cy[i] + ny) % ny
                    let zn = (z + Self.cz[i] + nz) % nz
                    switch cellFlags[(zn * ny + yn) * nx + xn] {
                    case .fluid: break
                    case .solid: sMask |= 1 << UInt32(i - 1)
                    case .lid: sMask |= 1 << UInt32(i - 1); lMask |= 1 << UInt32(i - 1)
                    }
                }
            }
            sPtr[idx] = sMask; lPtr[idx] = lMask
        }}}
    }

    private func currentLid(atStep t: Int) -> Float {
        guard rampSteps > 0 else { return ulidTarget }
        return ulidTarget * Float(min(1.0, Double(t) / Double(rampSteps)))
    }

    private func params(step t: Int) -> Params {
        Params(nx: UInt32(nx), ny: UInt32(ny), nz: UInt32(nz),
               parity: UInt32(t & 1), omega: omega, omegaMinus: omegaMinus,
               ulid: currentLid(atStep: t), fx: force.x, fy: force.y, fz: force.z)
    }

    /// Run `count` steps. Chunked into command buffers small enough to stay
    /// under the GPU watchdog; up to two buffers in flight.
    func run(steps count: Int) throws {
        let n = cells
        let stepsPerCB = max(2, min(2000, 40_000_000 / max(1, n / 16))) & ~1
        let inflight = DispatchSemaphore(value: 2)
        let state = runState

        var remaining = count
        while remaining > 0 {
            let batch = min(stepsPerCB, remaining)
            inflight.wait()
            guard let cb = gpu.queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                throw KarmanError.message("command buffer creation failed")
            }
            enc.setComputePipelineState(pipes.step)
            enc.setBuffer(fBuf, offset: 0, index: 0)
            enc.setBuffer(flagBuf, offset: 0, index: 1)
            enc.setBuffer(solidMaskBuf, offset: 0, index: 2)
            enc.setBuffer(lidMaskBuf, offset: 0, index: 3)
            for s in 0..<batch {
                var p = params(step: stepsDone + s)
                enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc.endEncoding()
            cb.addCompletedHandler { cb in
                state.record(seconds: cb.gpuEndTime - cb.gpuStartTime,
                             errorCode: (cb.error as? MTLCommandBufferError)?.code.rawValue)
                inflight.signal()
            }
            cb.commit()
            stepsDone += batch
            remaining -= batch
        }
        inflight.wait(); inflight.wait()
        inflight.signal(); inflight.signal()
        if let code = runState.firstErrorCode {
            throw KarmanError.message("GPU command buffer failed (code \(code)) — watchdog or recovery event")
        }
    }

    /// (ux, uy, uz, rho) per cell. Only valid after an even number of steps.
    func probeMoments() throws -> UnsafeBufferPointer<SIMD4<Float>> {
        precondition(stepsDone % 2 == 0, "moments probe requires even step count")
        guard let cb = gpu.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw KarmanError.message("command buffer creation failed")
        }
        enc.setComputePipelineState(pipes.moments)
        enc.setBuffer(fBuf, offset: 0, index: 0)
        enc.setBuffer(flagBuf, offset: 0, index: 1)
        enc.setBuffer(momentsBuf, offset: 0, index: 2)
        var p = params(step: stepsDone)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
        enc.dispatchThreads(MTLSize(width: cells, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let ptr = momentsBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: cells)
        return UnsafeBufferPointer(start: ptr, count: cells)
    }

    /// mode 1 = Taylor-Green (2D, one period per box), amplitude in lattice units.
    func initField(mode: UInt32, amplitude: Float) throws {
        guard let cb = gpu.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw KarmanError.message("command buffer creation failed")
        }
        enc.setComputePipelineState(pipes.initField)
        enc.setBuffer(fBuf, offset: 0, index: 0)
        var p = InitParams(nx: UInt32(nx), ny: UInt32(ny), nz: UInt32(nz),
                           mode: mode, amplitude: amplitude, tau: 1.0 / omega)
        enc.setBytes(&p, length: MemoryLayout<InitParams>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: cells, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    var stateDigest: String {
        let data = Data(bytes: fBuf.contents(), count: fBuf.length)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Total shifted-density sum (double accumulation) — mass drift probe.
    func massSum() -> Double {
        var total = 0.0
        switch precision {
        case .fp32:
            let ptr = fBuf.contents().bindMemory(to: Float.self, capacity: 19 * cells)
            for i in 0..<(19 * cells) { total += Double(ptr[i]) }
        case .fp16s:
            let ptr = fBuf.contents().bindMemory(to: Float16.self, capacity: 19 * cells)
            for i in 0..<(19 * cells) { total += Double(ptr[i]) }
        }
        return total
    }
}
