import Foundation
import Metal
import CryptoKit

public enum Precision: String, Sendable {
    case fp32, fp16s
    public var ddfBytes: Int { self == .fp32 ? 4 : 2 }
}

struct Params {
    var nx: UInt32, ny: UInt32, nz: UInt32, parity: UInt32
    var omega: Float
    var omegaMinus: Float
    var uin: Float
    var cSmago: Float
    var fx: Float, fy: Float, fz: Float
    var writeForce: UInt32
    var ulidX: Float, ulidY: Float, ulidZ: Float
    var inflowUniform: UInt32 = 0
    var spongeX0: Float = 1e9
    var spongeInvW: Float = 0
    var spongeTau: Float = 1.0
    var useEps: UInt32 = 0
}

/// Mirror of the colorize kernel's VizParams.
public struct VizParams {
    public var nx: UInt32, ny: UInt32, nz: UInt32, zSlice: UInt32
    public var uref: Float
    public var useEps: UInt32
    public var pad0: Float = 0, pad1: Float = 0
    public init(nx: UInt32, ny: UInt32, nz: UInt32, zSlice: UInt32,
                uref: Float, useEps: UInt32) {
        self.nx = nx; self.ny = ny; self.nz = nz; self.zSlice = zSlice
        self.uref = uref; self.useEps = useEps
    }
}

struct InitParams {
    var nx: UInt32, ny: UInt32, nz: UInt32, mode: UInt32
    var amplitude: Float
    var tau: Float = 1.0
    var pad1: Float = 0, pad2: Float = 0
}

public enum Cell: UInt8, Sendable {
    case fluid = 0
    case solid = 1
    case lid = 2     // solid + moving (+x)
    case inflow = 3  // velocity Dirichlet, parabolic +x profile
    case outflow = 4 // zero-gradient copy of the -x neighbor
}

struct Pipelines {
    let step: MTLComputePipelineState
    let moments: MTLComputePipelineState
    let initField: MTLComputePipelineState
}

/// Metal device plus a lazily-compiled pipeline cache.
///
/// Sendable by contract: `device` and `queue` are thread-safe Metal objects,
/// `source` is immutable, and the two caches are guarded by `cacheLock`. The
/// lock is not decorative — the instrument builds cases on a background
/// thread while the render loop draws, so both touch the cache concurrently.
public final class GPU: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    private let cacheLock = NSLock()
    private var pipelines: [Precision: Pipelines] = [:]
    private var libraries: [Precision: MTLLibrary] = [:]
    private let source: String

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw StrouhalError.noDevice
        }
        self.device = device
        self.queue = queue
        guard let url = Bundle.module.url(forResource: "Kernels", withExtension: "metal") else {
            throw StrouhalError.message("Kernels.metal resource missing")
        }
        self.source = try String(contentsOf: url, encoding: .utf8)
    }

    func pipelines(for precision: Precision) throws -> Pipelines {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return try pipelinesLocked(for: precision)
    }

    /// Caller must hold `cacheLock`.
    private func pipelinesLocked(for precision: Precision) throws -> Pipelines {
        if let p = pipelines[precision] { return p }
        let options = MTLCompileOptions()
        // The determinism contract: no fast-math transforms, precise library
        // functions. FMA contraction within a statement remains (deterministic
        // per build; pinned cross-device later via explicit fma if needed).
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        options.preprocessorMacros = ["FPXX": (precision == .fp32 ? "float" : "half") as NSString]
        let library = try device.makeLibrary(source: source, options: options)
        libraries[precision] = library
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw StrouhalError.message("kernel \(name) not found")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        let p = Pipelines(step: try pipeline("step"),
                          moments: try pipeline("momentsEven"),
                          initField: try pipeline("initField"))
        pipelines[precision] = p
        return p
    }

    /// Visualization pipelines: the |u| colorize compute kernel and a
    /// fullscreen-quad render pipeline for the given drawable pixel format.
    public func makeVizPipelines(precision: Precision, pixelFormat: MTLPixelFormat)
        throws -> (colorize: MTLComputePipelineState, quad: MTLRenderPipelineState) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        _ = try pipelinesLocked(for: precision) // ensures the library is compiled
        guard let library = libraries[precision] else {
            throw StrouhalError.message("library missing")
        }
        guard let cfn = library.makeFunction(name: "colorize"),
              let vfn = library.makeFunction(name: "fsqVertex"),
              let ffn = library.makeFunction(name: "fsqFragment") else {
            throw StrouhalError.message("viz kernels not found")
        }
        let colorize = try device.makeComputePipelineState(function: cfn)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = pixelFormat
        let quad = try device.makeRenderPipelineState(descriptor: desc)
        return (colorize, quad)
    }
}

public enum StrouhalError: Error, CustomStringConvertible {
    case noDevice
    case message(String)
    public var description: String {
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

public final class Simulation {
    public let gpu: GPU
    public let precision: Precision
    let pipes: Pipelines
    public let nx: Int, ny: Int, nz: Int
    public var cells: Int { nx * ny * nz }
    public let fBuf: MTLBuffer
    let flagBuf: MTLBuffer
    let solidMaskBuf: MTLBuffer
    let lidMaskBuf: MTLBuffer
    let momentsBuf: MTLBuffer
    public var omega: Float
    public var omegaMinus: Float
    public var lidVel: SIMD3<Float>
    public var uinTarget: Float
    public var rampSteps: Int
    public var force: SIMD3<Float>
    public var cSmago: Float
    public var inflowUniform = false
    public var sponge: (x0: Float, width: Float, tau: Float)? = nil
    let forceBuf: MTLBuffer
    private(set) var epsBuf: MTLBuffer
    private(set) var usesEps = false
    public private(set) var stepsDone: Int = 0
    private let runState = RunState()
    public var gpuSeconds: Double { runState.gpuSeconds }

    // D3Q19 direction table mirroring Kernels.metal (order is load-bearing).
    public static let cx: [Int] = [0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0]
    public static let cy: [Int] = [0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1]
    public static let cz: [Int] = [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1]

    /// tau -> (omega+, omega-) via the TRT magic parameter Lambda.
    /// Lambda = 1/4 is the stability optimum; 3/16 makes halfway bounce-back
    /// walls viscosity-exact. omegaMinus == omega+ recovers SRT identically.
    public static func trtOmegas(tau: Double, lambda: Double?) -> (Float, Float) {
        let wp = 1.0 / tau
        guard let lambda else { return (Float(wp), Float(wp)) } // SRT
        let tm = 0.5 + lambda / (tau - 0.5)
        return (Float(wp), Float(1.0 / tm))
    }

    public init(gpu: GPU, precision: Precision = .fp32, nx: Int, ny: Int, nz: Int,
         omega: Float, omegaMinus: Float? = nil,
         lid: SIMD3<Float> = .zero, uin: Float = 0, rampSteps: Int = 0,
         force: SIMD3<Float> = .zero, cSmago: Float = 0,
         wantsForces: Bool = false,
         flags: (Int, Int, Int) -> Cell) throws {
        self.gpu = gpu
        self.precision = precision
        self.pipes = try gpu.pipelines(for: precision)
        self.nx = nx; self.ny = ny; self.nz = nz
        self.omega = omega
        self.omegaMinus = omegaMinus ?? omega
        self.lidVel = lid
        self.uinTarget = uin
        self.rampSteps = rampSteps
        self.force = force
        self.cSmago = cSmago
        let n = nx * ny * nz

        guard let f = gpu.device.makeBuffer(length: 19 * n * precision.ddfBytes, options: .storageModeShared),
              let fl = gpu.device.makeBuffer(length: n, options: .storageModeShared),
              let sm = gpu.device.makeBuffer(length: n * 4, options: .storageModeShared),
              let lm = gpu.device.makeBuffer(length: n * 4, options: .storageModeShared),
              let mo = gpu.device.makeBuffer(length: n * 16, options: .storageModeShared),
              let fo = gpu.device.makeBuffer(length: wantsForces ? n * 16 : 16, options: .storageModeShared),
              let ep = gpu.device.makeBuffer(length: 16, options: .storageModeShared) else {
            throw StrouhalError.message("buffer allocation failed")
        }
        fBuf = f; flagBuf = fl; solidMaskBuf = sm; lidMaskBuf = lm; momentsBuf = mo; forceBuf = fo
        epsBuf = ep
        memset(fBuf.contents(), 0, fBuf.length) // shifted equilibrium at rest is exactly 0

        // Flags and per-cell neighbor masks (bit i-1: neighbor at n + c_i).
        let flagPtr = flagBuf.contents().bindMemory(to: UInt8.self, capacity: n)
        var cellFlags = [Cell](repeating: .fluid, count: n)
        for z in 0..<nz { for y in 0..<ny { for x in 0..<nx {
            let c = flags(x, y, z)
            cellFlags[(z * ny + y) * nx + x] = c
            flagPtr[(z * ny + y) * nx + x] = c.rawValue
        }}}
        let sPtr = solidMaskBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        let lPtr = lidMaskBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        for z in 0..<nz { for y in 0..<ny { for x in 0..<nx {
            let idx = (z * ny + y) * nx + x
            var sMask: UInt32 = 0, lMask: UInt32 = 0
            let c = cellFlags[idx]
            if c == .fluid {
                for i in 1..<19 {
                    let xn = (x + Self.cx[i] + nx) % nx
                    let yn = (y + Self.cy[i] + ny) % ny
                    let zn = (z + Self.cz[i] + nz) % nz
                    switch cellFlags[(zn * ny + yn) * nx + xn] {
                    case .fluid: break
                    case .solid: sMask |= 1 << UInt32(i - 1)
                    case .lid, .inflow, .outflow:
                        sMask |= 1 << UInt32(i - 1); lMask |= 1 << UInt32(i - 1)
                    }
                }
            }
            sPtr[idx] = sMask; lPtr[idx] = lMask
        }}}
    }

    private func ramp(_ target: Float, atStep t: Int) -> Float {
        guard rampSteps > 0 else { return target }
        let x = min(1.0, Double(t) / Double(rampSteps))
        // C¹ cosine ramp: a linear ramp's end-kink is a broadband acoustic
        // kick that rings essentially undamped in a low-viscosity duct
        // (measured: a duct-fundamental C_D oscillation as large as the
        // physical one). Smooth turn-on excites it far less.
        return target * Float(0.5 * (1.0 - cos(Double.pi * x)))
    }

    private func params(step t: Int, writeForce: Bool = false) -> Params {
        let r = ramp(1.0, atStep: t)
        return Params(nx: UInt32(nx), ny: UInt32(ny), nz: UInt32(nz),
                      parity: UInt32(t & 1), omega: omega, omegaMinus: omegaMinus,
                      uin: uinTarget * r, cSmago: cSmago,
                      fx: force.x, fy: force.y, fz: force.z,
                      writeForce: writeForce ? 1 : 0,
                      ulidX: lidVel.x * r, ulidY: lidVel.y * r, ulidZ: lidVel.z * r,
                      inflowUniform: inflowUniform ? 1 : 0,
                      spongeX0: sponge?.x0 ?? 1e9,
                      spongeInvW: sponge.map { 1.0 / $0.width } ?? 0,
                      spongeTau: sponge?.tau ?? 1.0,
                      useEps: usesEps ? 1 : 0)
    }

    /// Steps (absolute indices) whose odd pass should accumulate forces.
    private var forceSteps: Set<Int> = []

    /// Run `count` steps. Chunked into command buffers small enough to stay
    /// under the GPU watchdog; up to two buffers in flight.
    public func run(steps count: Int) throws {
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
                throw StrouhalError.message("command buffer creation failed")
            }
            enc.setBuffer(fBuf, offset: 0, index: 0)
            enc.setBuffer(flagBuf, offset: 0, index: 1)
            enc.setBuffer(solidMaskBuf, offset: 0, index: 2)
            enc.setBuffer(lidMaskBuf, offset: 0, index: 3)
            enc.setBuffer(forceBuf, offset: 0, index: 5)
            enc.setBuffer(epsBuf, offset: 0, index: 6)
            for s in 0..<batch {
                var p = params(step: stepsDone + s,
                               writeForce: forceSteps.contains(stepsDone + s))
                enc.setComputePipelineState(pipes.step)
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
            throw StrouhalError.message("GPU command buffer failed (code \(code)) — watchdog or recovery event")
        }
    }

    /// (ux, uy, uz, rho) per cell. Only valid after an even number of steps.
    public func probeMoments() throws -> UnsafeBufferPointer<SIMD4<Float>> {
        precondition(stepsDone % 2 == 0, "moments probe requires even step count")
        guard let cb = gpu.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw StrouhalError.message("command buffer creation failed")
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

    /// Momentum-exchange force probe: runs one even+odd step pair with force
    /// accumulation on the odd pass, then sums per-cell contributions on the
    /// CPU (double, fixed order — deterministic) over cells inside the given
    /// bounding box. Requires wantsForces at init and an even step count.
    public func probeForce(xRange: ClosedRange<Int>, yRange: ClosedRange<Int>) throws -> SIMD3<Double> {
        precondition(stepsDone % 2 == 0, "force probe requires even step count")
        memset(forceBuf.contents(), 0, forceBuf.length)
        forceSteps = [stepsDone + 1]
        try run(steps: 2)
        forceSteps = []
        let ptr = forceBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: cells)
        var total = SIMD3<Double>.zero
        for z in 0..<nz { for y in yRange { for x in xRange {
            let v = ptr[(z * ny + y) * nx + x]
            total += SIMD3(Double(v.x), Double(v.y), Double(v.z))
        }}}
        return total
    }

    /// Encode `count` steps onto an existing compute encoder (render-loop
    /// path: no waiting, no chunking — caller owns the command buffer and
    /// must keep dispatches under the GPU watchdog budget).
    public func encode(steps count: Int, on enc: MTLComputeCommandEncoder) {
        let n = cells
        enc.setComputePipelineState(pipes.step)
        enc.setBuffer(fBuf, offset: 0, index: 0)
        enc.setBuffer(flagBuf, offset: 0, index: 1)
        enc.setBuffer(solidMaskBuf, offset: 0, index: 2)
        enc.setBuffer(lidMaskBuf, offset: 0, index: 3)
        enc.setBuffer(forceBuf, offset: 0, index: 5)
        enc.setBuffer(epsBuf, offset: 0, index: 6)
        for s in 0..<count {
            var p = params(step: stepsDone + s)
            enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        stepsDone += count
    }

    /// Encode the moments probe (valid when total steps will be even).
    public func encodeMoments(on enc: MTLComputeCommandEncoder) {
        enc.setComputePipelineState(pipes.moments)
        enc.setBuffer(fBuf, offset: 0, index: 0)
        enc.setBuffer(flagBuf, offset: 0, index: 1)
        enc.setBuffer(momentsBuf, offset: 0, index: 2)
        var p = params(step: stepsDone)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
        enc.dispatchThreads(MTLSize(width: cells, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    public var momentsBuffer: MTLBuffer { momentsBuf }
    public var flagsBuffer: MTLBuffer { flagBuf }
    public var epsBuffer: MTLBuffer { epsBuf }
    public var usesEpsField: Bool { usesEps }

    /// Install a Noble-Torczynski solid-fraction field (curved boundaries).
    public func setSolidFractions(_ eps: [Float]) throws {
        precondition(eps.count == cells)
        guard let buf = gpu.device.makeBuffer(bytes: eps, length: cells * 4,
                                              options: .storageModeShared) else {
            throw StrouhalError.message("eps buffer allocation failed")
        }
        epsBuf = buf
        usesEps = true
    }

    /// mode 1 = Taylor-Green (2D, one period per box), amplitude in lattice units.
    public func initField(mode: UInt32, amplitude: Float) throws {
        guard let cb = gpu.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw StrouhalError.message("command buffer creation failed")
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

    public var stateDigest: String {
        let data = Data(bytes: fBuf.contents(), count: fBuf.length)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Total shifted-density sum (double accumulation) — mass drift probe.
    public func massSum() -> Double {
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
