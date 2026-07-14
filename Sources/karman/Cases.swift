import Foundation

// MARK: - Ghia, Ghia & Shin (1982) oracle — interior points only (boundary
// rows are identically satisfied by the BCs). Columns: coordinate, Re=100,
// Re=400, Re=1000. See docs/research/benchmarks-and-uq.md for provenance
// and known transcription caveats.

let ghiaU: [(y: Double, re100: Double, re400: Double, re1000: Double)] = [
    (0.9766,  0.84123,  0.75837,  0.65928),
    (0.9688,  0.78871,  0.68439,  0.57492),
    (0.9609,  0.73722,  0.61756,  0.51117),
    (0.9531,  0.68717,  0.55892,  0.46604),
    (0.8516,  0.23151,  0.29093,  0.33304),
    (0.7344,  0.00332,  0.16256,  0.18719),
    (0.6172, -0.13641,  0.02135,  0.05702),
    (0.5000, -0.20581, -0.11477, -0.06080),
    (0.4531, -0.21090, -0.17119, -0.10648),
    (0.2813, -0.15662, -0.32726, -0.27805),
    (0.1719, -0.10150, -0.24299, -0.38289),
    (0.1016, -0.06434, -0.14612, -0.29730),
    (0.0703, -0.04775, -0.10338, -0.22220),
    (0.0625, -0.04192, -0.09266, -0.20196),
    (0.0547, -0.03717, -0.08186, -0.18109),
]

let ghiaV: [(x: Double, re100: Double, re400: Double, re1000: Double)] = [
    (0.9688, -0.05906, -0.12146, -0.21388),
    (0.9609, -0.07391, -0.15663, -0.27669),
    (0.9531, -0.08864, -0.19254, -0.33714),
    (0.9453, -0.10313, -0.22847, -0.39188),
    (0.9063, -0.16914, -0.23827, -0.51550),
    (0.8594, -0.22445, -0.44993, -0.42665),
    (0.8047, -0.24533, -0.38598, -0.31966),
    (0.5000,  0.05454,  0.05186,  0.02526),
    (0.2344,  0.17527,  0.30174,  0.32235),
    (0.2266,  0.17507,  0.30203,  0.33075),
    (0.1563,  0.16077,  0.28124,  0.37095),
    (0.0938,  0.12317,  0.22965,  0.32627),
    (0.0781,  0.10890,  0.20920,  0.30353),
    (0.0703,  0.10091,  0.19713,  0.29012),
    (0.0625,  0.09233,  0.18360,  0.27485),
]

struct GateResult {
    let name: String
    let passed: Bool
    let detail: String
}

// MARK: - Self-tests (the parity/indexing proof)

func runSelftest(gpu: GPU) throws -> [GateResult] {
    var results: [GateResult] = []

    // 1. Rest state is a bitwise fixed point (periodic box, active collision).
    do {
        let sim = try Simulation(gpu: gpu, nx: 32, ny: 32, nz: 32, omega: 1.7) { _, _, _ in .fluid }
        let before = sim.stateDigest
        try sim.run(steps: 100)
        let after = sim.stateDigest
        results.append(GateResult(name: "rest fixed point (periodic)",
                                  passed: before == after,
                                  detail: before == after ? "bitwise stable over 100 steps" : "state changed"))
    }

    // 2. Rest cavity (walls + stationary lid) is also a bitwise fixed point.
    do {
        let n = 34
        let sim = try Simulation(gpu: gpu, nx: n, ny: n, nz: 1, omega: 1.7, ulid: 0) { x, y, _ in
            if y == n - 1 { return .lid }
            if x == 0 || x == n - 1 || y == 0 { return .solid }
            return .fluid
        }
        let before = sim.stateDigest
        try sim.run(steps: 100)
        let passed = sim.stateDigest == before
        results.append(GateResult(name: "rest fixed point (cavity walls)",
                                  passed: passed,
                                  detail: passed ? "bounce-back of zeros is zeros" : "wall handling perturbs rest state"))
    }

    // 3. Streaming: with omega = 0, a lone DDF in direction i must travel
    //    exactly T*c_i and remain bit-identical (verifies every slot in the
    //    AA parity scheme, all 18 directions at once).
    do {
        let n = 16
        let sim = try Simulation(gpu: gpu, nx: n, ny: n, nz: n, omega: 0) { _, _, _ in .fluid }
        let N = sim.cells
        let f = sim.fBuf.contents().bindMemory(to: Float.self, capacity: 19 * N)
        let c0 = (8, 8, 8)
        let cellIndex = { (x: Int, y: Int, z: Int) in (z * n + y) * n + x }
        var injected: [Float] = Array(repeating: 0, count: 19)
        for i in 1..<19 {
            injected[i] = Float(i) * 0.001
            f[i * N + cellIndex(c0.0, c0.1, c0.2)] = injected[i]
        }
        let T = 4
        try sim.run(steps: T)
        var ok = true
        var firstFailure = ""
        var nonzero = 0
        for i in 1..<19 {
            let dest = cellIndex((c0.0 + T * Simulation.cx[i] + 4 * n) % n,
                                 (c0.1 + T * Simulation.cy[i] + 4 * n) % n,
                                 (c0.2 + T * Simulation.cz[i] + 4 * n) % n)
            let v = f[i * N + dest]
            if v != injected[i] {
                ok = false
                if firstFailure.isEmpty {
                    firstFailure = "dir \(i): expected \(injected[i]) at dest, found \(v)"
                }
            }
        }
        for k in 0..<(19 * N) where f[k] != 0 { nonzero += 1 }
        if nonzero != 18 {
            ok = false
            if firstFailure.isEmpty { firstFailure = "\(nonzero) nonzero slots (expected 18) — leakage" }
        }
        results.append(GateResult(name: "streaming propagation (18 dirs, \(T) steps)",
                                  passed: ok,
                                  detail: ok ? "each DDF at exactly cell + \(T)·c_i, bit-identical, no leakage" : firstFailure))
    }

    return results
}

// MARK: - Benchmark

func runBench(gpu: GPU, n: Int, precision: Precision,
              warmup: Int = 20, timed: Int = 200) throws -> GateResult {
    let sim = try Simulation(gpu: gpu, precision: precision, nx: n, ny: n, nz: n,
                             omega: 1.9) { _, _, _ in .fluid }
    try sim.initField(mode: 1, amplitude: 0.05)
    try sim.run(steps: warmup)
    let t0 = sim.gpuSeconds
    try sim.run(steps: timed)
    let dt = sim.gpuSeconds - t0
    let mlups = Double(sim.cells) * Double(timed) / dt / 1e6
    // Bytes/cell/step: DDFs 19*2*ddfBytes + 1 flag + masks (8, odd steps only -> avg 4).
    let bytes = Double(19 * 2 * precision.ddfBytes + 1) + 4.0
    let gbps = mlups * bytes / 1000.0
    let gate = precision == .fp32 ? 600.0 : 1200.0
    let ref = precision == .fp32 ? "FluidX3D-OpenCL M5: 800 FP32" : "FluidX3D-OpenCL M5: 1613 FP16C"
    let passed = mlups >= gate
    return GateResult(name: "bench \(n)³ (\(precision.rawValue))",
                      passed: passed,
                      detail: String(format: "%.0f MLUPS, ~%.0f GB/s effective (gate ≥%.0f; %@)", mlups, gbps, gate, ref))
}

// MARK: - Cavity

struct CavityRun {
    let sim: Simulation
    let nInterior: Int
    let converged: Bool
    let steps: Int
    let residual: Double
    let howConverged: String
}

/// Lid-driven cavity: interior n×n fluid cells + 1-cell solid frame, lid on
/// top (+x). Effective cavity width with halfway bounce-back = n exactly.
/// Collision: TRT with the given magic parameter (nil = SRT).
func cavity(gpu: GPU, precision: Precision = .fp32, n: Int, re: Double,
            lambda: Double? = 0.25, ulid: Float = 0.1,
            maxSteps: Int, checkEvery: Int = 5000, tol: Double = 5e-7) throws -> CavityRun {
    let nx = n + 2, ny = n + 2
    let nu = Double(ulid) * Double(n) / re
    let tau = 3.0 * nu + 0.5
    let (wp, wm) = Simulation.trtOmegas(tau: tau, lambda: lambda)
    let sim = try Simulation(gpu: gpu, precision: precision, nx: nx, ny: ny, nz: 1,
                             omega: wp, omegaMinus: wm, ulid: ulid, rampSteps: 5000) { x, y, _ in
        if y == ny - 1 { return .lid }
        if x == 0 || x == nx - 1 || y == 0 { return .solid }
        return .fluid
    }
    var prev: [Float] = []
    var converged = false
    var residual = Double.infinity
    var history: [Double] = []
    var how = "max steps reached"
    while sim.stepsDone < maxSteps {
        try sim.run(steps: min(checkEvery, maxSteps - sim.stepsDone))
        let m = try sim.probeMoments()
        var cur = [Float](); cur.reserveCapacity(sim.cells * 2)
        for v in m { cur.append(v.x); cur.append(v.y) }
        if !prev.isEmpty && sim.stepsDone > sim.rampSteps {
            var dsum = 0.0, nsum = 0.0
            for k in 0..<cur.count {
                let d = Double(cur[k] - prev[k])
                dsum += d * d
                nsum += Double(cur[k]) * Double(cur[k])
            }
            residual = nsum > 0 ? (dsum / nsum).squareRoot() : 0
            history.append(residual)
            if residual < tol {
                converged = true; how = "residual < tol"
            } else if residual < (precision == .fp32 ? 5e-5 : 2e-3), history.count >= 5,
                      let best = history.dropLast().suffix(4).min(),
                      residual > 0.95 * best {
                // Round-off floor: the field has stopped improving at a low
                // level (FP32/FP16 noise + lid mass-drift micro-jitter).
                converged = true; how = "residual floor (plateau)"
            }
        }
        prev = cur
        if converged { break }
    }
    return CavityRun(sim: sim, nInterior: n, converged: converged,
                     steps: sim.stepsDone, residual: residual, howConverged: how)
}

/// Compare centerline profiles against Ghia. Interior fluid nodes are at
/// physical y = (iy - 0.5)/n for iy = 1...n (array row iy). x = 0.5 lies
/// exactly between columns n/2 and n/2+1 — average them.
func ghiaComparison(run: CavityRun, re: Double, verbose: Bool = true) throws -> GateResult {
    let sim = run.sim
    let n = run.nInterior
    let m = try sim.probeMoments()
    let nx = sim.nx
    let ulid = Double(sim.ulidTarget)

    func u(atRow iy: Int) -> Double {
        let a = m[iy * nx + n / 2].x
        let b = m[iy * nx + n / 2 + 1].x
        return Double(a + b) / 2.0 / ulid
    }
    func v(atCol ix: Int) -> Double {
        let a = m[(n / 2) * nx + ix].y
        let b = m[(n / 2 + 1) * nx + ix].y
        return Double(a + b) / 2.0 / ulid
    }
    func interp(_ coord: Double, _ value: (Int) -> Double) -> Double {
        let s = coord * Double(n) + 0.5
        let k0 = min(max(Int(s.rounded(.down)), 1), n - 1)
        let frac = s - Double(k0)
        return value(k0) * (1 - frac) + value(k0 + 1) * frac
    }
    func oracle(_ r100: Double, _ r400: Double, _ r1000: Double) -> Double {
        switch re {
        case 100: return r100
        case 400: return r400
        default: return r1000
        }
    }

    var sumSq = 0.0
    var maxDev = 0.0
    var count = 0
    var lines: [String] = []
    for row in ghiaU {
        let ours = interp(row.y, u(atRow:))
        let ref = oracle(row.re100, row.re400, row.re1000)
        let d = ours - ref
        sumSq += d * d; maxDev = max(maxDev, abs(d)); count += 1
        lines.append(String(format: "  u(y=%.4f): karman %+.5f  ghia %+.5f  Δ %+.5f", row.y, ours, ref, d))
    }
    for row in ghiaV {
        let ours = interp(row.x, v(atCol:))
        let ref = oracle(row.re100, row.re400, row.re1000)
        let d = ours - ref
        sumSq += d * d; maxDev = max(maxDev, abs(d)); count += 1
        lines.append(String(format: "  v(x=%.4f): karman %+.5f  ghia %+.5f  Δ %+.5f", row.x, ours, ref, d))
    }
    let rms = (sumSq / Double(count)).squareRoot()
    let passed = rms <= 0.02
    var detail = String(format: "RMS %.4f (gate ≤0.02), max |Δ| %.4f, %@ after %d steps (residual %.1e)",
                        rms, maxDev,
                        run.converged ? "converged (\(run.howConverged))" : "NOT converged",
                        run.steps, run.residual)
    if verbose { detail += "\n" + lines.joined(separator: "\n") }
    return GateResult(name: String(format: "cavity Re=%.0f vs Ghia (%d², %@)", re, n, run.sim.precision.rawValue),
                      passed: passed && run.converged,
                      detail: detail)
}

// MARK: - Poiseuille (exact-solution gate)

/// Body-force-driven channel flow. With TRT and Lambda = 3/16 the halfway
/// bounce-back wall location is viscosity-exact, so the discrete steady
/// profile should match the parabola to round-off.
func runPoiseuille(gpu: GPU, height H: Int = 64) throws -> GateResult {
    let tau = 0.8
    let nu = (tau - 0.5) / 3.0
    let uMax: Double = 0.05
    let F = 8.0 * nu * uMax / Double(H * H)
    let (wp, wm) = Simulation.trtOmegas(tau: tau, lambda: 3.0 / 16.0)
    let ny = H + 2
    let sim = try Simulation(gpu: gpu, nx: 16, ny: ny, nz: 1,
                             omega: wp, omegaMinus: wm,
                             force: SIMD3(Float(F), 0, 0)) { _, y, _ in
        (y == 0 || y == ny - 1) ? .solid : .fluid
    }
    // Diffusive time H^2/nu; run several to reach steady state.
    let tVisc = Double(H * H) / nu
    try sim.run(steps: (Int(6.0 * tVisc) + 1) & ~1)
    let m = try sim.probeMoments()
    var maxErr = 0.0
    var errWall = 0.0, errCenter = 0.0
    for j in 1...H {
        let yd = Double(j) - 0.5 // distance from bottom wall plane
        let exact = F / (2.0 * nu) * yd * (Double(H) - yd)
        let ours = Double(m[j * sim.nx + 8].x)
        let e = abs(ours - exact) / uMax
        maxErr = max(maxErr, e)
        if j == 1 || j == H { errWall = max(errWall, e) }
        if j == H / 2 || j == H / 2 + 1 { errCenter = max(errCenter, e) }
    }
    let passed = maxErr <= 3e-5
    return GateResult(name: "Poiseuille exact (TRT Λ=3/16, H=\(H))",
                      passed: passed,
                      detail: String(format: "max |u-u_exact|/u_max = %.2e (gate ≤3e-5, FP32 accumulation floor); wall %.2e, center %.2e", maxErr, errWall, errCenter))
}

// MARK: - Taylor-Green (order-of-accuracy gate)

/// 2D Taylor-Green decay under diffusive scaling (u0 ∝ 1/N, ν fixed):
/// both the spatial truncation error and the O(Ma²) compressibility error
/// scale as 1/N², so the observed convergence order should be ≈ 2.
/// Amplitude note: the default u0base = 0.2 runs the coarse grid at Ma≈0.35
/// deliberately — the errors are large but their SCALING is the measurand;
/// smaller amplitudes push the fine grid into the FP32 round-off floor and
/// the measured order collapses (verified: u0base 0.05 reads 1.76 for
/// exactly this reason).
func runTaylorGreenOrder(gpu: GPU, sizes: [Int] = [32, 64, 128], u0base: Double = 0.20) throws -> GateResult {
    let nu = 0.02
    let tau = 3.0 * nu + 0.5
    let (wp, wm) = Simulation.trtOmegas(tau: tau, lambda: 0.25)
    var errors: [Double] = []
    var details: [String] = []
    for N in sizes {
        // Base amplitude sized so the finest grid's error stays well above
        // the FP32 round-off floor (the N=256/u0=0.01 configuration hit it).
        let u0 = u0base * 32.0 / Double(N)
        let k = 2.0 * Double.pi / Double(N)
        let steps = Int(log(2.0) / (2.0 * nu * k * k)) & ~1 // decay to ~1/2 amplitude
        let sim = try Simulation(gpu: gpu, nx: N, ny: N, nz: 1,
                                 omega: wp, omegaMinus: wm) { _, _, _ in .fluid }
        try sim.initField(mode: 1, amplitude: Float(u0))
        try sim.run(steps: steps)
        let m = try sim.probeMoments()
        let decay = exp(-2.0 * nu * k * k * Double(steps))
        var sumSq = 0.0
        for y in 0..<N { for x in 0..<N {
            let xa = Double(x) + 0.5, ya = Double(y) + 0.5
            let ue =  u0 * decay * sin(k * xa) * cos(k * ya)
            let ve = -u0 * decay * cos(k * xa) * sin(k * ya)
            let v = m[y * N + x]
            let du = Double(v.x) - ue, dv = Double(v.y) - ve
            sumSq += du * du + dv * dv
        }}
        let l2 = (sumSq / Double(2 * N * N)).squareRoot() / (u0 * decay)
        errors.append(l2)
        details.append(String(format: "N=%d: rel L2 %.3e (%d steps)", N, l2, steps))
    }
    var orders: [Double] = []
    for i in 1..<errors.count {
        orders.append(log2(errors[i - 1] / errors[i]))
    }
    let minOrder = orders.min() ?? 0
    let passed = minOrder >= 1.9
    return GateResult(name: "Taylor-Green observed order",
                      passed: passed,
                      detail: details.joined(separator: "; ") + String(format: "; orders: %@ (gate: min ≥1.9)",
                          orders.map { String(format: "%.2f", $0) }.joined(separator: ", ")))
}

// MARK: - Determinism

func runDeterminism(gpu: GPU, precision: Precision = .fp32) throws -> GateResult {
    func cavityDigest() throws -> String {
        let run = try cavity(gpu: gpu, precision: precision, n: 128, re: 1000,
                             maxSteps: 10000, checkEvery: 10000, tol: 0)
        return run.sim.stateDigest
    }
    let a = try cavityDigest()
    let b = try cavityDigest()

    func benchDigest() throws -> String {
        let sim = try Simulation(gpu: gpu, precision: precision,
                                 nx: 128, ny: 128, nz: 128, omega: 1.9) { _, _, _ in .fluid }
        try sim.initField(mode: 1, amplitude: 0.05)
        try sim.run(steps: 100)
        return sim.stateDigest
    }
    let c = try benchDigest()
    let d = try benchDigest()

    let passed = a == b && c == d
    return GateResult(name: "bitwise determinism (run-twice, \(precision.rawValue))",
                      passed: passed,
                      detail: passed
                        ? "cavity 128² ×10k steps and 128³ TG ×100 steps: digests identical (\(a.prefix(16))…)"
                        : "DIGEST MISMATCH — cavity: \(a.prefix(16)) vs \(b.prefix(16)); bench: \(c.prefix(16)) vs \(d.prefix(16))")
}
