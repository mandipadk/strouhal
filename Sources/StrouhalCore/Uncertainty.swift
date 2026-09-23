import Foundation

// The truth layer: a per-QoI uncertainty budget, assembled from components we
// can actually estimate, with the parts nobody can estimate named as such.
//
//   U(φ) = k · sqrt( u_num² + u_stat² + u_Ma² )        k = 2 (95%)
//
// Deliberately NOT a Grid Convergence Index. Richardson extrapolation assumes
// monotone convergence at a fitted order; for scale-resolving (LES-like)
// simulations discretization error and subgrid-model error are coupled and
// non-monotone under refinement, so a fitted-order GCI is not valid and we
// refuse to print one. (Celik's LES_IQ is a resolution gauge, not a calibrated
// error; Klein 2005 varies grid AND model independently. See
// docs/research/benchmarks-and-uq.md.)
//
// The FDA nozzle round-robin is the cautionary datum: numerical uncertainty
// there was < 1% while the model was ~33% off experiment. Grid convergence is
// necessary and nowhere near sufficient — so the validation-domain flag below
// is a first-class part of the answer, not a footnote.

/// A time series of a scalar QoI sampled at (possibly uneven) step counts.
public struct QoISeries: Sendable {
    public var name: String
    public var samples: [(step: Double, value: Double)] = []

    public init(name: String) { self.name = name }

    public mutating func append(step: Double, value: Double) {
        samples.append((step, value))
    }

    public var values: [Double] { samples.map(\.value) }
    public var mean: Double {
        guard !samples.isEmpty else { return .nan }
        return values.reduce(0, +) / Double(samples.count)
    }

    /// Discard the first `fraction` of the record (startup transient).
    public func stationaryTail(discarding fraction: Double = 0.5) -> QoISeries {
        var out = QoISeries(name: name)
        let cut = Int(Double(samples.count) * fraction)
        out.samples = Array(samples.dropFirst(cut))
        return out
    }
}

/// Statistical uncertainty of a time-averaged QoI: non-overlapping batch
/// means. Batches must be long relative to the integral time scale, which we
/// verify by requiring the batch means to be effectively uncorrelated (lag-1
/// autocorrelation small) — and we report N_eff so the user sees how much
/// independent information the average actually rests on.
public struct StatUncertainty: Sendable {
    public let mean: Double
    public let halfWidth95: Double   // u_stat (already 1.96σ/√M)
    public let batches: Int
    public let lag1: Double          // batch-mean autocorrelation; want |ρ| ≲ 0.3
    /// The QoI does not vary over the sampling window (a steady wake, not a
    /// shedding one). Batch variance is then ~0, which makes the lag-1
    /// autocorrelation meaningless — without this flag a perfectly converged
    /// steady run was reported as "statistics weak".
    public let steady: Bool
    public var trustworthy: Bool { steady || (batches >= 6 && abs(lag1) <= 0.3) }

    /// Batch length must exceed the integral time scale or the batch means
    /// stay correlated and the CI is a lie (measured: 12 short batches on the
    /// vortex street gave lag-1 ρ = 0.61). We search DOWNWARD from `maxBatches`
    /// — fewer, longer batches — and take the largest batch count whose means
    /// are effectively uncorrelated (|ρ| ≤ 0.3): the tightest CI that is still
    /// honest. If none qualifies, we return the longest-batch attempt and mark
    /// it untrustworthy rather than quietly publishing it.
    public init(series: QoISeries, maxBatches: Int = 16, minBatches: Int = 6) {
        let v = series.values
        guard v.count >= minBatches * 4 else {
            mean = series.mean; halfWidth95 = .nan; batches = 0; lag1 = .nan
            steady = false
            return
        }
        func stats(_ M: Int) -> (mean: Double, half: Double, rho: Double) {
            let per = v.count / M
            var means: [Double] = []
            for b in 0..<M {
                let slice = v[(b * per)..<((b + 1) * per)]
                means.append(slice.reduce(0, +) / Double(slice.count))
            }
            let m = means.reduce(0, +) / Double(M)
            let varSum = means.reduce(0.0) { $0 + ($1 - m) * ($1 - m) }
            let sd = M > 1 ? (varSum / Double(M - 1)).squareRoot() : 0
            var num = 0.0, den = 0.0
            for i in means.indices {
                den += (means[i] - m) * (means[i] - m)
                if i > 0 { num += (means[i] - m) * (means[i - 1] - m) }
            }
            return (m, 1.96 * sd / Double(M).squareRoot(), den > 0 ? num / den : .nan)
        }
        // A flow that has genuinely gone steady: the whole record varies by a
        // negligible fraction of its own magnitude. Report that as steady
        // rather than running it through machinery meant for fluctuations.
        let whole = stats(1)
        let lo = v.min() ?? 0, hi = v.max() ?? 0
        let scale = max(abs(whole.mean), 1e-12)
        if (hi - lo) / scale < 1e-3 {
            steady = true
            batches = 1
            mean = whole.mean
            halfWidth95 = (hi - lo) / 2      // the full spread, honestly tiny
            lag1 = 0
            return
        }
        steady = false

        var chosen: (Int, Double, Double, Double)? = nil
        for M in stride(from: maxBatches, through: minBatches, by: -2) {
            let s = stats(M)
            if abs(s.rho) <= 0.3 { chosen = (M, s.mean, s.half, s.rho); break }
        }
        if let c = chosen {
            batches = c.0; mean = c.1; halfWidth95 = c.2; lag1 = c.3
        } else {
            let s = stats(minBatches)
            batches = minBatches; mean = s.mean; halfWidth95 = s.half; lag1 = s.rho
        }
    }
}

/// One rung of a resolution ladder.
public struct LadderRung: Sendable {
    public let cellsPerFeature: Int
    public let value: Double
    public let stat: Double   // u_stat of this rung (95% half-width)
    public init(cellsPerFeature: Int, value: Double, stat: Double) {
        self.cellsPerFeature = cellsPerFeature
        self.value = value
        self.stat = stat
    }
}

/// Numerical (discretization) uncertainty from a resolution ladder.
///
/// We report the *spread between rungs*, not a Richardson extrapolation. If
/// the finest two rungs agree within their own statistical noise, we say the
/// QoI is resolution-insensitive over the tested range — which is a claim we
/// can defend. We also report the observed order when three rungs are present
/// AND the sequence is monotone, as DIAGNOSTIC information only, explicitly
/// not as the basis of the error bar.
public struct NumUncertainty: Sendable {
    public let rungs: [LadderRung]
    public let uNum: Double            // the component that enters U(φ)
    public let observedOrder: Double?  // diagnostic only; nil unless monotone
    public let note: String

    public init(rungs sorted: [LadderRung]) {
        let rungs = sorted.sorted { $0.cellsPerFeature < $1.cellsPerFeature }
        self.rungs = rungs
        guard rungs.count >= 2 else {
            uNum = .nan
            observedOrder = nil
            note = "single resolution — u_num not estimated (run a ladder)"
            return
        }
        let fine = rungs[rungs.count - 1]
        let mid = rungs[rungs.count - 2]
        let delta = abs(fine.value - mid.value)
        // The finest-pair difference is the honest, assumption-free estimate
        // of what remains. Never smaller than the finest rung's own noise.
        uNum = max(delta, fine.stat.isNaN ? 0 : fine.stat)

        var order: Double? = nil
        if rungs.count >= 3 {
            let c = rungs[rungs.count - 3]
            let d1 = mid.value - c.value
            let d2 = fine.value - mid.value
            // monotone AND actually converging (|d2| < |d1|)
            if d1 * d2 > 0, abs(d2) < abs(d1), abs(d2) > 0 {
                let r = Double(fine.cellsPerFeature) / Double(mid.cellsPerFeature)
                if r > 1.05 { order = log(abs(d1 / d2)) / log(r) }
            }
        }
        observedOrder = order
        if delta <= (fine.stat.isNaN ? 0 : fine.stat) {
            note = "finest two rungs agree within statistical noise — resolution-insensitive over the tested range"
        } else {
            note = "u_num = |finest − next| (no Richardson/GCI: invalid for scale-resolving runs)"
        }
    }
}

/// Compressibility uncertainty: LBM is weakly compressible, error is O(Ma²).
/// Estimated by re-running the case at half the lattice velocity and taking
/// the shift in the QoI — an empirical bound, not a model.
public struct MachUncertainty: Sendable {
    public let baseline: Double
    public let halfMach: Double
    public let uMa: Double
    public init(baseline: Double, halfMach: Double) {
        self.baseline = baseline
        self.halfMach = halfMach
        // The residual at the baseline Mach is ~4/3 of the measured shift
        // (error ∝ Ma², so shift = e − e/4 = 3e/4 ⇒ e = 4/3 · shift).
        self.uMa = abs(baseline - halfMach) * 4.0 / 3.0
    }
}

/// Where this run sits relative to the cases the solver has actually been
/// validated against (Oberkampf & Roy's validation domain). Inside → the bar
/// is calibrated. Outside → say so; do not launder an extrapolation as a
/// measurement.
public struct ValidationDomain: Sendable {
    public struct Anchor: Sendable {
        public let name: String
        public let re: ClosedRange<Double>
        public let mach: ClosedRange<Double>
        public let cellsPerFeature: ClosedRange<Int>
        public let geometry: String
        public init(name: String, re: ClosedRange<Double>, mach: ClosedRange<Double>,
                    cellsPerFeature: ClosedRange<Int>, geometry: String) {
            self.name = name; self.re = re; self.mach = mach
            self.cellsPerFeature = cellsPerFeature; self.geometry = geometry
        }
    }

    /// The anchors are exactly the gates the CLI proves on every build.
    public static let anchors: [Anchor] = [
        .init(name: "Ghia lid-driven cavity (RMS 0.004 vs published)",
              re: 100...1000, mach: 0...0.18, cellsPerFeature: 128...256,
              geometry: "enclosed"),
        .init(name: "Schäfer–Turek DFG 2D-1 (C_D within 0.16%)",
              re: 20...20, mach: 0...0.14, cellsPerFeature: 40...64,
              geometry: "bluff body in channel"),
        .init(name: "Schäfer–Turek DFG 2D-2 (St 0.2996, max C_L 0.9998)",
              re: 100...100, mach: 0...0.14, cellsPerFeature: 40...80,
              geometry: "bluff body in channel"),
        .init(name: "Poiseuille (exact) / Taylor–Green (order 1.96)",
              re: 0...100, mach: 0...0.35, cellsPerFeature: 32...128,
              geometry: "periodic / channel"),
        // The sphere is anchored by the Schiller–Naumann correlation, which
        // is itself a ±5% fit to experiment — so a bar inside this domain is
        // calibrated no better than the correlation it rests on.
        .init(name: "Sphere drag vs Schiller–Naumann correlation (±5%)",
              re: 20...300, mach: 0...0.17, cellsPerFeature: 24...64,
              geometry: "bluff body in free stream"),
        // Taylor–Green Re=1600 anchors the 3D transitional regime, but it is
        // a periodic box with no body in it: it says nothing about drag on a
        // shape, so it deliberately does not serve as a free-stream anchor.
        .init(name: "Taylor–Green 3D Re=1600 vs Incompact3d 512³ DNS",
              re: 1600...1600, mach: 0...0.18, cellsPerFeature: 288...320,
              geometry: "periodic box"),
    ]

    public enum Verdict: Sendable {
        case inside(anchor: String)
        case nearEdge(anchor: String, factor: Double)  // inflate the bar by `factor`
        case outside(reasons: [String])
    }

    /// Classify a run. `geometry` should match an anchor's class for an
    /// "inside" verdict — a validated cylinder says nothing about a wing.
    public static func classify(re: Double, mach: Double, cellsPerFeature: Int,
                                geometry: String) -> Verdict {
        var best: (Anchor, Double)? = nil   // anchor, extrapolation distance
        for a in anchors where a.geometry == geometry {
            let dRe = distance(re, a.re)
            let dMa = distance(mach, a.mach)
            let dN = distance(Double(cellsPerFeature),
                              Double(a.cellsPerFeature.lowerBound)...Double(a.cellsPerFeature.upperBound))
            let d = max(dRe, max(dMa, dN))
            if best == nil || d < best!.1 { best = (a, d) }
        }
        guard let (anchor, d) = best else {
            return .outside(reasons: ["no validated case of geometry class “\(geometry)”"])
        }
        if d == 0 { return .inside(anchor: anchor.name) }
        if d <= 0.5 { return .nearEdge(anchor: anchor.name, factor: 1.0 + 2.0 * d) }
        var reasons: [String] = []
        if distance(re, anchor.re) > 0 {
            reasons.append(String(format: "Re %.0f outside validated %.0f–%.0f",
                                  re, anchor.re.lowerBound, anchor.re.upperBound))
        }
        if distance(mach, anchor.mach) > 0 {
            reasons.append(String(format: "Ma %.3f outside validated ≤%.2f", mach, anchor.mach.upperBound))
        }
        if distance(Double(cellsPerFeature),
                    Double(anchor.cellsPerFeature.lowerBound)...Double(anchor.cellsPerFeature.upperBound)) > 0 {
            reasons.append("\(cellsPerFeature) cells/feature outside validated \(anchor.cellsPerFeature.lowerBound)–\(anchor.cellsPerFeature.upperBound)")
        }
        return .outside(reasons: reasons.isEmpty ? ["outside the validated domain"] : reasons)
    }

    /// Fractional distance outside a range (0 when inside).
    private static func distance(_ x: Double, _ r: ClosedRange<Double>) -> Double {
        if r.contains(x) { return 0 }
        let span = max(r.upperBound - r.lowerBound, abs(r.upperBound) * 0.5, 1e-9)
        return x < r.lowerBound ? (r.lowerBound - x) / span : (x - r.upperBound) / span
    }
}

/// The assembled answer for one QoI.
public struct UncertaintyBudget: Sendable {
    public let qoi: String
    public let value: Double
    public let uNum: Double
    public let uStat: Double
    public let uMa: Double
    public let verdict: ValidationDomain.Verdict
    public let notes: [String]

    public var combined: Double {
        let parts = [uNum, uStat, uMa].filter { !$0.isNaN }
        let rss = parts.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
        switch verdict {
        case .inside: return 2.0 * rss
        case .nearEdge(_, let f): return 2.0 * rss * f
        case .outside: return .nan   // do not publish a calibrated bar
        }
    }

    public init(qoi: String, value: Double, uNum: Double, uStat: Double, uMa: Double,
                verdict: ValidationDomain.Verdict, notes: [String] = []) {
        self.qoi = qoi; self.value = value
        self.uNum = uNum; self.uStat = uStat; self.uMa = uMa
        self.verdict = verdict; self.notes = notes
    }

    /// One-line headline, honest about the outside-domain case.
    public var headline: String {
        switch verdict {
        case .outside:
            return String(format: "%@ = %.4f — OUTSIDE the validated domain: no calibrated uncertainty", qoi, value)
        default:
            return String(format: "%@ = %.4f ± %.4f (k=2)", qoi, value, combined)
        }
    }
}
