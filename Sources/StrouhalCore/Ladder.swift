import Foundation

/// Runs the extra simulations an honest error bar needs — a resolution ladder
/// plus a half-Mach anchor — and assembles the budget. This is the thing the
/// incumbents price as a paid add-on and practitioners therefore skip: on this
/// hardware it is minutes, so it is the default, not an upsell.
public struct CredibilityRun: Sendable {
    public let qoi: String
    public let budget: UncertaintyBudget
    public let rungs: [LadderRung]
    public let numNote: String
    public let observedOrder: Double?
    public let statBatches: Int
    public let statLag1: Double
    public let machBaseline: Double
    public let machHalf: Double
    public let wallSeconds: Double
    public let digest: String
    public let setup: String
    /// Whether every rung's QoI stopped moving before the transient cap.
    public var settleNote: String = ""
    /// Comparison against a published value, when one exists: the reference,
    /// its source, and whether U(φ) covers the gap.
    public var comparison: (reference: Double, source: String, covered: Bool)? = nil
}

/// A case the ladder can rebuild at an arbitrary resolution and Mach number.
/// (The vortex street is the reference implementation; STL bodies conform too.)
public protocol LadderableCase: Sendable {
    /// Human-readable setup, for the report.
    var setupDescription: String { get }
    /// Reynolds number, Mach number, geometry class — for the validation domain.
    var re: Double { get }
    var geometryClass: String { get }
    /// Build a variant at `cellsPerFeature` resolution and `machScale` × the
    /// baseline lattice velocity; return (sim, a probe for the QoI, steps to
    /// discard as transient, steps to sample, sampling stride).
    func variant(gpu: GPU, cellsPerFeature: Int, machScale: Double) throws -> LadderVariant
    /// A published value this QoI can be compared against, if one exists.
    /// ASME V&V 20 calls the gap the comparison error, E = S − D. Reporting
    /// it is the difference between "our uncertainty is small" and "our
    /// answer is right" — the FDA nozzle round-robin had the first without
    /// the second.
    var reference: (value: Double, source: String)? { get }
}

public extension LadderableCase {
    var reference: (value: Double, source: String)? { nil }
}

public struct LadderVariant {
    public let sim: Simulation
    public let mach: Double
    public let probe: (Simulation) throws -> Double
    public let transientSteps: Int
    public let sampleSteps: Int
    public let stride: Int
    /// Optional adaptive settling. When set, the transient runs until the QoI
    /// stops moving — peak-to-peak over `settleWindow` samples below this
    /// fraction of the mean — instead of for a fixed number of steps, and
    /// gives up at `maxTransientSteps`.
    ///
    /// This exists because a fixed transient is a guess, and a wrong guess is
    /// silent: the sphere's drag was MEASURED oscillating from 3.13 down to
    /// 0.97, up to 1.58, and only flattening at 1.3653 after ~150 convective
    /// times. Averaging inside that ramp produced a plausible-looking value
    /// with a ±14% "statistical" bar that was really an unconverged drift.
    /// A shedding case never flattens, so it runs to the cap and is averaged.
    public let settleTolerance: Double?
    public let settleWindow: Int
    public let maxTransientSteps: Int
    /// Probe cadence during settling. Coarser than `stride`, because the
    /// settling window must span a long physical time while the sampling
    /// window must contain enough points for batch means.
    public let settleStride: Int

    public init(sim: Simulation, mach: Double,
                probe: @escaping (Simulation) throws -> Double,
                transientSteps: Int, sampleSteps: Int, stride: Int,
                settleTolerance: Double? = nil, settleWindow: Int = 10,
                maxTransientSteps: Int = 0, settleStride: Int = 0) {
        self.sim = sim; self.mach = mach; self.probe = probe
        self.transientSteps = transientSteps; self.sampleSteps = sampleSteps
        self.stride = stride
        self.settleTolerance = settleTolerance
        self.settleWindow = settleWindow
        self.maxTransientSteps = maxTransientSteps > 0 ? maxTransientSteps : transientSteps
        self.settleStride = settleStride > 0 ? settleStride : stride
    }
}

public enum Ladder {
    /// Run one variant and return its time-averaged QoI with statistics.
    @discardableResult
    public static func measure(_ v: LadderVariant, qoi name: String) throws
        -> (series: QoISeries, stat: StatUncertainty, digest: String, converged: Bool) {
        var converged = true
        if let tol = v.settleTolerance {
            converged = try settle(v, tolerance: tol)
        } else {
            try v.sim.run(steps: v.transientSteps & ~1)
        }
        var series = QoISeries(name: name)
        var taken = 0
        while taken < v.sampleSteps {
            let chunk = max(2, v.stride - 2) & ~1
            try v.sim.run(steps: chunk)
            let value = try v.probe(v.sim)   // probe advances 2 steps
            taken += chunk + 2
            series.append(step: Double(v.sim.stepsDone), value: value)
        }
        // Discard the first half of the sampled record: even after the nominal
        // transient, slow drifts remain, and a batch-means CI over a drifting
        // record understates the true uncertainty.
        let tail = series.stationaryTail(discarding: 0.5)
        return (tail, StatUncertainty(series: tail), v.sim.stateDigest, converged)
    }

    /// Run the transient until the QoI stops moving, or until the cap.
    /// Returns having consumed at least `transientSteps`.
    /// - Returns: true if the QoI genuinely stopped moving, false if the run
    ///   hit the cap first — in which case the value is NOT converged and the
    ///   report must say so rather than presenting it as settled.
    private static func settle(_ v: LadderVariant, tolerance: Double) throws -> Bool {
        try v.sim.run(steps: v.transientSteps & ~1)
        var window: [Double] = []
        let chunk = max(2, v.settleStride - 2) & ~1
        while v.sim.stepsDone < v.maxTransientSteps {
            try v.sim.run(steps: chunk)
            window.append(try v.probe(v.sim))
            if window.count > v.settleWindow { window.removeFirst() }
            guard window.count == v.settleWindow else { continue }
            let lo = window.min()!, hi = window.max()!
            let mean = window.reduce(0, +) / Double(window.count)
            if abs(mean) > 1e-12, (hi - lo) / abs(mean) < tolerance { return true }
        }
        return false
    }

    /// The full credibility run: N resolutions + a half-Mach anchor.
    /// - Parameter machAnchorResolution: which rung to re-run at half Mach.
    ///   Defaults to the finest. 3D cases use a coarser rung because the
    ///   compressibility shift was MEASURED to be resolution-insensitive
    ///   (Taylor–Green Re=1600: peak 0.011888 at Ma 0.130 vs 0.011887 at
    ///   Ma 0.173, a difference of one part in 10⁴), and anchoring on the
    ///   finest 3D rung would roughly double an already long run.
    public static func credibility(gpu: GPU, case c: LadderableCase, qoi name: String,
                                   resolutions: [Int],
                                   machAnchorResolution: Int? = nil,
                                   progress: ((String) -> Void)? = nil) throws -> CredibilityRun {
        let t0 = Date()
        var rungs: [LadderRung] = []
        var unconverged: [Int] = []
        var finestStat: StatUncertainty? = nil
        var finestDigest = ""
        var finestMach = 0.0

        for n in resolutions.sorted() {
            progress?("resolution \(n) cells/feature…")
            let v = try c.variant(gpu: gpu, cellsPerFeature: n, machScale: 1.0)
            let (series, stat, digest, converged) = try measure(v, qoi: name)
            rungs.append(LadderRung(cellsPerFeature: n, value: series.mean,
                                    stat: stat.halfWidth95))
            if !converged { unconverged.append(n) }
            if n == resolutions.max() {
                finestStat = stat
                finestDigest = digest
                finestMach = v.mach
            }
        }

        let finest = resolutions.max()!
        let anchorAt = machAnchorResolution ?? finest
        progress?("half-Mach anchor at \(anchorAt) cells…")
        let vHalf = try c.variant(gpu: gpu, cellsPerFeature: anchorAt, machScale: 0.5)
        let (halfSeries, _, _, _) = try measure(vHalf, qoi: name)
        // Compare like with like: the shift must be measured against the same
        // rung it was run at, not against the finest.
        let machBase = rungs.first { $0.cellsPerFeature == anchorAt }?.value ?? rungs.last!.value

        let num = NumUncertainty(rungs: rungs)
        let stat = finestStat!
        let mach = MachUncertainty(baseline: machBase, halfMach: halfSeries.mean)
        let verdict = ValidationDomain.classify(re: c.re, mach: finestMach,
                                                cellsPerFeature: finest,
                                                geometry: c.geometryClass)
        var notes: [String] = [num.note]
        if stat.steady {
            notes.append("QoI is steady over the sampling window — statistical uncertainty is negligible, not unmeasured")
        } else if !stat.trustworthy {
            notes.append(String(format: "statistics weak: %d batches, lag-1 ρ = %.2f — average over more shedding cycles",
                                stat.batches, stat.lag1))
        }
        if !unconverged.isEmpty {
            notes.append("rungs \(unconverged.map(String.init).joined(separator: ", ")) hit the transient cap before the QoI stopped moving — those values are not converged")
        }
        if let l = c as? BodyLadder, !l.anchored {
            notes.append("no validated anchor for this geometry: the components below are measured, but that the solver reproduces reality for THIS shape is not established")
        }
        let budget = UncertaintyBudget(qoi: name, value: rungs.last!.value,
                                       uNum: num.uNum, uStat: stat.halfWidth95,
                                       uMa: mach.uMa, verdict: verdict, notes: notes)
        return CredibilityRun(qoi: name, budget: budget, rungs: rungs,
                              numNote: num.note, observedOrder: num.observedOrder,
                              statBatches: stat.batches, statLag1: stat.lag1,
                              machBaseline: mach.baseline, machHalf: mach.halfMach,
                              wallSeconds: -t0.timeIntervalSinceNow,
                              digest: finestDigest, setup: c.setupDescription,
                              settleNote: unconverged.isEmpty
                                ? "every rung settled before the transient cap"
                                : "cap reached at \(unconverged.count) of \(rungs.count) rungs",
                              comparison: c.reference.map { ref in
                                  (reference: ref.value, source: ref.source,
                                   covered: abs(budget.value - ref.value) <= budget.combined)
                              })
    }
}

// MARK: - Report

public enum CredibilityReport {
    /// Markdown, in ASME V&V 20 vocabulary (the standard the FDA recognizes),
    /// with the parts we cannot estimate named as such.
    public static func markdown(_ r: CredibilityRun, buildGates: [String]) -> String {
        var s = "# Credibility report — \(r.qoi)\n\n"
        s += "_Generated by Strouhal · \(ISO8601DateFormatter().string(from: Date()))_\n\n"
        s += "## Result\n\n"
        s += "**\(r.budget.headline)**\n\n"
        s += "\(r.setup)\n\n"

        s += "## Uncertainty budget\n\n"
        s += "U(φ) = k·√(u_num² + u_stat² + u_Ma²), k = 2 (≈95%)\n\n"
        s += "| Component | Value | Basis |\n|---|---|---|\n"
        s += String(format: "| u_num (discretization) | %.4f | %@ |\n", r.budget.uNum, r.numNote)
        s += String(format: "| u_stat (time average) | %.4f | non-overlapping batch means, %d batches, lag-1 ρ = %.2f |\n",
                    r.budget.uStat, r.statBatches, r.statLag1)
        s += String(format: "| u_Ma (compressibility) | %.4f | half-Mach anchor: %.4f → %.4f, O(Ma²) extrapolation |\n",
                    r.budget.uMa, r.machBaseline, r.machHalf)
        switch r.budget.verdict {
        case .inside(let a):
            s += String(format: "| **U(φ) (k=2)** | **%.4f** | inside the validated domain (%@) |\n",
                        r.budget.combined, a)
        case .nearEdge(let a, let f):
            s += String(format: "| **U(φ) (k=2)** | **%.4f** | near the edge of the validated domain (%@); inflated ×%.2f |\n",
                        r.budget.combined, a, f)
        case .outside:
            s += "| **U(φ)** | **not published** | outside the validated domain — see below |\n"
        }
        s += "\n"

        s += "## Resolution ladder\n\n| cells/feature | φ | u_stat |\n|---|---|---|\n"
        for rung in r.rungs {
            s += String(format: "| %d | %.4f | %.4f |\n", rung.cellsPerFeature, rung.value, rung.stat)
        }
        if let o = r.observedOrder {
            s += String(format: "\nObserved order of convergence: **%.2f** — reported as a diagnostic only. ", o)
            s += "It is **not** used to build the error bar: Richardson extrapolation assumes monotone convergence at a fitted order, which scale-resolving simulations do not guarantee (discretization and subgrid-model error are coupled). No GCI is claimed.\n"
        } else {
            s += "\nNo observed order reported: the ladder is not monotone-convergent, so a fitted order would be meaningless.\n"
        }
        s += "\n"

        s += "## Validation domain\n\n"
        switch r.budget.verdict {
        case .inside(let a):
            s += "This run lies **inside** the domain validated by: \(a).\n"
        case .nearEdge(let a, let f):
            s += String(format: "This run lies **near the edge** of the domain validated by: %@. The uncertainty is inflated by ×%.2f with extrapolation distance, per Oberkampf & Roy.\n", a, f)
        case .outside(let reasons):
            s += "**This run lies OUTSIDE the validated domain.** No calibrated uncertainty is published, because none is defensible.\n\n"
            for reason in reasons { s += "- \(reason)\n" }
            s += "\nThe numbers above are the solver's output; they are not a measurement with a warranty.\n"
        }
        s += "\n"

        if !r.budget.notes.isEmpty {
            s += "## Notes\n\n"
            for n in r.budget.notes where !n.isEmpty { s += "- \(n)\n" }
            s += "\n"
        }

        s += "## Solver verification (this build)\n\n"
        for g in buildGates { s += "- \(g)\n" }
        s += "\n"

        if let cmp = r.comparison {
            let e = r.budget.value - cmp.reference
            s += "## Comparison with the reference (ASME V&V 20: E = S − D)\n\n"
            s += String(format: "| Quantity | Value |\n|---|---|\n")
            s += String(format: "| Simulation, S | %.4f |\n", r.budget.value)
            s += String(format: "| Reference, D | %.4f (%@) |\n", cmp.reference, cmp.source)
            s += String(format: "| Comparison error, E = S − D | %+.4f (%+.1f%%) |\n",
                        e, e / cmp.reference * 100)
            s += String(format: "| U(φ) covers E? | %@ |\n\n", cmp.covered ? "yes" : "**no**")
            if cmp.covered {
                s += "The uncertainty covers the gap to the reference: within the resolution of this comparison, the solver reproduces the published value.\n\n"
            } else {
                s += "**The uncertainty does not cover the gap.** The numerical uncertainty quantified above is therefore not the dominant error in this result: something outside it — the model, the boundary conditions, or the setup — accounts for the difference. A small U(φ) is not a claim of correctness, and this is precisely the case that makes the distinction concrete.\n\n"
            }
        }

        s += "## Convergence of the transient\n\n"
        s += "Each rung ran until its quantity stopped changing (peak-to-peak below 0.5% over a window spanning more than one start-up oscillation), not for a fixed number of steps. "
        s += "\(r.settleNote).\n\n"

        s += "## Reproducibility\n\n"
        s += "- Final state digest (SHA-256): `\(r.digest)`\n"
        s += String(format: "- Total wall time for the full credibility run: %.1f s\n", r.wallSeconds)
        s += "- Same machine, same build, same settings ⇒ bitwise-identical state and identical QoIs.\n\n"

        s += "## What this report does not claim\n\n"
        s += "- **No model-form uncertainty.** The subgrid/collision closure's own error is not bounded here; that needs multi-model variation (Klein 2005), which this run does not perform.\n"
        s += "- **No Grid Convergence Index.** See above — invalid for scale-resolving runs.\n"
        s += "- Numerical uncertainty being small does **not** mean the answer is right: in the FDA nozzle round-robin, u_num was under 1% while the model was ~33% off experiment. Validation against data is the only cure, and the validation-domain section above is where that cure is (or is not) claimed.\n"
        return s
    }
}
