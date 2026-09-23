# Strouhal

A Metal-native lattice-Boltzmann CFD instrument for Apple silicon, built
verification-first: **every claim the solver makes is backed by a gate it
must pass** — analytic exact solutions, published benchmark data, observed
convergence order, and bitwise run-to-run determinism. And every quantity
the app reports can be **hardened**: a resolution ladder plus a half-Mach
anchor, run automatically, assemble a defensible uncertainty
U(φ) = k·√(u_num² + u_stat² + u_Ma²) — with a validation-domain verdict
attached, and a refusal where no calibrated bar can be defended.

Named for the Strouhal number — the dimensionless shedding frequency of the
Kármán vortex street, this project's first validated unsteady case and its
live demo. A SwiftUI instrument app runs alongside the CLI gates.

## What works today

D3Q19, TRT/SRT collision, FP32 arithmetic with FP32 or FP16 storage
(Lehmann-style shifted DDFs + well-conditioned equilibrium), in-place
AA-pattern streaming, halfway bounce-back with moving walls, Guo body
forcing — all under a determinism contract: `mathMode = .safe`, no atomics,
fixed-order reductions, hash-verified reproducibility.

Measured on a base Apple M5 (16 GB), 256³ cells:

| Configuration | Throughput |
|---|---|
| FP32 storage | ~785 MLUPS (~123 GB/s, ≈80% of peak bandwidth) |
| FP16 storage | ~1,445 MLUPS |

Physics gates currently passing (run them yourself, see below):

| Gate | Reference | Measured |
|---|---|---|
| Lid-driven cavity, Re 100–1000 | Ghia, Ghia & Shin (1982) | centerline RMS 0.0039 of u_lid |
| Poiseuille (TRT, Λ=3/16) | analytic, wall-exact | wall-adjacent error 4.3×10⁻⁷ |
| Taylor–Green 2D order | analytic | observed order 1.95–1.96 |
| Cylinder drag, steady (DFG 2D-1) | Schäfer & Turek (1996) | C_D within 0.16% |
| Vortex street (DFG 2D-2) | Schäfer & Turek (1996) | St 0.2996 [0.295, 0.305], max C_L 0.9998 [0.99, 1.01] |
| **Taylor–Green 3D, Re 1600** | Incompact3d 512³ DNS / HiOCFD | peak ε at t* 9.09 (DNS 8.98); finest pair converged to 0.25%; peak −7.5% with the deficit attributed by measurement (compressibility ruled out; 2nd-order resolution) |
| Curved boundaries (Noble–Torczynski) | DFG 2D-2 peaks | max C_L gate active at D=64 |
| Sphere drag, hardened | Schiller–Naumann correlation (±5%) | resolution ladder + Mach anchor, calibrated bar |
| Bitwise determinism | — | identical SHA-256 state digests, both precisions |

Streaming/parity proofs: rest states are bitwise fixed points; a lone
distribution injected in each of the 18 lattice directions propagates to
exactly `cell + T·c_i` with zero leakage.

The 3D TGV gate is worth reading as a statement of method: the 256³ rung
reads *closer* to DNS than the finer grids — and is excluded, because it
sits 1.3 convective times from its own numerical blow-up and rides on
spurious energy. Accidental agreement is not accuracy. The remaining
deficit is attributed by experiment, not hand-waved: halving the Mach
number moves the peak by one part in 10⁴, so compressibility is ruled out
and the deficit is what a second-order scheme at 320³ leaves on the table.

## Run it

Requires macOS 15+ on Apple silicon and Xcode command line tools.

```sh
swift build -c release
.build/release/strouhal m0      # spike gates: selftest, determinism, bench, cavity
.build/release/strouhal m1      # + FP16, Poiseuille exact, Taylor-Green order
.build/release/strouhal bench   # MLUPS on your machine, both precisions
.build/release/strouhal m2      # the Kármán vortex street: Strouhal number, peaks, replay
.build/release/strouhal m3      # STL voxelizer, 3D sphere wake
.build/release/strouhal m4      # the credibility run: ladder + Mach anchor + report
.build/release/strouhal m5      # Taylor-Green Re=1600 vs published DNS
.build/release/strouhal m6      # hardening a 3D body: the sphere credibility run
.build/release/StrouhalApp      # the live instrument
sh scripts/make-app.sh          # bundle dist/Strouhal.app
```

Every gate prints its measured value next to its threshold and its source
reference — if something fails on your machine, that's a bug report we want.

## Hardening a number

Any quantity the app reports can be *hardened*: it re-runs the case across a
resolution ladder and a half-Mach anchor and assembles

    U(φ) = k·√(u_num² + u_stat² + u_Ma²),  k = 2 (≈95%)

then attaches a validation-domain verdict and exports a Markdown report in
ASME V&V 20 vocabulary. This works on the built-in cases and on your own
imported geometry. It costs minutes, and it runs on a second GPU queue so the
live view keeps going.

The transient is not a guessed number of steps. Each run continues until its
quantity genuinely stops moving, because a guess here is silently wrong: the
sphere's drag was measured passing through 3.13, down to 0.97, up to 1.58, and
only flattening near 1.3653 after about 150 convective times. Averaging inside
that ramp yields a plausible value with an error bar that describes an
unconverged drift rather than any real uncertainty.

What it refuses to do matters as much:

- **No Grid Convergence Index.** Richardson extrapolation assumes monotone
  convergence at a fitted order, which scale-resolving simulations do not
  guarantee. The observed order is printed as a diagnostic, never as the basis
  of the bar.
- **No calibrated bar outside the validated domain.** An arbitrary imported
  shape has no validation anchor, so the report gives the measured components
  and states plainly that whether the solver reproduces reality *for that
  shape* is not established.
- **No model-form uncertainty claim**, which would need multi-model variation.

A small numerical uncertainty is not the same as a correct answer. In the FDA's
blood-nozzle round-robin, numerical uncertainty was under 1% while results were
about 33% off experiment.

## Notes

- FP16 storage pairs with SRT collision by design: TRT's antisymmetric mode
  is a difference of two half-quantized values, which amplifies quantization
  noise (measured: Ghia RMS degrades 5× with TRT at FP16).
- License: [Apache-2.0](LICENSE).
