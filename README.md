# karman (working title)

A Metal-native lattice-Boltzmann CFD engine for Apple silicon, built
verification-first: **every claim the solver makes is backed by a gate it
must pass** — analytic exact solutions, published benchmark data, observed
convergence order, and bitwise run-to-run determinism.

Early days: this is the engine underneath a macOS fluid-dynamics instrument
in development. CLI only for now.

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

- **Lid-driven cavity** vs Ghia, Ghia & Shin (1982): Re=1000 centerline
  profiles at RMS 0.0039 (FP32+TRT) / 0.0058 (FP16+SRT) of lid velocity.
- **Poiseuille flow** (TRT, Λ=3/16): wall-adjacent error 4.3×10⁻⁷ —
  the viscosity-exact wall-location property, verified.
- **Taylor–Green vortex**: observed convergence order 1.95–1.96 against the
  analytic solution (with consistent initialization: analytic pressure and
  non-equilibrium parts — equilibrium-only init is a first-order error).
- **Bitwise determinism**: identical state digests across independent runs,
  at both storage precisions.
- Streaming/parity proofs: rest states are bitwise fixed points; a lone
  distribution injected in each of the 18 lattice directions propagates to
  exactly `cell + T·c_i` with zero leakage.

## Run it

Requires macOS 15+ on Apple silicon and Xcode command line tools.

```sh
swift build -c release
.build/release/karman m0      # spike gates: selftest, determinism, bench, cavity
.build/release/karman m1      # + FP16, Poiseuille exact, Taylor-Green order
.build/release/karman bench   # MLUPS on your machine, both precisions
```

Every gate prints its measured value next to its threshold and its source
reference — if something fails on your machine, that's a bug report we want.

## Notes

- FP16 storage pairs with SRT collision by design: TRT's antisymmetric mode
  is a difference of two half-quantized values, which amplifies quantization
  noise (measured: Ghia RMS degrades 5× with TRT at FP16).
- License: [Apache-2.0](LICENSE).
