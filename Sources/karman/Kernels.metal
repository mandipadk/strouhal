#include <metal_stdlib>
using namespace metal;

// D3Q19 lattice-Boltzmann. TRT collision (SRT = omegaMinus == omega),
// optional Smagorinsky LES, FP32 arithmetic, shifted DDFs (f_i - w_i),
// storage type FPXX (float or half, set per-library; FP16S of Lehmann PRE
// 106, 015308 — always FP32 compute).
//
// In-place streaming: AA-pattern (Bailey et al. 2009):
//   even step: read f_i^in = A(n,i); collide; write f_i^post -> A(n, opp(i))
//   odd  step: read f_i^in = A(n - c_i, opp(i)); collide; write f_i^post -> A(n + c_i, i)
// Each slot has a unique writer and unique reader per pass; every thread
// stages all 19 DDFs in registers before storing.
//
// Wall interactions are fused into the odd pass only:
// - SOLID: halfway bounce-back.
// - LID / INFLOW (moving-wall family): bounce-back with the Ladd momentum
//   correction 6 w_i rho_w (c_i . u_w). A wall-NORMAL velocity injects mass
//   flux rho_w*u_w exactly — the flux-exact velocity inlet; matched
//   velocity walls at both channel ends close mass exactly (M1b lesson:
//   equilibrium inlets are soft, copy outlets don't conserve mass).
//   rho_w = 1 for ALL moving walls. Two measured reasons: for the inlet,
//   local-rho coupling is a positive feedback loop (flux -> pressure ->
//   flux -> blowup during the ramp); for the lid, local rho WORSENS the
//   known mass drift (6.3e-3 vs 2.8e-3 per 50k steps at Re=100/128²) —
//   the drift comes from corner-link asymmetry, not the rho_w
//   approximation, and is reported as a conservation diagnostic instead.
// - OUTFLOW (dormant): anti-bounce-back pressure wall — needs its u-terms
//   under AA before it is trustworthy; kept for M2's open wakes.
//
// Determinism: no atomics, no simdgroup ops, fixed-order sums. Host compiles
// with mathMode = .safe and precise math functions.

#ifndef FPXX
#define FPXX float
#endif

// Pairing: for i >= 1, opp(i) = i odd ? i+1 : i-1.
constant int Cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
constant int Cy[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
constant int Cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
constant float W[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
constant int OPP[19] = {0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15, 18,17};

struct Params {
    uint  nx, ny, nz, parity;    // parity: 0 = even step, 1 = odd step
    float omega;       // omega+ (1/tau; 0 disables collision exactly)
    float omegaMinus;  // omega- (== omega for SRT)
    float uin;         // current (ramped) inflow peak velocity, +x
    float cSmago;      // Smagorinsky Cs^2 (0 = LES off)
    float fx, fy, fz;  // uniform body force density (lattice units)
    uint  writeForce;  // odd steps: accumulate momentum exchange per cell
    float ulidX, ulidY, ulidZ; // lid wall velocity (ramped)
    uint  pad0;
    float spongeX0;    // sponge zone start (lattice x); >= nx disables
    float spongeInvW;  // 1 / sponge width
    float spongeTau;   // target tau at the far end of the sponge
    uint  useEps;      // partially-saturated (Noble-Torczynski) cells present
};

constant uchar FLAG_FLUID   = 0;
constant uchar FLAG_SOLID   = 1;
constant uchar FLAG_LID     = 2;  // moving wall, velocity (ulidX, ulidY, ulidZ)
constant uchar FLAG_INFLOW  = 3;  // moving wall, parabolic +x profile (flux-exact inlet)
constant uchar FLAG_OUTFLOW = 4;  // DORMANT: anti-bounce-back pressure wall

inline uint fidx(uint i, uint n, uint N) { return i * N + n; }

inline int wrap(int v, int n) {
    v += (v < 0) ? n : 0;
    v -= (v >= n) ? n : 0;
    return v;
}

// Wall velocity for a moving-wall cell at (wy): lid = global vector;
// inflow = parabolic +x profile across the channel (walls at y=0.5, ny-1.5).
inline float3 wallVel(uchar wallFlag, int wy, constant Params& p) {
    if (wallFlag == FLAG_LID) { return float3(p.ulidX, p.ulidY, p.ulidZ); }
    const float H  = (float)p.ny - 2.0f;
    const float yd = (float)wy - 0.5f;
    return float3(max(4.0f * p.uin * yd * (H - yd) / (H * H), 0.0f), 0.0f, 0.0f);
}

// TRT collision + Guo forcing on shifted DDFs, pairwise (no feq array).
// Well-conditioned equilibrium (Lehmann PRE 106, 015308): rho-1 comes from
// the shifted sum; pair-symmetric part w_i*(rhom1 + rho*(cu^2/2 - 1.5u^2)),
// antisymmetric part w_i*rho*cu, cu = 3(c_i . u).
// Smagorinsky LES (Hou et al. 1996): |Pi_neq| from the local non-equilibrium
// momentum flux (no stencils); tau_eff via the closed form
// tau_eff = (tau0 + sqrt(tau0^2 + 18*sqrt(2)*Cs^2*|Pi|/rho))/2; the TRT magic
// parameter Lambda is preserved when rescaling both rates.
// Noble-Torczynski partially saturated cells (Noble & Torczynski 1998):
// a boundary cell with solid fraction eps blends collision with a bounce
// operator, B = eps(tau-1/2)/((1-eps)+(tau-1/2)):
//   f_i <- f_i + (1-B)*Omega_TRT_i + B*(f_opp(i) - f_i)        [static walls]
// Fully local (AA-safe by construction), second-order-ish for curved walls,
// and the momentum exchange F = -sum_i c_i B (f_opp - f_i) = 2B sum_pairs
// c_i (f_i - f_opp) is smooth in time — the staircase MEM's peak noise was
// the measured blocker for the DFG 2D-2 peak gates.
inline void collide(thread float* fh, constant Params& p, int x, float eps,
                    thread float& rhoOut, thread float3& ntForce) {
    float wp = p.omega, wm = p.omegaMinus;
    // Viscous sponge (outlet damping): ramp tau toward spongeTau across the
    // zone so unsteady wakes arrive at the velocity-wall outlet near the
    // profile it enforces (an undamped vortex street meeting a forced
    // parabola is a measured blowup). Lambda is preserved for TRT.
    if ((float)x > p.spongeX0) {
        const float frac = min(((float)x - p.spongeX0) * p.spongeInvW, 1.0f);
        const float tau0 = 1.0f / wp;
        const float tauS = tau0 + frac * (p.spongeTau - tau0);
        if (p.omegaMinus == p.omega) {
            wp = 1.0f / tauS; wm = wp;
        } else {
            const float lambda = (1.0f/wp - 0.5f) * (1.0f/wm - 0.5f);
            wp = 1.0f / tauS;
            wm = 1.0f / (0.5f + lambda / (tauS - 0.5f));
        }
    }
    float rhom1 = 0.0f;
    for (int i = 0; i < 19; i++) { rhom1 += fh[i]; }
    const float rho = 1.0f + rhom1;
    rhoOut = rho;
    const float inv = 1.0f / rho;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    for (int i = 1; i < 19; i++) {
        px += (float)Cx[i] * fh[i];
        py += (float)Cy[i] * fh[i];
        pz += (float)Cz[i] * fh[i];
    }
    const float ux = (px + 0.5f * p.fx) * inv;
    const float uy = (py + 0.5f * p.fy) * inv;
    const float uz = (pz + 0.5f * p.fz) * inv;
    const float u2 = ux*ux + uy*uy + uz*uz;

    if (p.cSmago > 0.0f) {
        float pxx = 0.0f, pyy = 0.0f, pzz = 0.0f, pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;
        for (int i = 1; i < 19; i++) {
            const float cu = 3.0f * ((float)Cx[i]*ux + (float)Cy[i]*uy + (float)Cz[i]*uz);
            const float feq = W[i] * (rhom1 + rho * (cu + 0.5f*cu*cu - 1.5f*u2));
            const float d = fh[i] - feq;
            pxx += (float)(Cx[i]*Cx[i]) * d;
            pyy += (float)(Cy[i]*Cy[i]) * d;
            pzz += (float)(Cz[i]*Cz[i]) * d;
            pxy += (float)(Cx[i]*Cy[i]) * d;
            pxz += (float)(Cx[i]*Cz[i]) * d;
            pyz += (float)(Cy[i]*Cz[i]) * d;
        }
        const float Q = sqrt(pxx*pxx + pyy*pyy + pzz*pzz
                             + 2.0f*(pxy*pxy + pxz*pxz + pyz*pyz));
        const float tau0 = 1.0f / wp;
        // 18*sqrt(2) = 25.455844
        const float tauEff = 0.5f * (tau0 + sqrt(tau0*tau0 + 25.455844f * p.cSmago * Q * inv));
        if (p.omegaMinus == p.omega) {
            wp = 1.0f / tauEff;
            wm = wp; // SRT-LES: preserving the near-zero SRT Lambda would
                     // drive omega- toward 2.0 — the stability boundary.
        } else {
            const float lambda = (1.0f/wp - 0.5f) * (1.0f/wm - 0.5f);
            wp = 1.0f / tauEff;
            wm = 1.0f / (0.5f + lambda / (tauEff - 0.5f));
        }
    }

    const float uF = ux*p.fx + uy*p.fy + uz*p.fz;
    const float ap = 1.0f - 0.5f * wp;   // even-source prefactor
    const float am = 1.0f - 0.5f * wm;   // odd-source prefactor

    // The B == 0 path keeps the ORIGINAL arithmetic exactly: multiplying the
    // collision increments by (1-B)=1.0 changes the rounding pattern, and at
    // the FP32 accumulation floor that regressed the Poiseuille-exact gate
    // 12x. Bit-compatibility of the golden path is part of the contract.
    if (eps == 0.0f) {
        { // rest direction: purely symmetric
            const float feq0 = W[0] * (rhom1 - 1.5f * rho * u2);
            const float s0   = W[0] * ap * (-3.0f * uF);
            fh[0] = fma(wp, feq0 - fh[0], fh[0]) + s0;
        }
        for (int i = 1; i < 19; i += 2) {
            const float cu = 3.0f * ((float)Cx[i]*ux + (float)Cy[i]*uy + (float)Cz[i]*uz);
            const float cF = (float)Cx[i]*p.fx + (float)Cy[i]*p.fy + (float)Cz[i]*p.fz;
            const float feqp = W[i] * (rhom1 + rho * (0.5f*cu*cu - 1.5f*u2)); // symmetric eq
            const float feqm = W[i] * rho * cu;                               // antisymmetric eq
            const float fp = 0.5f * (fh[i] + fh[i+1]);
            const float fm = 0.5f * (fh[i] - fh[i+1]);
            const float dp = wp * (feqp - fp);
            const float dm = wm * (feqm - fm);
            // Guo source, TRT-split: odd part 3(c.F); even part 3 cu (c.F) - 3 u.F
            const float sm = W[i] * am * (3.0f * cF);
            const float sp = W[i] * ap * (3.0f * cu * cF - 3.0f * uF);
            fh[i]   += dp + dm + sp + sm;
            fh[i+1] += dp - dm + sp - sm;
        }
        return;
    }

    const float tw = 1.0f / wp - 0.5f;
    const float B = eps * tw / ((1.0f - eps) + tw);
    const float oneMinusB = 1.0f - B;

    { // rest direction: purely symmetric (bounce term vanishes)
        const float feq0 = W[0] * (rhom1 - 1.5f * rho * u2);
        const float s0   = W[0] * ap * (-3.0f * uF);
        fh[0] += oneMinusB * (wp * (feq0 - fh[0]) + s0);
    }
    for (int i = 1; i < 19; i += 2) {
        const float cu = 3.0f * ((float)Cx[i]*ux + (float)Cy[i]*uy + (float)Cz[i]*uz);
        const float cF = (float)Cx[i]*p.fx + (float)Cy[i]*p.fy + (float)Cz[i]*p.fz;
        const float feqp = W[i] * (rhom1 + rho * (0.5f*cu*cu - 1.5f*u2)); // symmetric eq
        const float feqm = W[i] * rho * cu;                               // antisymmetric eq
        const float fp = 0.5f * (fh[i] + fh[i+1]);
        const float fm = 0.5f * (fh[i] - fh[i+1]);
        const float dp = wp * (feqp - fp);
        const float dm = wm * (feqm - fm);
        const float sm = W[i] * am * (3.0f * cF);
        const float sp = W[i] * ap * (3.0f * cu * cF - 3.0f * uF);
        const float bounce = B * (fh[i+1] - fh[i]); // static wall (u_w = 0)
        fh[i]   += oneMinusB * (dp + dm + sp + sm) + bounce;
        fh[i+1] += oneMinusB * (dp - dm + sp - sm) - bounce;
        ntForce.x -= 2.0f * (float)Cx[i] * bounce;
        ntForce.y -= 2.0f * (float)Cy[i] * bounce;
        ntForce.z -= 2.0f * (float)Cz[i] * bounce;
    }
}

kernel void step(device FPXX*        f         [[buffer(0)]],
                 device const uchar* flags     [[buffer(1)]],
                 device const uint*  solidMask [[buffer(2)]],  // bit i-1: neighbor n + c_i is solid
                 device const uint*  lidMask   [[buffer(3)]],  // subset: moving-wall family / ABB
                 constant Params&    p         [[buffer(4)]],
                 device float4*      force     [[buffer(5)]],  // momentum exchange (writeForce)
                 device const float* epsBuf    [[buffer(6)]],  // NT solid fraction (useEps)
                 uint n [[thread_position_in_grid]])
{
    const uint N = p.nx * p.ny * p.nz;
    if (n >= N || flags[n] != FLAG_FLUID) { return; }
    const float eps = (p.useEps != 0u) ? epsBuf[n] : 0.0f;

    const int x = (int)(n % p.nx);
    const int t = (int)(n / p.nx);
    const int y = t % (int)p.ny;
    const int z = t / (int)p.ny;

    float fh[19];
    float rho = 1.0f;
    float3 ntF = float3(0.0f);

    if (p.parity == 0u) {
        // -------- even: pure in-cell pass --------
        for (int i = 0; i < 19; i++) { fh[i] = (float)f[fidx(i, n, N)]; }
        collide(fh, p, x, eps, rho, ntF);
        f[fidx(0, n, N)] = (FPXX)fh[0];
        for (int i = 1; i < 19; i++) { f[fidx(OPP[i], n, N)] = (FPXX)fh[i]; }
        if (p.writeForce != 0u && eps > 0.0f) {
            force[n] = float4(ntF, 0.0f);
        }
    } else {
        // -------- odd: streaming pass (neighbors + walls) --------
        // Momentum exchange (Mei/Ladd): flux fluid->wall across a link with
        // toward-wall direction j is c_j (f_j^out + f_i^in), i = opp(j), in
        // RAW DDFs; shifted DDFs add 2 w_j c_j per link (cancels over closed
        // bodies). Load side handles f_i^in (-c_i = c_j), store side
        // f_j^out + 2 w_j.
        const uint sMask = solidMask[n];
        const uint lMask = lidMask[n];
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        uint deferred = 0; // moving-wall load corrections, applied after rho_pre
        fh[0] = (float)f[fidx(0, n, N)];
        for (int i = 1; i < 19; i++) {
            // incoming along i from source s = n - c_i (neighbor in dir opp(i))
            const int srcBit = OPP[i] - 1;
            if ((sMask >> srcBit) & 1u) {
                fh[i] = (float)f[fidx(i, n, N)];
                if ((lMask >> srcBit) & 1u) { deferred |= (1u << (uint)i); }
                if (p.writeForce != 0u) {
                    ax -= (float)Cx[i] * fh[i];
                    ay -= (float)Cy[i] * fh[i];
                    az -= (float)Cz[i] * fh[i];
                }
            } else {
                const int sx = wrap(x - Cx[i], (int)p.nx);
                const int sy = wrap(y - Cy[i], (int)p.ny);
                const int sz = wrap(z - Cz[i], (int)p.nz);
                const uint s = ((uint)sz * p.ny + (uint)sy) * p.nx + (uint)sx;
                fh[i] = (float)f[fidx(OPP[i], s, N)];
            }
        }
        if (deferred != 0u) {
            for (int i = 1; i < 19; i++) {
                if ((deferred >> (uint)i) & 1u) {
                    const int sy = wrap(y - Cy[i], (int)p.ny);
                    const uint s = ((uint)wrap(z - Cz[i], (int)p.nz) * p.ny + (uint)sy) * p.nx
                                 + (uint)wrap(x - Cx[i], (int)p.nx);
                    const uchar wf = flags[s];
                    if (wf == FLAG_OUTFLOW) {
                        fh[i] = -fh[i]; // dormant ABB
                    } else {
                        const float3 uw = wallVel(wf, sy, p);
                        const float cu = (float)Cx[i]*uw.x + (float)Cy[i]*uw.y + (float)Cz[i]*uw.z;
                        const float corr = 6.0f * W[i] * cu; // rho_w = 1 (see header)
                        fh[i] += corr;
                        if (p.writeForce != 0u) {
                            ax -= (float)Cx[i] * corr;
                            ay -= (float)Cy[i] * corr;
                            az -= (float)Cz[i] * corr;
                        }
                    }
                }
            }
        }
        collide(fh, p, x, eps, rho, ntF);
        ax += ntF.x; ay += ntF.y; az += ntF.z;
        f[fidx(0, n, N)] = (FPXX)fh[0];
        for (int i = 1; i < 19; i++) {
            const int dstBit = i - 1;
            if ((sMask >> dstBit) & 1u) {
                float corr = 0.0f;
                float sign = 1.0f;
                if ((lMask >> dstBit) & 1u) {
                    const int dy = wrap(y + Cy[i], (int)p.ny);
                    const uint d = ((uint)wrap(z + Cz[i], (int)p.nz) * p.ny + (uint)dy) * p.nx
                                 + (uint)wrap(x + Cx[i], (int)p.nx);
                    const uchar wf = flags[d];
                    if (wf == FLAG_OUTFLOW) {
                        sign = -1.0f; // dormant ABB
                    } else {
                        const float3 uw = wallVel(wf, dy, p);
                        corr = 6.0f * W[i]
                             * ((float)Cx[i]*uw.x + (float)Cy[i]*uw.y + (float)Cz[i]*uw.z);
                    }
                }
                f[fidx(OPP[i], n, N)] = (FPXX)(sign * fh[i] - corr);
                if (p.writeForce != 0u) {
                    const float v = sign * fh[i] + 2.0f * W[i];
                    ax += (float)Cx[i] * v;
                    ay += (float)Cy[i] * v;
                    az += (float)Cz[i] * v;
                }
            } else {
                const int dx = wrap(x + Cx[i], (int)p.nx);
                const int dy = wrap(y + Cy[i], (int)p.ny);
                const int dz = wrap(z + Cz[i], (int)p.nz);
                const uint d = ((uint)dz * p.ny + (uint)dy) * p.nx + (uint)dx;
                f[fidx(i, d, N)] = (FPXX)fh[i];
            }
        }
        if (p.writeForce != 0u && (sMask != 0u || eps > 0.0f)) {
            force[n] = float4(ax, ay, az, 0.0f);
        }
    }
}

// Moments probe. Valid only after an even number of completed steps (natural
// slots). Velocity includes the Guo half-force shift.
kernel void momentsEven(device const FPXX*  f     [[buffer(0)]],
                        device const uchar* flags [[buffer(1)]],
                        device float4*      out   [[buffer(2)]],
                        constant Params&    p     [[buffer(3)]],
                        uint n [[thread_position_in_grid]])
{
    const uint N = p.nx * p.ny * p.nz;
    if (n >= N) { return; }
    if (flags[n] != FLAG_FLUID) { out[n] = float4(0.0f); return; }
    float rhom1 = 0.0f, px = 0.0f, py = 0.0f, pz = 0.0f;
    for (int i = 0; i < 19; i++) {
        const float v = (float)f[fidx(i, n, N)];
        rhom1 += v;
        px += (float)Cx[i] * v;
        py += (float)Cy[i] * v;
        pz += (float)Cz[i] * v;
    }
    const float rho = 1.0f + rhom1;
    out[n] = float4((px + 0.5f * p.fx) / rho,
                    (py + 0.5f * p.fy) / rho,
                    (pz + 0.5f * p.fz) / rho, rho);
}

// Initializer. mode 0: rest. mode 1: 2D Taylor-Green (one period per box)
// with CONSISTENT initialization — analytic pressure (rho-1 = 3p; sign for
// the u=+A sin x cos y phase is +) and the analytic non-equilibrium part
// (equilibrium-only init is a first-order error; Mei et al. 2006).
struct InitParams {
    uint  nx, ny, nz, mode;
    float amplitude;
    float tau;
    float pad1, pad2;
};

kernel void initField(device FPXX*         f [[buffer(0)]],
                      constant InitParams& p [[buffer(1)]],
                      uint n [[thread_position_in_grid]])
{
    const uint N = p.nx * p.ny * p.nz;
    if (n >= N) { return; }
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    float sxx = 0.0f, rhom1 = 0.0f;
    if (p.mode == 1u) {
        const uint x = n % p.nx;
        const uint t = n / p.nx;
        const uint y = t % p.ny;
        const float kx = 2.0f * M_PI_F / (float)p.nx;
        const float ky = 2.0f * M_PI_F / (float)p.ny;
        const float xa = (float)x + 0.5f, ya = (float)y + 0.5f;
        ux =  p.amplitude * sin(kx * xa) * cos(ky * ya);
        uy = -p.amplitude * cos(kx * xa) * sin(ky * ya);
        sxx = p.amplitude * kx * cos(kx * xa) * cos(ky * ya);
        rhom1 = 0.75f * p.amplitude * p.amplitude
              * (cos(2.0f * kx * xa) + cos(2.0f * ky * ya)); // rho-1 = 3p
    }
    const float rho = 1.0f + rhom1;
    const float u2 = ux*ux + uy*uy + uz*uz;
    for (int i = 0; i < 19; i++) {
        const float cu = 3.0f * ((float)Cx[i]*ux + (float)Cy[i]*uy + (float)Cz[i]*uz);
        const float feq = W[i] * (rhom1 + rho * (cu + 0.5f*cu*cu - 1.5f*u2));
        const float cc = (float)(Cx[i]*Cx[i] - Cy[i]*Cy[i]);
        const float fneq = -3.0f * W[i] * p.tau * sxx * cc; // S_xy = 0, tr S = 0 for symmetric TG
        f[fidx(i, n, N)] = (FPXX)(feq + fneq);
    }
}
