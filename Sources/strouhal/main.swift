import Foundation
import StrouhalCore

// strouhal CLI — verification gate driver.
//   karman selftest              parity/indexing proofs
//   karman bench [n]             MLUPS benchmark (default 256)
//   karman cavity [re] [n]       Ghia validation (default 1000, 256)
//   karman determinism           run-twice digest gate
//   karman m0                    all M0 gates + summary

func printResult(_ r: GateResult) {
    print("\(r.passed ? "PASS" : "FAIL")  \(r.name)")
    for line in r.detail.split(separator: "\n") {
        print("      \(line)")
    }
}

func main() throws {
    let args = CommandLine.arguments.dropFirst()
    let cmd = args.first ?? "m0"
    let gpu = try GPU()
    print("device: \(gpu.device.name)")

    var results: [GateResult] = []
    let wall = Date()

    switch cmd {
    case "selftest":
        results = try runSelftest(gpu: gpu)
    case "bench":
        let n = args.dropFirst().first.flatMap { Int($0) } ?? 256
        results = [try runBench(gpu: gpu, n: n, precision: .fp32),
                   try runBench(gpu: gpu, n: n, precision: .fp16s)]
    case "cavity":
        let re = args.dropFirst().first.flatMap { Double($0) } ?? 1000
        let n = args.dropFirst(2).first.flatMap { Int($0) } ?? 256
        let maxSteps = re <= 100 ? 100_000 : 1_200_000
        let run = try cavity(gpu: gpu, n: n, re: re, maxSteps: maxSteps)
        print(String(format: "mass drift: %.3e", run.sim.massSum()))
        results = [try ghiaComparison(run: run, re: re)]
    case "determinism":
        results = [try runDeterminism(gpu: gpu, precision: .fp32),
                   try runDeterminism(gpu: gpu, precision: .fp16s)]
    case "sphere":
        let ds = args.dropFirst().first.flatMap { Int($0) } ?? 24
        let bx = args.dropFirst(2).first.flatMap { Int($0) } ?? (ds * 4)
        results = [try runSphere(gpu: gpu, D: ds, box: bx)]
    case "tgv":
        let n = args.dropFirst().first.flatMap { Int($0) } ?? 128
        let u0 = args.dropFirst(2).first.flatMap { Float($0) } ?? 0.1
        let tEnd = args.dropFirst(3).first.flatMap { Double($0) } ?? 10.5
        _ = try tgvRun(gpu: gpu, n: n, u0: u0, tEnd: tEnd) { print("      \($0)") }
        results = []
    case "debugsettle":
        try debugSettleTrigger(gpu: gpu, D: args.dropFirst().first.flatMap { Int($0) } ?? 28)
        results = []
    case "debugbody":
        let d = args.dropFirst().first.flatMap { Int($0) } ?? 28
        let tc = args.dropFirst(2).first.flatMap { Double($0) } ?? 80
        try debugBodySettling(gpu: gpu, D: d, convectiveTimes: tc)
        results = []
    case "m6":
        print("— hardening a 3D body: the sphere credibility run —")
        results = try runBodyCredibility(gpu: gpu)
    case "m5":
        print("— TGV Re=1600 vs DNS (the 3D turbulence anchor) —")
        results = try runTGV1600(gpu: gpu)
    case "m4":
        print("— the credibility run (resolution ladder + Mach anchor) —")
        results = try runCredibility(gpu: gpu)
    case "genstl":
        let path = args.dropFirst().first ?? "/tmp/testsphere.stl"
        try sphereSTLData(radius: 1.0).write(to: URL(fileURLWithPath: path))
        print("wrote \(path)")
        results = []
    case "m3":
        print("— STL voxelizer —")
        let stl = try runStlVoxelizer()
        printResult(stl); results.append(stl)
        print("— 3D sphere wake (demo-case gate) —")
        let sph = try runSphere(gpu: gpu, D: 24, box: 96)
        printResult(sph); results.append(sph)
        print("NOTE: app-side M3 gates are measured via STROUHAL_APP_SECONDS autotests;")
        print("      latest: street 55 fps/5.5k steps/s; cavity 59 fps/15k steps/s,")
        print("      u_center matches Ghia live; sphere 192³ 33 fps/85 steps/s.")
    case "m2":
        print("— DFG 2D-1 steady, NT curved boundaries —")
        let d1nt = try runDFG1(gpu: gpu, D: 40, curved: true)
        printResult(d1nt); results.append(d1nt)
        print("— DFG 2D-2 (Kármán vortex street), NT curved, D=64 —")
        for r in try runDFG2(gpu: gpu, D: 64, uinMax: 0.05, curved: true) {
            printResult(r); results.append(r)
        }
        let rep = try runDFG2Replay(gpu: gpu)
        printResult(rep); results.append(rep)
    case "debugdfg2":
        try runDebugDFG2(gpu: gpu, lambda: 0.25)
        results = []
    case "dfg2":
        let d2 = args.dropFirst().first.flatMap { Int($0) } ?? 40
        let u2 = args.dropFirst(2).first.flatMap { Float($0) } ?? 0.075
        let len = args.dropFirst(3).first.flatMap { Int($0) } ?? 22
        let spD = args.dropFirst(4).first.flatMap { Int($0) } ?? 3
        let spT = args.dropFirst(5).first.flatMap { Float($0) } ?? 1.0
        let c2 = args.contains("nt")
        results = try runDFG2(gpu: gpu, D: d2, uinMax: u2, lengthD: len,
                              spongeD: spD, spongeTau: spT, curved: c2)
    case "debugles":
        try runDebugLES(gpu: gpu, re: 1e5, cSmago: 0.01, lambda: 0.25)
        try runDebugLES(gpu: gpu, re: 1e5, cSmago: 0.04, lambda: 0.25)
        try runDebugLES(gpu: gpu, re: 1e4, cSmago: 0.01, lambda: 0.25)
        results = []
    case "m1c":
        print("— LES —")
        for r in [try runLESStability(gpu: gpu), try runLESLaminarCost(gpu: gpu)] {
            printResult(r); results.append(r)
        }
        print("— 3D —")
        for r in [try runCavity3DPeriodicZ(gpu: gpu), try runCavityCubic(gpu: gpu)] {
            printResult(r); results.append(r)
        }
        print("— symmetry / units / conservation —")
        for r in [try runRotationTest(gpu: gpu), runUnitsTest(), try runMassDrift(gpu: gpu)] {
            printResult(r); results.append(r)
        }
    case "debugchan":
        results = [try runDebugChannel(gpu: gpu)]
    case "channel":
        results = [try runChannelTest(gpu: gpu)]
    case "dfg":
        let d = args.dropFirst().first.flatMap { Int($0) } ?? 40
        let u = args.dropFirst(2).first.flatMap { Float($0) } ?? 0.075
        let up = args.dropFirst(3).first.flatMap { Double($0) } ?? 2.0
        let curved = args.contains("nt")
        results = [try runDFG1(gpu: gpu, D: d, uinMax: u, upstreamD: up, curved: curved)]
    case "poiseuille":
        results = [try runPoiseuille(gpu: gpu)]
    case "tgorder":
        let u0 = args.dropFirst().first.flatMap { Double($0) } ?? 0.20
        results = [try runTaylorGreenOrder(gpu: gpu, u0base: u0)]
    case "m1":
        print("— selftest —")
        results += try runSelftest(gpu: gpu)
        results.forEach(printResult)
        guard results.allSatisfy(\.passed) else { break }

        print("— bench (both precisions) —")
        for prec in [Precision.fp32, .fp16s] {
            let r = try runBench(gpu: gpu, n: 256, precision: prec)
            printResult(r); results.append(r)
        }
        print("— determinism (fp16s) —")
        let det16 = try runDeterminism(gpu: gpu, precision: .fp16s)
        printResult(det16); results.append(det16)

        print("— Poiseuille exact —")
        let poise = try runPoiseuille(gpu: gpu)
        printResult(poise); results.append(poise)

        print("— Taylor-Green order —")
        let tg = try runTaylorGreenOrder(gpu: gpu)
        printResult(tg); results.append(tg)

        print("— Schäfer–Turek 2D-1 —")
        let dfg = try runDFG1(gpu: gpu, D: 40)
        printResult(dfg); results.append(dfg)

        print("— cavity Re=1000: fp32+TRT, fp16s+SRT —")
        // FP16S pairs with SRT: TRT's antisymmetric mode is a difference of
        // two half-quantized values, which amplifies quantization noise
        // (FluidX3D defaults to SRT for the same reason).
        for (prec, lam) in [(Precision.fp32, Optional(0.25)), (.fp16s, nil)] {
            let run = try cavity(gpu: gpu, precision: prec, n: 256, re: 1000,
                                 lambda: lam, maxSteps: 1_200_000)
            let g = try ghiaComparison(run: run, re: 1000, verbose: false)
            printResult(g); results.append(g)
        }
    case "m0":
        print("— selftest —")
        results += try runSelftest(gpu: gpu)
        results.forEach(printResult)
        guard results.allSatisfy(\.passed) else { break }

        print("— determinism —")
        let det = try runDeterminism(gpu: gpu, precision: .fp32)
        printResult(det)
        results.append(det)

        print("— bench 256³ —")
        let bench = try runBench(gpu: gpu, n: 256, precision: .fp32)
        printResult(bench)
        results.append(bench)

        print("— cavity Re=100 (smoke) —")
        let run100 = try cavity(gpu: gpu, n: 128, re: 100, maxSteps: 100_000)
        let g100 = try ghiaComparison(run: run100, re: 100)
        printResult(g100)
        results.append(g100)

        print("— cavity Re=1000 (gate) —")
        let run1000 = try cavity(gpu: gpu, n: 256, re: 1000, maxSteps: 1_200_000)
        let g1000 = try ghiaComparison(run: run1000, re: 1000)
        printResult(g1000)
        results.append(g1000)
    default:
        print("unknown command \(cmd)")
        exit(2)
    }

    if cmd != "m0" { results.forEach(printResult) }
    let passed = results.filter(\.passed).count
    print(String(format: "\n%d/%d gates passed in %.1fs wall",
                 passed, results.count, -wall.timeIntervalSinceNow))
    exit(passed == results.count ? 0 : 1)
}

do {
    try main()
} catch {
    print("error: \(error)")
    exit(1)
}
