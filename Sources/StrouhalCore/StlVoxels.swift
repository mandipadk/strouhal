import Foundation
import simd

/// Binary-STL triangle soup. (ASCII STL is not supported yet — say so
/// loudly rather than mis-parse.)
public struct StlMesh: Sendable {
    public var triangles: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)]
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>

    public init(binarySTL data: Data) throws {
        guard data.count >= 84 else { throw StrouhalError.message("STL too short") }
        if let head = String(data: data.prefix(5), encoding: .ascii),
           head.lowercased() == "solid",
           data.count < 84 + 50 { // heuristic: real binary can also start with "solid"
            throw StrouhalError.message("ASCII STL not supported (binary only)")
        }
        let count = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 80, as: UInt32.self) }
        guard data.count >= 84 + Int(count) * 50 else {
            throw StrouhalError.message("STL truncated (or ASCII; binary only)")
        }
        var tris: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
        tris.reserveCapacity(Int(count))
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        data.withUnsafeBytes { raw in
            for k in 0..<Int(count) {
                let base = 84 + k * 50 + 12 // skip normal
                func v(_ j: Int) -> SIMD3<Float> {
                    SIMD3(raw.loadUnaligned(fromByteOffset: base + j * 12 + 0, as: Float.self),
                          raw.loadUnaligned(fromByteOffset: base + j * 12 + 4, as: Float.self),
                          raw.loadUnaligned(fromByteOffset: base + j * 12 + 8, as: Float.self))
                }
                let t = (v(0), v(1), v(2))
                tris.append(t)
                for p in [t.0, t.1, t.2] {
                    lo = min(lo, p); hi = max(hi, p)
                }
            }
        }
        triangles = tris
        boundsMin = lo
        boundsMax = hi
    }
}

/// Voxelize a (closed, watertight) mesh into Noble-Torczynski solid
/// fractions. Parity ray casting along +x: per (y,z) sub-ray the interior
/// x-spans are EXACT; 2×2 sub-rays per cell give fractional lateral
/// coverage. `scale`/`offset` map mesh coordinates into lattice cells
/// (cell centers at integer coordinates).
public func meshSolidFractions(mesh: StlMesh, nx: Int, ny: Int, nz: Int,
                               scale: Float, offset: SIMD3<Float>) -> [Float] {
    var eps = [Float](repeating: 0, count: nx * ny * nz)
    // lattice-space triangles
    let tris = mesh.triangles.map { t in
        (t.0 * scale + offset, t.1 * scale + offset, t.2 * scale + offset)
    }
    // bucket triangles over the (y,z) cell grid by their bbox
    var buckets = [[Int]](repeating: [], count: ny * nz)
    for (i, t) in tris.enumerated() {
        let ylo = max(0, Int(floor(min(t.0.y, t.1.y, t.2.y) - 0.5)))
        let yhi = min(ny - 1, Int(ceil(max(t.0.y, t.1.y, t.2.y) + 0.5)))
        let zlo = max(0, Int(floor(min(t.0.z, t.1.z, t.2.z) - 0.5)))
        let zhi = min(nz - 1, Int(ceil(max(t.0.z, t.1.z, t.2.z) + 0.5)))
        if ylo > yhi || zlo > zhi { continue }
        for z in zlo...zhi { for y in ylo...yhi { buckets[z * ny + y].append(i) } }
    }
    let sub = 2 // sub-rays per cell edge
    for z in 0..<nz { for y in 0..<ny {
        let cands = buckets[z * ny + y]
        if cands.isEmpty { continue }
        for sy in 0..<sub { for sz in 0..<sub {
            let ry = Float(y) - 0.5 + (Float(sy) + 0.5) / Float(sub)
            let rz = Float(z) - 0.5 + (Float(sz) + 0.5) / Float(sub)
            // gather x-crossings of the ray (t, ry, rz)
            var xs: [Float] = []
            for i in cands {
                let (a, b, c) = tris[i]
                // 2D point-in-triangle in (y,z); solve x by barycentric interp
                let d1 = SIMD2(b.y - a.y, b.z - a.z)
                let d2 = SIMD2(c.y - a.y, c.z - a.z)
                let denom = d1.x * d2.y - d1.y * d2.x
                if abs(denom) < 1e-12 { continue } // ray-parallel triangle
                let p = SIMD2(ry - a.y, rz - a.z)
                let u = (p.x * d2.y - p.y * d2.x) / denom
                let v = (d1.x * p.y - d1.y * p.x) / denom
                if u < 0 || v < 0 || u + v > 1 { continue }
                xs.append(a.x + u * (b.x - a.x) + v * (c.x - a.x))
            }
            guard xs.count >= 2 else { continue }
            xs.sort()
            // merge tangency duplicates, pair into interior spans
            var spans: [(Float, Float)] = []
            var k = 0
            while k + 1 < xs.count {
                if xs[k + 1] - xs[k] < 1e-5 { k += 1; continue } // grazing hit
                spans.append((xs[k], xs[k + 1]))
                k += 2
            }
            for (x0, x1) in spans {
                let c0 = max(0, Int(floor(x0 + 0.5)))
                let c1 = min(nx - 1, Int(floor(x1 + 0.5)))
                if c0 > c1 { continue }
                for x in c0...c1 {
                    let cellLo = Float(x) - 0.5, cellHi = Float(x) + 0.5
                    let cover = min(x1, cellHi) - max(x0, cellLo)
                    if cover > 0 {
                        eps[(z * ny + y) * nx + x] += cover / Float(sub * sub)
                    }
                }
            }
        }}
    }}
    for i in 0..<eps.count { eps[i] = min(eps[i], 1.0) }
    return eps
}

/// A programmatic lat-long sphere STL (for tests and demos).
public func sphereSTLData(radius: Float, segments: Int = 48, rings: Int = 24) -> Data {
    var tris: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
    func pt(_ i: Int, _ j: Int) -> SIMD3<Float> {
        let theta = Float(j) * .pi / Float(rings)
        let phi = Float(i) * 2 * .pi / Float(segments)
        return SIMD3(radius * sin(theta) * cos(phi),
                     radius * sin(theta) * sin(phi),
                     radius * cos(theta))
    }
    for j in 0..<rings { for i in 0..<segments {
        let a = pt(i, j), b = pt(i + 1, j), c = pt(i, j + 1), d = pt(i + 1, j + 1)
        if j > 0 { tris.append((a, b, c)) }
        if j < rings - 1 { tris.append((b, d, c)) }
    }}
    var data = Data(count: 80)
    var n = UInt32(tris.count)
    withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }
    for t in tris {
        var rec = [Float](repeating: 0, count: 3) // normal (unused)
        rec += [t.0.x, t.0.y, t.0.z, t.1.x, t.1.y, t.1.z, t.2.x, t.2.y, t.2.z]
        rec.withUnsafeBytes { data.append(contentsOf: $0) }
        data.append(contentsOf: [0, 0]) // attribute byte count
    }
    return data
}
