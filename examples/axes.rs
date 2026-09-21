// Writes a tiny PLY with colored blobs on +X (red), +Y (green), +Z (blue) for orientation checks.
use mesh2splat::{ply, GaussianVertex, PlyFormat};
fn main() {
    let mut gs = Vec::new();
    let mut add = |p: [f32; 3], c: [f32; 3]| {
        gs.push(GaussianVertex {
            position: [p[0], p[1], p[2], 1.0],
            color: [c[0], c[1], c[2], 1.0],
            scale: [0.1, 0.1, 0.1, 0.0],
            normal: [0.0, 0.0, 1.0, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
            pbr: [0.0; 4],
        })
    };
    add([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    add([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
    add([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
    add([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    ply::write_ply(
        std::env::args().nth(1).unwrap_or("axes.ply".into()),
        &gs,
        PlyFormat::Standard,
        1.0,
    )
    .unwrap();
}
