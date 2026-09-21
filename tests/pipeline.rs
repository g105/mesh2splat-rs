//! glTF loading + conversion + PLY round trip on a real asset
//! (tests/assets/BoxTextured.glb, CC-BY 4.0 (c) 2017 Cesium, from KhronosGroup/glTF-Sample-Assets).

use mesh2splat::gpu::{BBoxMode, ConvertSettings, Converter, GaussianBuffer, GpuContext, GpuScene};
use mesh2splat::{ply, scene, PlyFormat};

const BOX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/BoxTextured.glb");

#[test]
fn loads_box_textured() {
    let s = scene::load_gltf(BOX).unwrap();
    assert_eq!(s.meshes.len(), 1);
    let m = &s.meshes[0];
    assert_eq!(m.triangle_count(), 12);
    assert!(m.material.base_color_texture.is_some());
    assert!(m.material.normal_texture.is_none());
    let t = m.material.base_color_texture.as_ref().unwrap();
    assert_eq!(t.rgba.len(), (t.width * t.height * 4) as usize);
    // Unit cube centered at the origin.
    assert!(
        (s.bbox.size() - glam::Vec3::ONE).length() < 1e-4,
        "{:?}",
        s.bbox
    );
    for v in &m.vertices {
        let n = glam::Vec3::from_slice(&v.normal[..3]);
        assert!((n.length() - 1.0).abs() < 1e-4);
    }
    assert!((m.surface_area - 6.0).abs() < 1e-3);
}

#[test]
fn convert_export_reload() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let s = scene::load_gltf(BOX).unwrap();
    let gpu = GpuScene::upload(&ctx, &s);
    let mut conv = Converter::new(&ctx);
    let mut gb = GaussianBuffer::new_empty(&ctx);
    let res = 128;
    let stats = conv.convert(
        &ctx,
        &gpu,
        ConvertSettings {
            resolution: res,
            bbox_mode: BBoxMode::Scene,
        },
        &mut gb,
    );
    // Each of the 6 faces is axis aligned and covers the full normalized UV square.
    assert_eq!(stats.gaussians, 6 * res * res);

    let data = gb.download(&ctx);
    // Textured: colors must vary (the texture is not a flat color).
    let first = data[0].color;
    assert!(data
        .iter()
        .any(|g| (g.color[0] - first[0]).abs() > 0.1 || (g.color[1] - first[1]).abs() > 0.1));
    // All splats sit on the cube surface.
    for g in &data {
        let p = glam::Vec3::from_slice(&g.position[..3]);
        let m = p.abs().max_element();
        assert!((m - 0.5).abs() < 1e-3, "{p}");
    }

    let dir = std::env::temp_dir();
    for fmt in PlyFormat::ALL {
        let path = dir.join(format!("m2s_box_{}_{:?}.ply", std::process::id(), fmt));
        let mult = gb.scale_multiplier(0.65);
        ply::write_ply(&path, &data, fmt, mult).unwrap();
        let back = ply::load_gaussian_ply(&path).unwrap();
        assert_eq!(back.gaussians.len(), data.len());
        let expected_sx = data[10].scale[0] * mult;
        assert!((back.gaussians[10].scale[0] - expected_sx).abs() / expected_sx < 1e-4);
        std::fs::remove_file(&path).ok();
    }
}
