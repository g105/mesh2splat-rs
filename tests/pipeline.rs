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
            detail: None,
            merge: None,
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
        let path = dir.join(format!(
            "m2s_box_{}_{:?}.{}",
            std::process::id(),
            fmt,
            fmt.extension()
        ));
        let mult = gb.scale_multiplier(0.65);
        ply::write_ply(&path, &data, fmt, mult).unwrap();
        let back = ply::load_gaussian_ply(&path).unwrap();
        assert_eq!(back.gaussians.len(), data.len());
        let expected_sx = data[10].scale[0] * mult;
        // SPZ stores log scales in steps of 1/16.
        let tol = if fmt == PlyFormat::Spz { 1.0 / 32.0 + 1e-4 } else { 1e-4 };
        assert!(
            (back.gaussians[10].scale[0] / expected_sx).ln().abs() < tol,
            "{fmt:?}: {} vs {expected_sx}",
            back.gaussians[10].scale[0]
        );
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn merge_after_gpu_conversion() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let s = scene::load_gltf(BOX).unwrap();
    let gpu = GpuScene::upload(&ctx, &s);
    let mut conv = Converter::new(&ctx);
    let settings = |merge| ConvertSettings {
        resolution: 128,
        bbox_mode: BBoxMode::Scene,
        detail: None,
        merge,
    };
    let mut plain = GaussianBuffer::new_empty(&ctx);
    conv.convert(&ctx, &gpu, settings(None), &mut plain);
    let mut merged = GaussianBuffer::new_empty(&ctx);
    let stats = conv.convert(
        &ctx,
        &gpu,
        settings(Some(mesh2splat::merge::MergeSettings::from_strength(1.0))),
        &mut merged,
    );
    let m = stats.merge.expect("merge stats");
    assert_eq!(m.input, plain.count as usize);
    assert_eq!(m.output, merged.count as usize);
    // Opposite cube faces share grid cells; depth layering must still let them merge.
    assert!(merged.count * 2 < plain.count, "{m:?}");

    // Merged splats stay on the surface, and the covered area is preserved
    // (sum of the two in-plane scales' product).
    let area = |g: &[mesh2splat::GaussianVertex]| -> f64 {
        g.iter()
            .map(|g| {
                let mut s = [g.scale[0], g.scale[1], g.scale[2]];
                s.sort_by(f32::total_cmp);
                (s[1] * s[2]) as f64
            })
            .sum()
    };
    let (a, b) = (plain.download(&ctx), merged.download(&ctx));
    for g in &b {
        let p = glam::Vec3::from_slice(&g.position[..3]);
        assert!((p.abs().max_element() - 0.5).abs() < 1e-3, "{p}");
    }
    let (aa, ab) = (area(&a), area(&b));
    assert!((aa - ab).abs() / aa < 0.01, "{aa} vs {ab}");
}

#[test]
fn gpu_merge_matches_cpu_merge() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let s = scene::load_gltf(BOX).unwrap();
    let gpu = GpuScene::upload(&ctx, &s);
    let mut conv = Converter::new(&ctx);
    for strength in [0.0, 0.25, 1.0] {
        let settings = ConvertSettings {
            resolution: 256,
            bbox_mode: BBoxMode::Scene,
            detail: None,
            merge: Some(mesh2splat::merge::MergeSettings::from_strength(strength)),
        };
        let mut runs = Vec::new();
        for gpu_merge in [false, true] {
            conv.gpu_merge = gpu_merge;
            let mut gb = GaussianBuffer::new_empty(&ctx);
            let stats = conv.convert(&ctx, &gpu, settings, &mut gb);
            runs.push(stats.merge.unwrap());
        }
        let (cpu, gpu) = (&runs[0], &runs[1]);
        // f32 on the GPU vs f64 on the CPU can flip blocks sitting exactly on a
        // tolerance, nothing more.
        let diff = cpu.output.abs_diff(gpu.output) as f64 / cpu.output as f64;
        assert!(diff < 0.005, "strength {strength}: cpu {cpu:?} gpu {gpu:?}");
    }
}

#[test]
fn detail_aware_density() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let s = scene::load_gltf(BOX).unwrap();
    let gpu = GpuScene::upload(&ctx, &s);
    let mut conv = Converter::new(&ctx);
    let settings = |detail| ConvertSettings {
        resolution: 128,
        bbox_mode: BBoxMode::Scene,
        merge: None,
        detail,
    };
    let mut count = |detail| {
        let mut gb = GaussianBuffer::new_empty(&ctx);
        let stats = conv.convert(&ctx, &gpu, settings(detail), &mut gb);
        (stats.gaussians, gb.download(&ctx))
    };
    use mesh2splat::gpu::converter::DetailSettings;
    let (plain, _) = count(None);
    // Zero tolerance keeps every triangle at level 0.
    let (exact, _) = count(Some(DetailSettings::from_strength(0.0)));
    assert_eq!(exact, plain);
    // A loose tolerance coarsens the flat cube faces by one level: a quarter
    // of the splats, each twice as wide.
    let (coarse, splats) = count(Some(DetailSettings::from_strength(1.0)));
    assert!(
        (coarse as f64 - plain as f64 / 4.0).abs() / plain as f64 * 4.0 < 0.05,
        "{coarse} vs {plain} / 4"
    );
    let width = |g: &[mesh2splat::GaussianVertex]| {
        g.iter()
            .map(|g| g.scale[0].max(g.scale[1]))
            .fold(0.0f32, f32::max)
    };
    let (_, fine) = count(None);
    assert!((width(&splats) - 2.0 * width(&fine)).abs() < 1e-3);
    // Coarser splats still sit on the cube surface.
    for g in &splats {
        let p = glam::Vec3::from_slice(&g.position[..3]);
        assert!((p.abs().max_element() - 0.5).abs() < 1e-3, "{p}");
    }
}

#[test]
fn baked_occlusion_is_lower_inside_a_cloud() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    // A solid block of splats: the middle is buried, the corners are not.
    let n = 12;
    let mut splats = Vec::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = glam::Vec3::new(x as f32, y as f32, z as f32) / n as f32 - 0.5;
                splats.push(mesh2splat::GaussianVertex {
                    position: p.extend(1.0).to_array(),
                    color: [0.5, 0.5, 0.5, 1.0],
                    scale: [0.05, 0.05, 0.01, 0.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    rotation: [1.0, 0.0, 0.0, 0.0],
                    pbr: [0.0, 0.5, 0.0, 1.0],
                });
            }
        }
    }
    let mut gb = GaussianBuffer::new_empty(&ctx);
    gb.upload_ply(&ctx, &splats, false);
    let bounds = mesh2splat::types::BBox {
        min: glam::Vec3::splat(-0.5),
        max: glam::Vec3::splat(0.5),
    };
    mesh2splat::gpu::ao::AoBaker::new(&ctx).bake(
        &ctx,
        &gb,
        &bounds,
        &mesh2splat::gpu::ao::AoSettings::default(),
    );
    let baked = gb.download(&ctx);
    assert_eq!(baked.len(), splats.len());

    let center = |g: &mesh2splat::GaussianVertex| {
        glam::Vec3::from_slice(&g.position[..3]).length()
    };
    let middle: Vec<f32> = baked
        .iter()
        .filter(|g| center(g) < 0.15)
        .map(|g| g.pbr[2])
        .collect();
    let outside: Vec<f32> = baked
        .iter()
        .filter(|g| center(g) > 0.6)
        .map(|g| g.pbr[2])
        .collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    assert!(!middle.is_empty() && !outside.is_empty());
    assert!(
        mean(&middle) < mean(&outside) * 0.8,
        "buried {:.3} vs exposed {:.3}",
        mean(&middle),
        mean(&outside)
    );
    // Every splat gets a usable bent normal.
    assert!(baked.iter().all(|g| g.pbr[3].to_bits() != 0));
}

#[test]
fn gpu_pooling_matches_the_cpu_pooler() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    // A block whose interior is marked buried, as the occlusion bake would.
    let n = 6i32;
    let mut splats = Vec::new();
    for z in -n..=n {
        for y in -n..=n {
            for x in -n..=n {
                let p = glam::Vec3::new(x as f32, y as f32, z as f32) * 0.1;
                let edge = x.abs() == n || y.abs() == n || z.abs() == n;
                splats.push(mesh2splat::GaussianVertex {
                    position: p.extend(1.0).to_array(),
                    color: [0.5, 0.4, 0.3, 0.8],
                    scale: [0.05, 0.02, 0.001, 0.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    rotation: [1.0, 0.0, 0.0, 0.0],
                    pbr: [0.0, 0.5, if edge { 0.9 } else { 0.1 }, 0.0],
                });
            }
        }
    }
    let bounds = mesh2splat::types::BBox {
        min: glam::Vec3::splat(-0.7),
        max: glam::Vec3::splat(0.7),
    };
    let cfg = mesh2splat::merge::VolumeMergeSettings {
        // Match the GPU's fixed grid so the two cluster the same way.
        cell: Some(bounds.size().x * 1.1 / 256.0 * 64.0),
        ..Default::default()
    };
    let (cpu, cpu_stats) = mesh2splat::merge::merge_occluded(&splats, &cfg);

    let mut gb = GaussianBuffer::new_empty(&ctx);
    gb.upload_ply(&ctx, &splats, false);
    let mut pooler = mesh2splat::gpu::pool::GpuPooler::new(&ctx);
    let (buffer, count, gpu_stats) = pooler.run(&ctx, &gb, &bounds, &cfg);
    gb.buffer = buffer;
    gb.count = count;
    gb.capacity = count.max(1);
    let gpu = gb.download(&ctx);

    assert!(gpu_stats.gpu && !cpu_stats.gpu);
    assert_eq!(gpu.len(), count as usize);
    assert!(count < splats.len() as u32, "nothing pooled");
    // The two cluster differently at the edges of cells, so compare in bulk.
    let ratio = gpu.len() as f64 / cpu.len() as f64;
    assert!(
        (0.8..1.25).contains(&ratio),
        "gpu {} vs cpu {} splats",
        gpu.len(),
        cpu.len()
    );
    let stops = |v: &[mesh2splat::GaussianVertex]| -> f64 {
        v.iter()
            .map(|g| {
                let mut s = [g.scale[0] as f64, g.scale[1] as f64, g.scale[2] as f64];
                s.sort_by(f64::total_cmp);
                g.color[3] as f64 * std::f64::consts::PI * s[2] * s[1]
            })
            .sum()
    };
    // A cluster denser than one blob can represent saturates at alpha 1 and
    // loses some extinction; what matters is that both paths lose the same.
    let (raw, on_gpu, on_cpu) = (stops(&splats), stops(&gpu), stops(&cpu));
    assert!(
        (on_gpu - on_cpu).abs() / on_cpu < 0.15,
        "gpu {on_gpu:.4} vs cpu {on_cpu:.4} (unpooled {raw:.4})"
    );
    assert!(on_gpu > raw * 0.6, "lost too much: {raw:.4} -> {on_gpu:.4}");
}
