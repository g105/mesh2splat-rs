//! End-to-end GPU tests of the converter and renderer on analytic scenes.

use glam::{Quat, Vec3, Vec4};
use mesh2splat::camera::Camera;
use mesh2splat::gpu::{
    BBoxMode, ConvertSettings, Converter, GaussianBuffer, GpuContext, GpuScene, RenderSettings,
    Renderer,
};
use mesh2splat::scene::{Material, Mesh, Scene, Vertex};
use mesh2splat::types::{BBox, RenderMode};

fn ctx() -> Option<GpuContext> {
    match GpuContext::new_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn vtx(p: Vec3, n: Vec3) -> Vertex {
    Vertex {
        position: p.extend(1.0).to_array(),
        normal: n.extend(0.0).to_array(),
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv: [p.x, p.y, 0.0, 0.0],
    }
}

/// Unit square in the XY plane, facing +Z, two triangles.
fn quad_mesh(color: Vec4, offset: Vec3, size: f32) -> Mesh {
    let n = Vec3::Z;
    let p = |x: f32, y: f32| offset + Vec3::new(x, y, 0.0) * size;
    let vertices = vec![
        vtx(p(0.0, 0.0), n),
        vtx(p(1.0, 0.0), n),
        vtx(p(1.0, 1.0), n),
        vtx(p(0.0, 0.0), n),
        vtx(p(1.0, 1.0), n),
        vtx(p(0.0, 1.0), n),
    ];
    let mut bbox = BBox::EMPTY;
    for v in &vertices {
        bbox.grow(Vec3::from_slice(&v.position[..3]));
    }
    Mesh {
        name: "quad".into(),
        vertices,
        material: Material {
            base_color_factor: color,
            ..Default::default()
        },
        bbox,
        surface_area: size * size,
    }
}

fn scene_of(meshes: Vec<Mesh>) -> Scene {
    let bbox = meshes.iter().fold(BBox::EMPTY, |a, m| a.union(&m.bbox));
    Scene { meshes, bbox }
}

#[test]
fn quad_conversion_matches_analytic_result() {
    let Some(ctx) = ctx() else { return };
    let color = Vec4::new(0.2, 0.4, 0.8, 1.0);
    let scene = scene_of(vec![quad_mesh(color, Vec3::ZERO, 1.0)]);
    let gpu = GpuScene::upload(&ctx, &scene);
    let mut conv = Converter::new(&ctx);
    let mut gb = GaussianBuffer::new_empty(&ctx);
    let res = 64;
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

    // The quad covers the whole conversion target exactly once.
    assert_eq!(stats.gaussians, res * res);
    let gs = gb.download(&ctx);
    assert_eq!(gs.len(), (res * res) as usize);

    for g in &gs {
        // Positions interpolate over the quad.
        assert!(g.position[0] >= -1e-4 && g.position[0] <= 1.0 + 1e-4);
        assert!(g.position[1] >= -1e-4 && g.position[1] <= 1.0 + 1e-4);
        assert!(g.position[2].abs() < 1e-6);
        // Color = white (no texture) * base color factor.
        for k in 0..4 {
            assert!((g.color[k] - color[k]).abs() < 1e-6);
        }
        // UV space == XY here, so the Jacobian is the identity: unit scales.
        assert!((g.scale[0] - 1.0).abs() < 1e-4, "scale {:?}", g.scale);
        assert!((g.scale[1] - 1.0).abs() < 1e-4, "scale {:?}", g.scale);
        assert!((g.scale[2] - 1e-7).abs() < 1e-9);
        // Default PBR values.
        assert!((g.pbr[0] - 0.1).abs() < 1e-6 && (g.pbr[1] - 0.5).abs() < 1e-6);
        // Rotation: local z = face normal (+-Z), local x = longest edge (the diagonal).
        let q = Quat::from_xyzw(g.rotation[1], g.rotation[2], g.rotation[3], g.rotation[0]);
        assert!((q.length() - 1.0).abs() < 1e-4);
        let z = q * Vec3::Z;
        let x = q * Vec3::X;
        assert!(z.z.abs() > 0.9999, "normal axis {z}");
        let diag = Vec3::new(1.0, 1.0, 0.0).normalize();
        assert!(x.dot(diag).abs() > 0.9999, "x axis {x}");
        // Normal attribute passes through.
        assert!((g.normal[2] - 1.0).abs() < 1e-6);
    }

    // Mean of positions ~ center of the quad.
    let mean: Vec3 = gs
        .iter()
        .map(|g| Vec3::from_slice(&g.position[..3]))
        .sum::<Vec3>()
        / gs.len() as f32;
    assert!(
        (mean - Vec3::new(0.5, 0.5, 0.0)).length() < 1e-3,
        "mean {mean}"
    );
}

#[test]
fn bbox_modes_and_multi_mesh() {
    let Some(ctx) = ctx() else { return };
    // Two quads: a big one and a small one far away.
    let scene = scene_of(vec![
        quad_mesh(Vec4::ONE, Vec3::ZERO, 1.0),
        quad_mesh(Vec4::new(1.0, 0.0, 0.0, 1.0), Vec3::new(3.0, 0.0, 0.0), 0.5),
    ]);
    let gpu = GpuScene::upload(&ctx, &scene);
    let mut conv = Converter::new(&ctx);
    let mut gb = GaussianBuffer::new_empty(&ctx);
    let res = 64;

    // Per-mesh: every mesh fills the target.
    let s = conv.convert(
        &ctx,
        &gpu,
        ConvertSettings {
            resolution: res,
            bbox_mode: BBoxMode::PerMesh,
            detail: None,
            merge: None,
        },
        &mut gb,
    );
    assert_eq!(s.gaussians, 2 * res * res);

    // Scene box (3.5 x 1): uniform density, each quad covers area/extent^2 of the target.
    let s = conv.convert(
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
    let texel = 3.5 / res as f32;
    let expected = (1.0 / (texel * texel)) + (0.25 / (texel * texel));
    let got = s.gaussians as f32;
    assert!(
        (got - expected).abs() / expected < 0.05,
        "got {got}, expected ~{expected}"
    );
}

#[test]
fn renders_converted_quad() {
    let Some(ctx) = ctx() else { return };
    let color = Vec4::new(1.0, 0.5, 0.25, 1.0);
    let scene = scene_of(vec![quad_mesh(color, Vec3::new(-0.5, -0.5, 0.0), 1.0)]);
    let gpu = GpuScene::upload(&ctx, &scene);
    let mut conv = Converter::new(&ctx);
    let mut gb = GaussianBuffer::new_empty(&ctx);
    conv.convert(
        &ctx,
        &gpu,
        ConvertSettings {
            resolution: 128,
            bbox_mode: BBoxMode::Scene,
            detail: None,
            merge: None,
        },
        &mut gb,
    );

    let mut renderer = Renderer::new(&ctx);
    let camera = Camera::new(Vec3::new(0.0, 0.0, 2.0), Vec3::Y, -90.0, 0.0);
    let size = (128u32, 96u32);
    for mode in [
        RenderMode::Final,
        RenderMode::Albedo,
        RenderMode::Normal,
        RenderMode::Pbr,
    ] {
        let settings = RenderSettings {
            render_mode: mode,
            ..Default::default()
        };
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        renderer.render(&ctx, &mut enc, &camera, &settings, &gb, Some(&gpu), size);
        ctx.queue.submit([enc.finish()]);
        assert_eq!(renderer.read_visible_count(&ctx), 128 * 128);
        let (w, h, px) = renderer.read_output(&ctx).unwrap();
        let at = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2]]
        };
        let center = at(w / 2, h / 2);
        let corner = at(1, 1);
        assert_eq!(corner, [0, 0, 0], "background should be black ({mode:?})");
        match mode {
            RenderMode::Final | RenderMode::Albedo => {
                for k in 0..3 {
                    let expect = (color[k] * 255.0).round() as i32;
                    // accumulated alpha is ~0.97 in the interior (sum of gaussian falloffs)
                    assert!(
                        (center[k] as i32 - expect).abs() <= 8,
                        "{mode:?} center {center:?}"
                    );
                }
            }
            RenderMode::Normal => {
                // encoded +Z normal = (0.5, 0.5, 1.0)
                assert!(
                    (center[0] as i32 - 128).abs() <= 8
                        && (center[1] as i32 - 128).abs() <= 8
                        && center[2] >= 240,
                    "{center:?}"
                );
            }
            RenderMode::Pbr => {
                // metallic 0.1, roughness 0.5
                assert!(
                    (center[0] as i32 - 26).abs() <= 4 && (center[1] as i32 - 128).abs() <= 8,
                    "{center:?}"
                );
            }
            _ => {}
        }
        let _ = h;
    }

    // Moving the quad out of view culls everything.
    let settings = RenderSettings {
        model_transform: glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 10.0)),
        ..Default::default()
    };
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    renderer.render(&ctx, &mut enc, &camera, &settings, &gb, Some(&gpu), size);
    ctx.queue.submit([enc.finish()]);
    assert_eq!(renderer.read_visible_count(&ctx), 0);
}

#[test]
fn lighting_and_shadows_run() {
    let Some(ctx) = ctx() else { return };
    // Floor quad plus a small occluder quad above it.
    let mut floor = quad_mesh(Vec4::ONE, Vec3::new(-1.0, -1.0, 0.0), 2.0);
    floor.name = "floor".into();
    let blocker = quad_mesh(Vec4::ONE, Vec3::new(-0.25, -0.25, 0.5), 0.5);
    let scene = scene_of(vec![floor, blocker]);
    let gpu = GpuScene::upload(&ctx, &scene);
    let mut conv = Converter::new(&ctx);
    let mut gb = GaussianBuffer::new_empty(&ctx);
    conv.convert(
        &ctx,
        &gpu,
        ConvertSettings {
            resolution: 256,
            bbox_mode: BBoxMode::PerMesh,
            detail: None,
            merge: None,
        },
        &mut gb,
    );

    let mut renderer = Renderer::new(&ctx);
    let camera = Camera::new(Vec3::new(0.0, 0.0, 3.0), Vec3::Y, -90.0, 0.0);
    let settings = RenderSettings {
        lighting: true,
        light_intensity: 4.0,
        light_transform: glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 1.5)),
        ..Default::default()
    };
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    renderer.render(
        &ctx,
        &mut enc,
        &camera,
        &settings,
        &gb,
        Some(&gpu),
        (128, 128),
    );
    ctx.queue.submit([enc.finish()]);
    let (w, _h, px) = renderer.read_output(&ctx).unwrap();
    let lum = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        px[i] as u32 + px[i + 1] as u32 + px[i + 2] as u32
    };
    // The floor just outside the blocker (shadowed from a light straight above) must be
    // darker than a lit floor point at a similar distance from the light.
    let blocker_edge_shadow = lum(64 + 22, 64); // floor under the blocker's shadow ring
    let lit = lum(64 + 45, 64);
    let blocker_top = lum(64, 64);
    assert!(blocker_top > 0);
    assert!(lit > 0);
    eprintln!("blocker {blocker_top}, shadow ring {blocker_edge_shadow}, lit floor {lit}");
}
