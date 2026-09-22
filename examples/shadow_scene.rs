// Renders a floor with a floating occluder lit by the point light, to eyeball shadows.
use glam::{Mat4, Vec3, Vec4};
use mesh2splat::camera::Camera;
use mesh2splat::gpu::*;
use mesh2splat::scene::{Material, Mesh, Scene, Vertex};
use mesh2splat::types::BBox;

fn quad(offset: Vec3, size: f32, color: Vec4) -> Mesh {
    let n = Vec3::Y;
    let p = |x: f32, z: f32| offset + Vec3::new(x, 0.0, -z) * size;
    let v = |p: Vec3| Vertex {
        position: p.extend(1.0).to_array(),
        normal: n.extend(0.0).to_array(),
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv: [0.0; 4],
    };
    let vertices = vec![
        v(p(0.0, 0.0)),
        v(p(1.0, 0.0)),
        v(p(1.0, 1.0)),
        v(p(0.0, 0.0)),
        v(p(1.0, 1.0)),
        v(p(0.0, 1.0)),
    ];
    let mut bbox = BBox::EMPTY;
    vertices
        .iter()
        .for_each(|v| bbox.grow(Vec3::from_slice(&v.position[..3])));
    Mesh {
        name: "q".into(),
        vertices,
        material: Material {
            base_color_factor: color,
            ..Default::default()
        },
        bbox,
        surface_area: 1.0,
    }
}

fn main() {
    let ctx = GpuContext::new_headless().unwrap();
    let meshes = vec![
        quad(
            Vec3::new(-2.0, 0.0, 2.0),
            4.0,
            Vec4::new(0.8, 0.8, 0.8, 1.0),
        ),
        quad(
            Vec3::new(-0.4, 0.8, 0.4),
            0.8,
            Vec4::new(0.9, 0.3, 0.2, 1.0),
        ),
    ];
    let bbox = meshes.iter().fold(BBox::EMPTY, |a, m| a.union(&m.bbox));
    let scene = Scene { meshes, bbox };
    let gpu = GpuScene::upload(&ctx, &scene);
    let mut conv = Converter::new(&ctx);
    let mut gb = GaussianBuffer::new_empty(&ctx);
    conv.convert(
        &ctx,
        &gpu,
        ConvertSettings {
            resolution: 512,
            bbox_mode: BBoxMode::PerMesh,
            merge: None,
        },
        &mut gb,
    );
    let mut r = Renderer::new(&ctx);
    let mut cam = Camera::new(Vec3::new(0.0, 3.0, 4.0), Vec3::Y, -90.0, -35.0);
    cam.fov = 50.0;
    let s = RenderSettings {
        lighting: true,
        light_intensity: 12.0,
        light_transform: Mat4::from_translation(Vec3::new(0.3, 2.0, 0.3)),
        ..Default::default()
    };
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    r.render(&ctx, &mut enc, &cam, &s, &gb, Some(&gpu), (480, 360));
    ctx.queue.submit([enc.finish()]);
    let (w, h, px) = r.read_output(&ctx).unwrap();
    image::save_buffer(
        std::env::args().nth(1).unwrap_or("shadow.png".into()),
        &px,
        w,
        h,
        image::ExtendedColorType::Rgba8,
    )
    .unwrap();
}
