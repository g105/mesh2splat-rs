//! Convert a hair groom (Cem Yuksel's `.hair` strand format) into splats.
//!
//! ```text
//! cargo run --release --example groom -- assets/straight.hair [--strands 4000]
//!     [--resolution 2048] [--width-cells 1.5] [--std 0.65] [--light] [--out target/groom]
//!     [--direct [splats per segment]] [--width W] [--alpha A]
//! ```
//!
//! The converter samples triangles on a planar grid, so a strand only becomes
//! splats if its ribbon is at least about one grid cell wide — this builds each
//! strand as a ribbon of `--width-cells` cells and reports how the two compare.
//! Per-point strand colours are baked into a texture, one row per strand.
//!
//! Two ways to get splats out of a groom:
//!
//! * the **mesh pipeline** (default) rasterizes the ribbons on the conversion
//!   grid, which gives round cell-sized splats — the strand direction is lost
//!   and long strands cost a great many splats;
//! * `--direct` places one splat per strand segment instead, oriented by the
//!   segment's tangent and shaped like it (long along the strand, thin across,
//!   flat). This is what "strand aligned" means here, and it needs roughly two
//!   orders of magnitude fewer splats.
//!
//! Hair models: https://www.cemyuksel.com/research/hairmodels (free for
//! personal and research use).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use glam::{Vec3, Vec4};
use mesh2splat::camera::Camera;
use mesh2splat::gpu::*;
use mesh2splat::scene::{Material, Mesh, Scene, TextureData, Vertex};
use mesh2splat::types::BBox;
use mesh2splat::{ply, PlyFormat};

/// Rows of the strand colour texture; GPUs cap textures at 16384.
const MAX_TEXTURE_ROWS: u32 = 16384;

/// One strand: a polyline with a colour per point.
struct Strand {
    points: Vec<Vec3>,
    colors: Vec<Vec3>,
}

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

/// A loaded groom plus the file's own default strand thickness.
struct Groom {
    strands: Vec<Strand>,
    thickness: f32,
}

/// Read the binary `.hair` format: a 128-byte header, then the arrays the
/// flags mark as present (segments, points, thickness, transparency, colour).
fn load_hair(path: &PathBuf) -> Result<Groom> {
    let data = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    if data.len() < 128 || &data[..4] != b"HAIR" {
        bail!("{} is not a .hair file", path.display());
    }
    let strand_count = u32_at(&data, 4) as usize;
    let point_count = u32_at(&data, 8) as usize;
    let flags = u32_at(&data, 12);
    let default_segments = u32_at(&data, 16) as usize;
    let default_thickness = f32_at(&data, 20);
    let default_color = Vec3::new(f32_at(&data, 28), f32_at(&data, 32), f32_at(&data, 36));
    let (has_segments, has_points) = (flags & 1 != 0, flags & 2 != 0);
    let (has_thickness, has_transparency) = (flags & 4 != 0, flags & 8 != 0);
    let has_color = flags & 16 != 0;
    if !has_points {
        bail!("{} has no point array", path.display());
    }

    let mut off = 128;
    let segments: Vec<usize> = if has_segments {
        let s = (0..strand_count)
            .map(|i| u16::from_le_bytes(data[off + i * 2..off + i * 2 + 2].try_into().unwrap()) as usize)
            .collect();
        off += strand_count * 2;
        s
    } else {
        vec![default_segments; strand_count]
    };
    let points_at = off;
    off += point_count * 12;
    if has_thickness {
        off += point_count * 4;
    }
    if has_transparency {
        off += point_count * 4;
    }
    let colors_at = has_color.then_some(off);

    let mut strands = Vec::with_capacity(strand_count);
    let mut p = 0usize;
    for seg in segments {
        let n = seg + 1;
        let mut points = Vec::with_capacity(n);
        let mut colors = Vec::with_capacity(n);
        for k in 0..n {
            let o = points_at + (p + k) * 12;
            points.push(Vec3::new(f32_at(&data, o), f32_at(&data, o + 4), f32_at(&data, o + 8)));
            colors.push(match colors_at {
                Some(c) => {
                    let o = c + (p + k) * 12;
                    Vec3::new(f32_at(&data, o), f32_at(&data, o + 4), f32_at(&data, o + 8))
                }
                None => default_color,
            });
        }
        p += n;
        strands.push(Strand { points, colors });
    }
    Ok(Groom {
        strands,
        thickness: default_thickness,
    })
}

/// Colours of every strand as a texture: one row per strand, one column per
/// point, so a ribbon's `u` runs along the strand and `v` picks its row.
fn strand_texture(strands: &[Strand]) -> TextureData {
    let width = strands.iter().map(|s| s.points.len()).max().unwrap_or(1) as u32;
    // Wrap around the maximum texture size; neighbouring strands share a row.
    let height = (strands.len() as u32).min(MAX_TEXTURE_ROWS);
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for (row, s) in strands.iter().enumerate() {
        let row = row % height as usize;
        for x in 0..width as usize {
            let c = s.colors[x.min(s.colors.len() - 1)];
            let o = (row * width as usize + x) * 4;
            rgba[o] = (c.x.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[o + 1] = (c.y.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[o + 2] = (c.z.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[o + 3] = 255;
        }
    }
    TextureData {
        width,
        height,
        rgba,
        id: 0,
    }
}

/// Build one ribbon per strand: two triangles per segment, `width` across,
/// turned to face away from `center` so the groom reads from every side.
fn ribbons(strands: &[Strand], width: f32, center: Vec3) -> Mesh {
    let cols = strands.iter().map(|s| s.points.len()).max().unwrap_or(1) as f32;
    let mut vertices = Vec::new();
    let mut bbox = BBox::EMPTY;
    let mut area = 0.0;
    let rows = (strands.len() as u32).min(MAX_TEXTURE_ROWS) as f32;
    for (row, s) in strands.iter().enumerate() {
        let v = ((row % rows as usize) as f32 + 0.5) / rows;
        for i in 0..s.points.len().saturating_sub(1) {
            let (p0, p1) = (s.points[i], s.points[i + 1]);
            let tangent = (p1 - p0).normalize_or(Vec3::Y);
            let radial = (p0 - center).normalize_or(Vec3::Z);
            let side = tangent.cross(radial).normalize_or(Vec3::X) * (width * 0.5);
            let normal = side.normalize_or(Vec3::X).cross(tangent);
            let u = |k: usize| (k as f32 + 0.5) / cols;
            let corner = |p: Vec3, s: Vec3, k: usize, vv: f32| Vertex {
                position: (p + s).extend(1.0).to_array(),
                normal: normal.extend(0.0).to_array(),
                tangent: tangent.extend(1.0).to_array(),
                uv: [u(k), vv, 0.0, 0.0],
            };
            let quad = [
                corner(p0, -side, i, v),
                corner(p1, -side, i + 1, v),
                corner(p1, side, i + 1, v),
                corner(p0, -side, i, v),
                corner(p1, side, i + 1, v),
                corner(p0, side, i, v),
            ];
            for c in quad {
                bbox.grow(Vec3::from_slice(&c.position[..3]));
                vertices.push(c);
            }
            area += (p1 - p0).length() * width;
        }
    }
    Mesh {
        name: "groom".into(),
        vertices,
        material: Material {
            name: "hair".into(),
            base_color_factor: Vec4::ONE,
            metallic_factor: 0.0,
            roughness_factor: 0.6,
            base_color_texture: Some(Arc::new(strand_texture(strands))),
            ..Default::default()
        },
        bbox,
        surface_area: area,
    }
}

/// Median ratio of a splat's two in-plane standard deviations. Strand-aligned
/// splats are long along the strand and thin across it, so this is >> 1.
fn median_anisotropy(splats: &[mesh2splat::GaussianVertex]) -> f32 {
    let mut ratios: Vec<f32> = splats
        .iter()
        .map(|g| {
            let mut s = [g.scale[0], g.scale[1], g.scale[2]];
            s.sort_by(f32::total_cmp);
            // s[0] is the flat axis; compare the two that span the surface.
            s[2] / s[1].max(1e-12)
        })
        .collect();
    if ratios.is_empty() {
        return 0.0;
    }
    ratios.sort_by(f32::total_cmp);
    ratios[ratios.len() / 2]
}

/// One splat per piece of strand, shaped and oriented like that piece:
/// long along the tangent, `width` across, flat in the third axis.
fn strand_splats(
    strands: &[Strand],
    width: f32,
    center: Vec3,
    per_segment: usize,
    alpha: f32,
) -> Vec<mesh2splat::GaussianVertex> {
    let mut out = Vec::new();
    let k = per_segment.max(1);
    for s in &strands.iter().collect::<Vec<_>>() {
        for i in 0..s.points.len() - 1 {
            let (p0, p1) = (s.points[i], s.points[i + 1]);
            let (c0, c1) = (s.colors[i], s.colors[i + 1]);
            for j in 0..k {
                let (t0, t1) = (j as f32 / k as f32, (j + 1) as f32 / k as f32);
                let (a, b) = (p0.lerp(p1, t0), p0.lerp(p1, t1));
                let mid = (a + b) * 0.5;
                let tangent = (b - a).normalize_or(Vec3::Y);
                let radial = (mid - center).normalize_or(Vec3::Z);
                let side = tangent.cross(radial).normalize_or(Vec3::X);
                let normal = side.cross(tangent).normalize_or(Vec3::Z);
                // Columns of the rotation are the splat's axes, in scale order.
                let q = glam::Quat::from_mat3(&glam::Mat3::from_cols(tangent, side, normal))
                    .normalize();
                let color = (c0.lerp(c1, (t0 + t1) * 0.5)).clamp(Vec3::ZERO, Vec3::ONE);
                out.push(mesh2splat::GaussianVertex {
                    position: mid.extend(1.0).to_array(),
                    color: color.extend(alpha).to_array(),
                    // Half-length along the strand: neighbouring splats overlap.
                    scale: [
                        (b - a).length() * 0.5,
                        width * 0.5,
                        (width * 0.05).max(1e-6),
                        0.0,
                    ],
                    normal: normal.extend(0.0).to_array(),
                    rotation: [q.w, q.x, q.y, q.z],
                    pbr: [0.05, 0.5, 0.0, 1.0],
                });
            }
        }
    }
    out
}

struct Args {
    file: PathBuf,
    strands: usize,
    resolution: u32,
    width_cells: f32,
    std: f32,
    /// Strand width in model units; overrides the grid- or file-derived value.
    width: Option<f32>,
    alpha: f32,
    light: bool,
    opacity_shadows: bool,
    forward: bool,
    transmission: f32,
    out: PathBuf,
    /// Splats per strand segment; `None` uses the mesh pipeline.
    direct: Option<usize>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        file: "assets/straight.hair".into(),
        strands: 4000,
        resolution: 2048,
        width_cells: 1.5,
        std: 0.65,
        width: None,
        alpha: 1.0,
        light: false,
        opacity_shadows: true,
        forward: false,
        transmission: 0.6,
        out: "target/groom".into(),
        direct: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--strands" => a.strands = val()?.parse()?,
            "--resolution" => a.resolution = val()?.parse()?,
            "--width-cells" => a.width_cells = val()?.parse()?,
            "--std" => a.std = val()?.parse()?,
            "--width" => a.width = Some(val()?.parse()?),
            "--alpha" => a.alpha = val()?.parse()?,
            "--light" => a.light = true,
            "--depth-shadows" => a.opacity_shadows = false,
            "--forward" => a.forward = true,
            "--transmission" => a.transmission = val()?.parse()?,
            "--direct" => {
                let n = it.next().and_then(|v| v.parse().ok()).unwrap_or(2);
                a.direct = Some(n);
            }
            "--out" => a.out = val()?.into(),
            _ => a.file = arg.into(),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    env_logger::init();
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;

    let groom = load_hair(&args.file)?;
    let step = (groom.strands.len() / args.strands.max(1)).max(1);
    let strands: Vec<Strand> = groom
        .strands
        .into_iter()
        .step_by(step)
        .map(|s| Strand {
            points: s.points,
            colors: s.colors,
        })
        .collect();
    // Grooms are modelled at their own scale (tens of units); the renderer's
    // near/far planes, shadow bias and gaussian scale all assume a unit-ish
    // model, so normalize the groom into a 2-unit box around the origin.
    let mut raw = BBox::EMPTY;
    for s in &strands {
        for p in &s.points {
            raw.grow(*p);
        }
    }
    let fit = 2.0 / raw.size().max_element().max(1e-6);
    let offset = raw.center();
    let mut strands: Vec<Strand> = strands
        .into_iter()
        .map(|s| Strand {
            points: s.points.iter().map(|p| (*p - offset) * fit).collect(),
            colors: s.colors,
        })
        .collect();
    strands.retain(|s| s.points.len() > 1);
    let mut bbox = BBox::EMPTY;
    for s in &strands {
        for p in &s.points {
            bbox.grow(*p);
        }
    }
    let size = bbox.size();
    // Grid cells are square in the two axes of the dominant-normal projection,
    // so the widest extent sets the cell size.
    let cell = size.max_element() / args.resolution as f32;
    // The mesh pipeline needs ribbons about a cell wide to be sampled at all;
    // strand-aligned splats are free to use the groom's real thickness.
    let width = match (args.width, args.direct) {
        (Some(w), _) => w,
        (None, Some(_)) => groom.thickness * fit,
        (None, None) => cell * args.width_cells,
    };
    let segments: usize = strands.iter().map(|s| s.points.len() - 1).sum();
    println!(
        "{}: {} strands ({} segments), bbox {:.3} x {:.3} x {:.3}",
        args.file.display(),
        strands.len(),
        segments,
        size.x,
        size.y,
        size.z
    );
    println!(
        "grid {} px -> cell {:.5}; strand width {:.5} ({:.2} cells, file default {:.5} x {:.4} scale)",
        args.resolution,
        cell,
        width,
        width / cell,
        groom.thickness,
        fit
    );

    let center = bbox.center();
    let ctx = GpuContext::new_headless()?;
    println!(
        "GPU: {} ({:?})",
        ctx.adapter_info.name, ctx.adapter_info.backend
    );
    // Strand-aligned splats do not go through the mesh pipeline at all.
    let scene = args.direct.is_none().then(|| {
        let mesh = ribbons(&strands, width, center);
        println!("{} triangles", mesh.triangle_count());
        Scene {
            bbox: mesh.bbox,
            meshes: vec![mesh],
        }
    });
    let gpu = scene.as_ref().map(|s| GpuScene::upload(&ctx, s));
    let mut gb = GaussianBuffer::new_empty(&ctx);
    let started = std::time::Instant::now();
    if let Some(per_segment) = args.direct {
        // Strand-aligned: one splat per piece of segment, no rasterization.
        let splats = strand_splats(&strands, width, center, per_segment, args.alpha);
        gb.upload_ply(&ctx, &splats, false);
        println!(
            "{} strand-aligned splats in {:.1} ms ({} per segment)",
            splats.len(),
            started.elapsed().as_secs_f64() * 1e3,
            per_segment
        );
    } else {
        let mut conv = Converter::new(&ctx);
        let stats = conv.convert(
            &ctx,
            gpu.as_ref().unwrap(),
            ConvertSettings {
                resolution: args.resolution,
                bbox_mode: BBoxMode::Scene,
                merge: None,
                detail: None,
            },
            &mut gb,
        );
        println!(
            "{} splats in {:.1} ms ({:.1} per strand, {:.2} per segment)",
            stats.gaussians,
            stats.duration.as_secs_f64() * 1e3,
            stats.gaussians as f64 / strands.len() as f64,
            stats.gaussians as f64 / segments as f64,
        );
        if stats.fragments > stats.capacity {
            println!(
                "warning: {} splats dropped (capacity {}); use fewer --strands or a smaller --resolution",
                stats.fragments - stats.capacity,
                stats.capacity
            );
        }
    }
    let splats = gb.download(&ctx);
    println!(
        "median in-plane anisotropy: {:.2}x (1 = round, higher = strand shaped)",
        median_anisotropy(&splats)
    );

    let name = args
        .file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let name = if args.direct.is_some() {
        format!("{name}_direct")
    } else {
        name
    };
    let ply_path = args.out.join(format!("{name}.ply"));
    ply::write_ply(&ply_path, &splats, PlyFormat::PlayCanvas, gb.scale_multiplier(args.std))?;
    println!(
        "wrote {} ({:.1} MB)",
        ply_path.display(),
        std::fs::metadata(&ply_path)?.len() as f64 / 1e6
    );

    // Renders: framed, from the side, and close in on the strands.
    let mut renderer = Renderer::new(&ctx);
    let mut framed = Camera::default();
    framed.frame_bbox(&bbox);
    let radius = bbox.size().length() * 0.5;
    let settings = RenderSettings {
        gaussian_std: args.std,
        lighting: args.light,
        // Strand splats are long and thin: shade them as fibres.
        hair_shading: true,
        opacity_shadows: args.opacity_shadows,
        forward_shading: args.forward,
        transmission: args.transmission,
        // The point light falls off with distance squared, and grooms are
        // modelled at their own scale (this one is ~90 units across).
        light_intensity: 25.0 * radius * radius,
        light_transform: glam::Mat4::from_translation(center + Vec3::new(1.0, 1.5, 1.2) * radius),
        background: [0.05, 0.05, 0.06, 1.0],
        ..Default::default()
    };
    for (label, cam) in [
        ("front", framed.clone()),
        ("side", {
            let mut c = framed.clone();
            c.tumble(90.0 / (c.mouse_sensitivity * 2.0), 0.0);
            c
        }),
        ("close", {
            let mut c = framed.clone();
            // Close, but still outside the groom.
            c.dolly(1.7f32.ln());
            c
        }),
    ] {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        // The mesh is only shown for the split-screen comparison; pass it along.
        renderer.render(&ctx, &mut enc, &cam, &settings, &gb, gpu.as_ref(), (1600, 1200));
        ctx.queue.submit([enc.finish()]);
        let (w, h, px) = renderer.read_output(&ctx).context("no output")?;
        let file = args.out.join(format!("{name}_{label}.png"));
        image::save_buffer(&file, &px, w, h, image::ExtendedColorType::Rgba8)?;
        println!("wrote {}", file.display());
    }
    Ok(())
}
