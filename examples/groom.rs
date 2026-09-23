//! Convert a hair groom (Cem Yuksel's `.hair` strand format) into splats.
//!
//! ```text
//! cargo run --release --example groom -- assets/straight.hair [--strands 10000]
//!     [--per-segment 2] [--width W] [--alpha A] [--light] [--forward]
//!     [--ribbons] [--resolution 2048] [--width-cells 1.5]
//!     [--bench 60] [--out target/groom]
//! ```
//!
//! By default each strand segment becomes a splat oriented and shaped like that
//! segment ("strand aligned"). `--ribbons` instead builds ribbon geometry and
//! pushes it through the normal mesh pipeline, which samples it on the
//! conversion grid: the splats come out round and cell-sized, the strand
//! direction is lost, and long strands cost a great many splats.
//!
//! `--bench N` renders N frames per view with deferred and with forward
//! (per-splat) shading and reports the GPU cost of each.
//!
//! Hair models: <https://www.cemyuksel.com/research/hairmodels> (free for
//! personal and research use).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use glam::{Vec3, Vec4};
use mesh2splat::camera::Camera;
use mesh2splat::gpu::ao::{self, AoSettings};
use mesh2splat::gpu::*;
use mesh2splat::hair::{Groom, Strand, StrandSplats};
use mesh2splat::scene::{Material, Mesh, Scene, TextureData, Vertex};
use mesh2splat::types::BBox;
use mesh2splat::{ply, GaussianVertex, PlyFormat};

/// Rows of the strand colour texture; GPUs cap textures at 16384.
const MAX_TEXTURE_ROWS: u32 = 16384;

/// Colours of every strand as a texture: one row per strand, one column per
/// point, so a ribbon's `u` runs along the strand and `v` picks its row.
fn strand_texture(strands: &[Strand]) -> TextureData {
    let width = strands.iter().map(|s| s.points.len()).max().unwrap_or(1) as u32;
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

/// One ribbon per strand: two triangles per segment, `width` across, turned to
/// face away from `center` so the groom reads from every side.
fn ribbons(strands: &[Strand], width: f32, center: Vec3) -> Mesh {
    let cols = strands.iter().map(|s| s.points.len()).max().unwrap_or(1) as f32;
    let rows = (strands.len() as u32).min(MAX_TEXTURE_ROWS) as f32;
    let mut vertices = Vec::new();
    let mut bbox = BBox::EMPTY;
    let mut area = 0.0;
    for (row, s) in strands.iter().enumerate() {
        let v = ((row % rows as usize) as f32 + 0.5) / rows;
        for i in 0..s.points.len().saturating_sub(1) {
            let (p0, p1) = (s.points[i], s.points[i + 1]);
            let tangent = (p1 - p0).normalize_or(Vec3::Y);
            let radial = (p0 - center).normalize_or(Vec3::Z);
            let side = tangent.cross(radial).normalize_or(Vec3::X) * (width * 0.5);
            let normal = side.normalize_or(Vec3::X).cross(tangent);
            let u = |k: usize| (k as f32 + 0.5) / cols;
            let corner = |p: Vec3, s: Vec3, k: usize| Vertex {
                position: (p + s).extend(1.0).to_array(),
                normal: normal.extend(0.0).to_array(),
                tangent: tangent.extend(1.0).to_array(),
                uv: [u(k), v, 0.0, 0.0],
            };
            for c in [
                corner(p0, -side, i),
                corner(p1, -side, i + 1),
                corner(p1, side, i + 1),
                corner(p0, -side, i),
                corner(p1, side, i + 1),
                corner(p0, side, i),
            ] {
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
fn median_anisotropy(splats: &[GaussianVertex]) -> f32 {
    let mut ratios: Vec<f32> = splats
        .iter()
        .map(|g| {
            let mut s = [g.scale[0], g.scale[1], g.scale[2]];
            s.sort_by(f32::total_cmp);
            s[2] / s[1].max(1e-12)
        })
        .collect();
    if ratios.is_empty() {
        return 0.0;
    }
    ratios.sort_by(f32::total_cmp);
    ratios[ratios.len() / 2]
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

struct Args {
    file: PathBuf,
    strands: usize,
    per_segment: usize,
    width: Option<f32>,
    alpha: f32,
    light: bool,
    forward: bool,
    opacity_shadows: bool,
    transmission: f32,
    ribbons: bool,
    resolution: u32,
    width_cells: f32,
    std: f32,
    bench: usize,
    no_ao: bool,
    merge_occluded: Option<f32>,
    out: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        file: "assets/straight.hair".into(),
        strands: 10000,
        per_segment: 2,
        width: None,
        alpha: 0.85,
        light: false,
        forward: false,
        opacity_shadows: true,
        transmission: 0.6,
        ribbons: false,
        resolution: 2048,
        width_cells: 1.5,
        std: 0.65,
        bench: 0,
        no_ao: false,
        merge_occluded: None,
        out: "target/groom".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--strands" => a.strands = val()?.parse()?,
            "--per-segment" | "--direct" => a.per_segment = val()?.parse()?,
            "--width" => a.width = Some(val()?.parse()?),
            "--alpha" => a.alpha = val()?.parse()?,
            "--light" => a.light = true,
            "--forward" => a.forward = true,
            "--depth-shadows" => a.opacity_shadows = false,
            "--transmission" => a.transmission = val()?.parse()?,
            "--ribbons" => a.ribbons = true,
            "--resolution" => a.resolution = val()?.parse()?,
            "--width-cells" => a.width_cells = val()?.parse()?,
            "--std" => a.std = val()?.parse()?,
            "--bench" => a.bench = val()?.parse()?,
            "--no-ao" => a.no_ao = true,
            "--merge-occluded" => a.merge_occluded = Some(val()?.parse()?),
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

    // Grooms are modelled at their own scale (tens of units); the renderer
    // assumes a unit-ish model, so fit this one into a 2-unit box.
    let groom = Groom::load(&args.file)?
        .subsampled(args.strands)
        .normalized(2.0);
    let bbox = groom.bbox();
    let cell = bbox.size().max_element() / args.resolution as f32;
    let width = args.width.unwrap_or(if args.ribbons {
        // Ribbons thinner than a grid cell are not sampled at all.
        cell * args.width_cells
    } else {
        groom.thickness
    });
    println!(
        "{}: {} strands ({} segments), strand width {:.5} ({:.2} grid cells at {} px)",
        args.file.display(),
        groom.strands.len(),
        groom.segment_count(),
        width,
        width / cell,
        args.resolution
    );

    let ctx = GpuContext::new_headless()?;
    println!(
        "GPU: {} ({:?})",
        ctx.adapter_info.name, ctx.adapter_info.backend
    );
    let mut gb = GaussianBuffer::new_empty(&ctx);
    let scene = args.ribbons.then(|| {
        let mesh = ribbons(&groom.strands, width, bbox.center());
        println!("{} triangles", mesh.triangle_count());
        Scene {
            bbox: mesh.bbox,
            meshes: vec![mesh],
        }
    });
    let gpu = scene.as_ref().map(|s| GpuScene::upload(&ctx, s));
    let started = std::time::Instant::now();
    if let Some(gpu) = &gpu {
        let stats = Converter::new(&ctx).convert(
            &ctx,
            gpu,
            ConvertSettings {
                resolution: args.resolution,
                ..Default::default()
            },
            &mut gb,
        );
        println!(
            "{} splats from the mesh pipeline in {:.1} ms ({:.2} per segment)",
            stats.gaussians,
            stats.duration.as_secs_f64() * 1e3,
            stats.gaussians as f64 / groom.segment_count() as f64
        );
        if stats.fragments > stats.capacity {
            println!(
                "warning: {} splats dropped (capacity {}); use fewer --strands or a smaller --resolution",
                stats.fragments - stats.capacity,
                stats.capacity
            );
        }
    } else {
        let splats = groom.splats(&StrandSplats {
            width: Some(width),
            per_segment: args.per_segment,
            alpha: args.alpha,
        });
        gb.upload_ply(&ctx, &splats, false);
        println!(
            "{} strand-aligned splats in {:.1} ms ({} per segment)",
            splats.len(),
            started.elapsed().as_secs_f64() * 1e3,
            args.per_segment
        );
    }
    if !args.no_ao {
        // Bake occlusion and a bent normal into the splats' spare channels.
        let t = std::time::Instant::now();
        ao::AoBaker::new(&ctx).bake(&ctx, &gb, &bbox, &AoSettings::default());
        ctx.wait_idle();
        println!("baked occlusion in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    if let Some(occlusion) = args.merge_occluded {
        // Interior strands only carry bulk opacity: pool them into coarse
        // splats and leave the visible shell alone.
        let splats = gb.download(&ctx);
        let t = std::time::Instant::now();
        let (merged, stats) = mesh2splat::merge::merge_occluded(
            &splats,
            &mesh2splat::merge::VolumeMergeSettings {
                occlusion,
                ..Default::default()
            },
        );
        gb.upload_ply(&ctx, &merged, false);
        println!(
            "occlusion merge: {} -> {} splats ({:.2}x fewer, {} clusters) in {:.1} ms",
            stats.input,
            stats.output,
            stats.input as f64 / stats.output.max(1) as f64,
            stats.merged_per_level.first().copied().unwrap_or(0),
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let splats = gb.download(&ctx);
    let mean_ao = splats.iter().map(|g| g.pbr[2] as f64).sum::<f64>() / splats.len().max(1) as f64;
    println!("mean occlusion {:.2} (1 = fully open)", mean_ao);
    let widths: Vec<f32> = splats.iter().map(|g| g.scale[1] * 2.0).collect();
    let mean_width = widths.iter().sum::<f32>() / widths.len().max(1) as f32;
    let max_width = widths.iter().copied().fold(0.0f32, f32::max);
    println!(
        "splat width: mean {:.4}, max {:.4} (per-point thickness from the file)",
        mean_width, max_width
    );
    println!(
        "median in-plane anisotropy: {:.2}x (1 = round, higher = strand shaped)",
        median_anisotropy(&splats)
    );

    let stem = args
        .file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let name = if args.ribbons {
        format!("{stem}_ribbons")
    } else {
        stem
    };
    let ply_path = args.out.join(format!("{name}.ply"));
    ply::write_ply(
        &ply_path,
        &splats,
        PlyFormat::PlayCanvas,
        gb.scale_multiplier(args.std),
    )?;
    println!(
        "wrote {} ({:.1} MB)",
        ply_path.display(),
        std::fs::metadata(&ply_path)?.len() as f64 / 1e6
    );

    let mut renderer = Renderer::new(&ctx);
    let mut framed = Camera::default();
    framed.frame_bbox(&bbox);
    let radius = bbox.size().length() * 0.5;
    let settings = RenderSettings {
        gaussian_std: args.std,
        lighting: args.light,
        hair_shading: true,
        opacity_shadows: args.opacity_shadows,
        transmission: args.transmission,
        forward_shading: args.forward,
        // The point light falls off with distance squared.
        light_intensity: 25.0 * radius * radius,
        light_transform: glam::Mat4::from_translation(
            bbox.center() + Vec3::new(1.0, 1.5, 1.2) * radius,
        ),
        background: [0.05, 0.05, 0.06, 1.0],
        ..Default::default()
    };
    let views = [
        ("front", framed.clone()),
        ("side", {
            let mut c = framed.clone();
            c.tumble(90.0 / (c.mouse_sensitivity * 2.0), 0.0);
            c
        }),
        ("close", {
            let mut c = framed.clone();
            c.dolly(1.7f32.ln());
            c
        }),
    ];

    if args.bench > 0 {
        // Forward shading moves lighting from once per pixel to once per
        // fragment, which hair's overdraw makes expensive: measure it.
        println!(
            "\n{:<8} {:>11} {:>11} {:>11} {:>11}",
            "view", "deferred", "forward", "raster def", "raster fwd"
        );
        for (name, cam) in &views {
            let mut times = [(0.0, 0.0); 2];
            for (i, forward) in [false, true].into_iter().enumerate() {
                let s = RenderSettings {
                    forward_shading: forward,
                    lighting: true,
                    ..settings.clone()
                };
                let (mut frame, mut raster) = (vec![], vec![]);
                let (mut pre, mut sort) = (vec![], vec![]);
                for f in 0..args.bench + 5 {
                    let mut enc = ctx.device.create_command_encoder(&Default::default());
                    renderer.render(&ctx, &mut enc, cam, &s, &gb, gpu.as_ref(), (1600, 1200));
                    ctx.queue.submit([enc.finish()]);
                    renderer.after_submit(&ctx);
                    ctx.wait_idle();
                    renderer.after_submit(&ctx);
                    if f < 5 {
                        continue; // warm-up
                    }
                    let st = renderer.stats();
                    frame.extend(st.gpu_ms);
                    raster.extend(st.stages.map(|s| s.splat));
                    pre.extend(st.stages.map(|s| s.prepass));
                    sort.extend(st.stages.map(|s| s.sort));
                }
                if i == 0 {
                    println!(
                        "  {name}: prepass {:.1} ms, sort {:.1} ms",
                        median(&mut pre),
                        median(&mut sort)
                    );
                }
                times[i] = (median(&mut frame), median(&mut raster));
            }
            println!(
                "{:<8} {:>9.2} ms {:>9.2} ms {:>9.2} ms {:>9.2} ms",
                name, times[0].0, times[1].0, times[0].1, times[1].1
            );
        }
        println!();
    }

    for (label, cam) in &views {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        renderer.render(&ctx, &mut enc, cam, &settings, &gb, gpu.as_ref(), (1600, 1200));
        ctx.queue.submit([enc.finish()]);
        let (w, h, px) = renderer.read_output(&ctx).context("no output")?;
        let file = args.out.join(format!("{name}_{label}.png"));
        image::save_buffer(&file, &px, w, h, image::ExtendedColorType::Rgba8)?;
        println!("wrote {}", file.display());
    }
    Ok(())
}
