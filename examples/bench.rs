//! Conversion / render / export benchmark.
//!
//! ```text
//! cargo run --release --example bench -- [model.glb] [--res 520,1024] [--size 1920x1080]
//!                                        [--frames 60] [--out target/bench] [--baseline dir]
//!                                        [--merge strength] [--cpu-merge]
//! ```
//!
//! For each sampling resolution it converts the model, renders a fixed set of
//! views and reports splat counts, median GPU times per stage and PLY sizes per
//! export format. Every view is saved as a PNG in `--out`, so
//! later changes can be compared visually; with `--baseline` it also prints the
//! PSNR against the PNG of the same name in that folder.
//!
//! `bench psnr a.png b.png [...]` just prints the PSNR of each image pair.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use mesh2splat::camera::Camera;
use mesh2splat::gpu::*;
use mesh2splat::{ply, scene, PlyFormat};

struct Args {
    model: PathBuf,
    resolutions: Vec<u32>,
    size: (u32, u32),
    frames: usize,
    out: PathBuf,
    baseline: Option<PathBuf>,
    merge: Option<f32>,
    cpu_merge: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: "assets/DamagedHelmet.glb".into(),
        resolutions: vec![520, 1024],
        size: (1920, 1080),
        frames: 60,
        out: "target/bench".into(),
        baseline: None,
        merge: None,
        cpu_merge: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--res" => {
                a.resolutions = val()?
                    .split(',')
                    .map(|r| r.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').context("--size is WxH")?;
                a.size = (w.parse()?, h.parse()?);
            }
            "--frames" => a.frames = val()?.parse()?,
            "--out" => a.out = val()?.into(),
            "--baseline" => a.baseline = Some(val()?.into()),
            "--merge" => a.merge = Some(val()?.parse()?),
            "--cpu-merge" => a.cpu_merge = true,
            _ => a.model = arg.into(),
        }
    }
    Ok(a)
}

/// Named camera placements: four orbits around the framed model plus a close-up.
fn views(framed: &Camera) -> Vec<(&'static str, Camera)> {
    let orbit = |deg: f32| {
        let mut c = framed.clone();
        c.tumble(deg / (c.mouse_sensitivity * 2.0), 0.0);
        c
    };
    let mut close = framed.clone();
    close.dolly(2.5f32.ln());
    vec![
        ("front", orbit(0.0)),
        ("right", orbit(90.0)),
        ("back", orbit(180.0)),
        ("left", orbit(270.0)),
        ("close", close),
    ]
}

/// PSNR (dB) of the RGB channels of two same-sized RGBA8 images.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let (mut se, mut n) = (0.0f64, 0usize);
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            let d = pa[c] as f64 - pb[c] as f64;
            se += d * d;
            n += 1;
        }
    }
    if se == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / (se / n as f64)).log10()
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() -> Result<()> {
    env_logger::init();
    // `bench psnr a.png b.png [c.png d.png ...]`: compare image pairs and exit.
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("psnr") {
        for pair in raw[1..].chunks_exact(2) {
            let (a, b) = (image::open(&pair[0])?.to_rgba8(), image::open(&pair[1])?.to_rgba8());
            anyhow::ensure!(a.dimensions() == b.dimensions(), "size mismatch: {pair:?}");
            println!("{:.2} dB  {} vs {}", psnr(a.as_raw(), b.as_raw()), pair[0], pair[1]);
        }
        return Ok(());
    }
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;
    let ctx = GpuContext::new_headless()?;
    println!(
        "GPU: {} ({:?}), timestamps: {}",
        ctx.adapter_info.name, ctx.adapter_info.backend, ctx.timestamps
    );
    let scene = scene::load_gltf(&args.model)?;
    let gpu_scene = GpuScene::upload(&ctx, &scene);
    println!(
        "model: {} ({} meshes, {} triangles), render {}x{}, {} frames/view\n",
        args.model.display(),
        scene.meshes.len(),
        scene.triangle_count(),
        args.size.0,
        args.size.1,
        args.frames
    );

    let mut converter = Converter::new(&ctx);
    converter.gpu_merge = !args.cpu_merge;
    let mut renderer = Renderer::new(&ctx);
    let settings = RenderSettings::default();
    let mut framed = Camera::default();
    framed.frame_bbox(&scene.bbox);

    for &res in &args.resolutions {
        let mut gaussians = GaussianBuffer::new_empty(&ctx);
        let conv = converter.convert(
            &ctx,
            &gpu_scene,
            ConvertSettings {
                resolution: res,
                merge: args.merge.map(mesh2splat::merge::MergeSettings::from_strength),
                ..Default::default()
            },
            &mut gaussians,
        );
        println!(
            "== resolution {res}: {} splats, conversion {:.1} ms",
            conv.gaussians,
            conv.duration.as_secs_f64() * 1e3
        );
        if let Some(m) = &conv.merge {
            println!(
                "   merged {} -> {} ({:.2}x fewer) in {:.1} ms on the {}, new splats per level {:?}",
                m.input,
                m.output,
                m.input as f64 / m.output.max(1) as f64,
                m.duration.as_secs_f64() * 1e3,
                if m.gpu { "GPU" } else { "CPU" },
                m.merged_per_level
            );
        }

        println!(
            "{:<6} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "view", "visible", "frame", "prepass", "sort", "raster", "wall"
        );
        for (name, cam) in views(&framed).iter() {
            let (mut frame, mut pre, mut sort, mut raster, mut wall) =
                (vec![], vec![], vec![], vec![], vec![]);
            let mut visible = 0;
            for f in 0..args.frames + 5 {
                let t0 = Instant::now();
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                renderer.render(&ctx, &mut enc, cam, &settings, &gaussians, Some(&gpu_scene), args.size);
                ctx.queue.submit([enc.finish()]);
                renderer.after_submit(&ctx);
                ctx.wait_idle();
                renderer.after_submit(&ctx);
                if f < 5 {
                    continue; // warm-up
                }
                wall.push(t0.elapsed().as_secs_f64() * 1e3);
                let s = renderer.stats();
                visible = s.visible_gaussians;
                frame.extend(s.gpu_ms);
                if let Some(st) = s.stages {
                    pre.push(st.prepass);
                    sort.push(st.sort);
                    raster.push(st.splat);
                }
            }
            println!(
                "{:<6} {:>10} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
                name,
                visible,
                median(&mut frame),
                median(&mut pre),
                median(&mut sort),
                median(&mut raster),
                median(&mut wall)
            );
            {
                let (w, h, px) = renderer.read_output(&ctx).context("no output")?;
                let file = format!("res{res}_{name}.png");
                image::save_buffer(args.out.join(&file), &px, w, h, image::ExtendedColorType::Rgba8)?;
                if let Some(dir) = &args.baseline {
                    match image::open(dir.join(&file)) {
                        Ok(base) if base.width() == w && base.height() == h => {
                            println!("       PSNR vs baseline: {:.2} dB", psnr(&px, base.to_rgba8().as_raw()));
                        }
                        _ => println!("       (no matching baseline {file})"),
                    }
                }
            }
        }

        let splats = gaussians.download(&ctx);
        let mult = gaussians.scale_multiplier(settings.gaussian_std);
        let mut sizes = Vec::new();
        for format in PlyFormat::ALL {
            let path = args.out.join("export.ply");
            let t0 = Instant::now();
            ply::write_ply(&path, &splats, format, mult)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let bytes = std::fs::metadata(&path)?.len();
            std::fs::remove_file(&path)?;
            sizes.push(format!(
                "{} {:.1} MB ({:.0} B/splat, {:.0} ms)",
                format.label(),
                bytes as f64 / 1e6,
                bytes as f64 / splats.len().max(1) as f64,
                ms
            ));
        }
        println!("export: {}\n", sizes.join(" | "));
    }
    println!("reference images in {}", args.out.display());
    Ok(())
}
