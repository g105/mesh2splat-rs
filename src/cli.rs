//! Command line interface: headless conversion (single file or batch) and offscreen rendering.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use glam::{Mat4, Vec3};

use crate::camera::Camera;
use crate::gpu::converter::resolution_from_quality;
use crate::gpu::{
    BBoxMode, ConvertSettings, Converter, GaussianBuffer, GpuContext, GpuScene, RenderSettings,
    Renderer,
};
use crate::types::{PlyFormat, RenderMode};
use crate::{ply, scene};

#[derive(Parser, Debug)]
#[command(
    name = "mesh2splat",
    version,
    about = "Fast mesh to 3D Gaussian Splatting conversion (Rust/wgpu port of EA SEED's Mesh2Splat)"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Open the interactive viewer (default).
    Gui {
        /// Optional .glb/.gltf/.ply to open at startup.
        file: Option<PathBuf>,
    },
    /// Convert a mesh (or a folder of meshes) to a 3DGS .ply.
    Convert(ConvertArgs),
    /// Render a mesh or .ply to a PNG without opening a window.
    Render(RenderArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum FormatArg {
    Standard,
    /// Standard without the (all-zero) higher-order SH coefficients.
    Sh0,
    Pbr,
    Compressed,
    /// PlayCanvas / SuperSplat compressed PLY (~16 bytes per splat).
    Playcanvas,
}

impl From<FormatArg> for PlyFormat {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Standard => PlyFormat::Standard,
            FormatArg::Sh0 => PlyFormat::StandardSh0,
            FormatArg::Pbr => PlyFormat::Pbr,
            FormatArg::Compressed => PlyFormat::CompressedPbr,
            FormatArg::Playcanvas => PlyFormat::PlayCanvas,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BBoxArg {
    Scene,
    PerMesh,
}

#[derive(Args, Debug, Clone)]
pub struct SamplingArgs {
    /// Conversion render-target size (overrides --quality).
    #[arg(long)]
    pub resolution: Option<u32>,
    /// Sampling density in [0, 1], mapped to 16..=max-res like the UI slider.
    #[arg(long, default_value_t = 0.5)]
    pub quality: f32,
    /// Upper end of the quality slider (1024, 2048 or 4096 in the UI).
    #[arg(long, default_value_t = 1024)]
    pub max_res: u32,
    /// Gaussian scale (standard deviation multiplier).
    #[arg(long, default_value_t = 0.65)]
    pub std: f32,
    /// Bounding box used for the planar re-projection.
    #[arg(long, value_enum, default_value_t = BBoxArg::Scene)]
    pub bbox: BBoxArg,
    /// Merge alike neighbouring splats into larger ones, with an optional
    /// strength in [0, 1] (0 = only identical splats, default 0.25).
    #[arg(long, value_name = "STRENGTH", num_args = 0..=1, default_missing_value = "0.25")]
    pub merge: Option<f32>,
    /// Sample low-detail triangles on a coarser grid, with an optional
    /// tolerance in [0, 1] (default 0.25).
    #[arg(long, value_name = "TOLERANCE", num_args = 0..=1, default_missing_value = "0.25")]
    pub detail: Option<f32>,
}

impl SamplingArgs {
    pub fn settings(&self) -> ConvertSettings {
        ConvertSettings {
            resolution: self
                .resolution
                .unwrap_or_else(|| resolution_from_quality(self.quality, self.max_res)),
            bbox_mode: match self.bbox {
                BBoxArg::Scene => BBoxMode::Scene,
                BBoxArg::PerMesh => BBoxMode::PerMesh,
            },
            merge: self.merge.map(crate::merge::MergeSettings::from_strength),
            detail: self
                .detail
                .map(crate::gpu::converter::DetailSettings::from_strength),
        }
    }
}

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Input .glb/.gltf file, or a folder when --batch is given.
    pub input: PathBuf,
    /// Output .ply (single file) or output folder (batch). Defaults next to the input.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Convert every .glb/.gltf in the input folder.
    #[arg(long)]
    pub batch: bool,
    /// With --batch, also look in subfolders.
    #[arg(long)]
    pub recursive: bool,
    #[arg(long, value_enum, default_value_t = FormatArg::Standard)]
    pub format: FormatArg,
    #[command(flatten)]
    pub sampling: SamplingArgs,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ModeArg {
    Final,
    Albedo,
    Depth,
    Normal,
    Geometry,
    Overdraw,
    Pbr,
}

impl From<ModeArg> for RenderMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Final => RenderMode::Final,
            ModeArg::Albedo => RenderMode::Albedo,
            ModeArg::Depth => RenderMode::Depth,
            ModeArg::Normal => RenderMode::Normal,
            ModeArg::Geometry => RenderMode::Geometry,
            ModeArg::Overdraw => RenderMode::Overdraw,
            ModeArg::Pbr => RenderMode::Pbr,
        }
    }
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    /// .glb/.gltf (converted first) or 3DGS .ply
    pub input: PathBuf,
    #[arg(short, long, default_value = "render.png")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 1280)]
    pub width: u32,
    #[arg(long, default_value_t = 720)]
    pub height: u32,
    #[arg(long, value_enum, default_value_t = ModeArg::Final)]
    pub mode: ModeArg,
    /// Enable the point light (Final mode).
    #[arg(long)]
    pub light: bool,
    /// Light position (x,y,z). Defaults to the upper-front-right of the model.
    #[arg(long, value_delimiter = ',', num_args = 3)]
    pub light_pos: Option<Vec<f32>>,
    #[arg(long, default_value_t = 10.0)]
    pub light_intensity: f32,
    /// Split-screen mesh (left) vs splats (right). Mesh inputs only.
    #[arg(long)]
    pub split: bool,
    /// Mesh/gaussian depth test (mesh inputs only).
    #[arg(long)]
    pub depth_test: bool,
    /// Camera yaw around the model, degrees.
    #[arg(long, default_value_t = 0.0)]
    pub orbit: f32,
    /// Background color (r,g,b) in [0,1].
    #[arg(long, value_delimiter = ',', num_args = 3)]
    pub background: Option<Vec<f32>>,
    #[command(flatten)]
    pub sampling: SamplingArgs,
}

fn is_mesh(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("glb") | Some("gltf")
    )
}

fn collect_meshes(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("cannot read {}", dir.display()))? {
        let p = entry?.path();
        if p.is_dir() {
            if recursive {
                collect_meshes(&p, recursive, out)?;
            }
        } else if is_mesh(&p) {
            out.push(p);
        }
    }
    Ok(())
}

pub fn run_convert(args: ConvertArgs) -> Result<()> {
    let ctx = GpuContext::new_headless()?;
    log::info!(
        "GPU: {} ({:?})",
        ctx.adapter_info.name,
        ctx.adapter_info.backend
    );
    let mut converter = Converter::new(&ctx);
    let settings = args.sampling.settings();
    let format: PlyFormat = args.format.into();

    let jobs: Vec<(PathBuf, PathBuf)> = if args.batch {
        if !args.input.is_dir() {
            bail!("--batch expects a folder");
        }
        let mut inputs = Vec::new();
        collect_meshes(&args.input, args.recursive, &mut inputs)?;
        inputs.sort();
        let out_dir = args.output.clone().unwrap_or_else(|| args.input.clone());
        std::fs::create_dir_all(&out_dir)?;
        inputs
            .into_iter()
            .map(|p| {
                let rel = p
                    .strip_prefix(&args.input)
                    .unwrap_or(&p)
                    .with_extension("ply");
                (p.clone(), out_dir.join(rel))
            })
            .collect()
    } else {
        let out = args
            .output
            .clone()
            .unwrap_or_else(|| args.input.with_extension("ply"));
        vec![(args.input.clone(), out)]
    };
    if jobs.is_empty() {
        bail!("no .glb/.gltf files found");
    }

    let mut failures = 0;
    for (input, output) in &jobs {
        match convert_one(
            &ctx,
            &mut converter,
            input,
            output,
            settings,
            format,
            args.sampling.std,
        ) {
            Ok(()) => {}
            Err(e) => {
                failures += 1;
                eprintln!("FAILED {}: {e:#}", input.display());
            }
        }
    }
    if failures > 0 {
        bail!("{failures} of {} conversions failed", jobs.len());
    }
    Ok(())
}

fn convert_one(
    ctx: &GpuContext,
    converter: &mut Converter,
    input: &Path,
    output: &Path,
    settings: ConvertSettings,
    format: PlyFormat,
    std: f32,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let scene = scene::load_gltf(input)?;
    let t_load = t0.elapsed();
    let gpu_scene = GpuScene::upload(ctx, &scene);
    let mut gaussians = GaussianBuffer::new_empty(ctx);
    let stats = converter.convert(ctx, &gpu_scene, settings, &mut gaussians);
    let data = gaussians.download(ctx);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    ply::write_ply(output, &data, format, gaussians.scale_multiplier(std))?;
    println!(
        "{} -> {}: {} meshes, {} triangles, resolution {}, {} gaussians (load {:.0} ms, convert {:.1} ms, total {:.0} ms)",
        input.display(),
        output.display(),
        scene.meshes.len(),
        scene.triangle_count(),
        settings.resolution,
        stats.gaussians,
        t_load.as_secs_f64() * 1e3,
        stats.duration.as_secs_f64() * 1e3,
        t0.elapsed().as_secs_f64() * 1e3,
    );
    if stats.fragments > stats.capacity {
        eprintln!(
            "warning: {} splats dropped (capacity {})",
            stats.fragments - stats.capacity,
            stats.capacity
        );
    }
    Ok(())
}

pub fn run_render(args: RenderArgs) -> Result<()> {
    let ctx = GpuContext::new_headless()?;
    log::info!(
        "GPU: {} ({:?})",
        ctx.adapter_info.name,
        ctx.adapter_info.backend
    );
    let mut gaussians = GaussianBuffer::new_empty(&ctx);
    let mut gpu_scene = None;
    let bbox;
    if is_mesh(&args.input) {
        let scene = scene::load_gltf(&args.input)?;
        bbox = scene.bbox;
        let gs = GpuScene::upload(&ctx, &scene);
        let mut converter = Converter::new(&ctx);
        let stats = converter.convert(&ctx, &gs, args.sampling.settings(), &mut gaussians);
        println!(
            "converted {} gaussians in {:.1} ms",
            stats.gaussians,
            stats.duration.as_secs_f64() * 1e3
        );
        gpu_scene = Some(gs);
    } else {
        let loaded = ply::load_gaussian_ply(&args.input)?;
        let mut b = crate::types::BBox::EMPTY;
        for g in &loaded.gaussians {
            b.grow(Vec3::from_slice(&g.position[..3]));
        }
        bbox = b;
        gaussians.upload_ply(&ctx, &loaded.gaussians, loaded.has_pbr);
        println!(
            "loaded {} gaussians (pbr: {})",
            loaded.gaussians.len(),
            loaded.has_pbr
        );
    }

    let mut camera = Camera::default();
    camera.frame_bbox(&bbox);
    if args.orbit != 0.0 {
        let c = bbox.center();
        let r = camera.position - c;
        let rot = glam::Quat::from_rotation_y(-args.orbit.to_radians());
        camera.position = c + rot * r;
        camera.yaw += args.orbit;
        camera.process_mouse_movement(0.0, 0.0, true);
    }

    let radius = (bbox.size().length() * 0.5).max(1e-3);
    let light_pos = match &args.light_pos {
        Some(v) => Vec3::new(v[0], v[1], v[2]),
        None => bbox.center() + Vec3::new(1.0, 1.2, 1.5) * radius,
    };
    let settings = RenderSettings {
        render_mode: args.mode.into(),
        gaussian_std: args.sampling.std,
        depth_test: args.depth_test,
        lighting: args.light,
        light_intensity: args.light_intensity,
        light_transform: Mat4::from_translation(light_pos),
        background: args
            .background
            .as_ref()
            .map(|b| [b[0], b[1], b[2], 1.0])
            .unwrap_or([0.0, 0.0, 0.0, 1.0]),
        split_screen: args.split,
        ..Default::default()
    };

    let mut renderer = Renderer::new(&ctx);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    renderer.render(
        &ctx,
        &mut enc,
        &camera,
        &settings,
        &gaussians,
        gpu_scene.as_ref(),
        (args.width, args.height),
    );
    ctx.queue.submit([enc.finish()]);
    let visible = renderer.read_visible_count(&ctx);
    let (w, h, px) = renderer.read_output(&ctx).context("no output")?;
    image::save_buffer(&args.output, &px, w, h, image::ExtendedColorType::Rgba8)?;
    println!(
        "wrote {} ({}x{}, {} visible splats)",
        args.output.display(),
        w,
        h,
        visible
    );
    Ok(())
}
