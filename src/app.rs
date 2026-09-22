//! Interactive viewer (port of the original ImGui UI + mediator), built on eframe/egui.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use eframe::egui_wgpu;
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use glam::{DMat4, DQuat, DVec3, Mat4, Vec3};
use transform_gizmo_egui::math::Transform;
use transform_gizmo_egui::{Gizmo, GizmoConfig, GizmoExt, GizmoMode, GizmoOrientation};

use crate::camera::{Camera, CameraKeys};
use crate::gpu::converter::{resolution_from_quality, ConversionStats, DetailSettings};
use crate::merge::MergeSettings;
use crate::gpu::renderer::OUTPUT_FORMAT;
use crate::gpu::{
    BBoxMode, ConvertSettings, Converter, GaussianBuffer, GpuContext, GpuScene, RenderSettings,
    Renderer,
};
use crate::ply::{self, LoadedPly};
use crate::scene::{self, Scene};
use crate::types::{BBox, PlyFormat, RenderMode, SourceFormat};

pub fn run(file: Option<PathBuf>) -> Result<()> {
    let mut create = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    // Compute + storage-in-fragment needs a "real" backend (Vulkan / Metal / DX12).
    create.instance_descriptor.backends = wgpu::Backends::PRIMARY;
    create.power_preference = wgpu::PowerPreference::HighPerformance;
    create.device_descriptor = Arc::new(crate::gpu::device_descriptor);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Mesh2Splat")
            .with_inner_size([1400.0, 900.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(create),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "Mesh2Splat",
        options,
        Box::new(move |cc| {
            let app =
                App::new(cc, file).map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    e.to_string().into()
                })?;
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

// ---------------------------------------------------------------------------

enum Loaded {
    Mesh {
        path: PathBuf,
        scene: Scene,
        batch: bool,
    },
    Ply {
        path: PathBuf,
        ply: LoadedPly,
    },
    Failed {
        path: PathBuf,
        error: String,
        batch: bool,
    },
}

fn spawn_load(path: PathBuf, batch: bool, tx: flume::Sender<Loaded>, egui_ctx: egui::Context) {
    std::thread::spawn(move || {
        let is_ply = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("ply"));
        let msg = if is_ply {
            match ply::load_gaussian_ply(&path) {
                Ok(ply) => Loaded::Ply { path, ply },
                Err(e) => Loaded::Failed {
                    path,
                    error: format!("{e:#}"),
                    batch,
                },
            }
        } else {
            match scene::load_gltf(&path) {
                Ok(scene) => Loaded::Mesh { path, scene, batch },
                Err(e) => Loaded::Failed {
                    path,
                    error: format!("{e:#}"),
                    batch,
                },
            }
        };
        let _ = tx.send(msg);
        egui_ctx.request_repaint();
    });
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GizmoTarget {
    Model,
    Light,
}

/// Viewport navigation scheme.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraControls {
    /// Original Mesh2Splat fly camera (RMB look + WASD).
    Fly,
    /// Maya: Alt+LMB tumble, Alt+MMB (or Alt+Cmd+LMB) pan, Alt+RMB / wheel / pinch dolly, F frame.
    Maya,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GizmoOp {
    Translate,
    Rotate,
    Scale,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BatchStatus {
    Queued,
    Processing,
    Done,
    Failed(String),
}

#[derive(Clone, Debug)]
struct BatchItem {
    path: PathBuf,
    output: PathBuf,
    status: BatchStatus,
    selected: bool,
}

struct Status {
    text: String,
    error: bool,
    at: Instant,
}

pub struct App {
    ctx: GpuContext,
    renderer: Renderer,
    converter: Converter,
    gaussians: GaussianBuffer,
    scene: Option<GpuScene>,
    scene_bbox: BBox,
    loaded_path: Option<PathBuf>,
    output_texture: Option<(egui::TextureId, (u32, u32))>,

    camera: Camera,
    camera_controls: CameraControls,
    settings: RenderSettings,
    quality: f32,
    max_res: u32,
    bbox_mode: BBoxMode,
    merge_enabled: bool,
    merge_strength: f32,
    detail_enabled: bool,
    detail_strength: f32,
    needs_conversion: bool,
    last_conversion: Option<ConversionStats>,

    // export
    output_folder: String,
    output_name: String,
    format: PlyFormat,
    path_input: String,

    // gizmo
    gizmo: Gizmo,
    gizmo_target: GizmoTarget,
    gizmo_op: GizmoOp,
    gizmo_orientation: GizmoOrientation,

    // loading
    tx: flume::Sender<Loaded>,
    rx: flume::Receiver<Loaded>,
    loading: Option<PathBuf>,

    // batch
    batch_folder: String,
    batch_recursive: bool,
    batch: Vec<BatchItem>,
    batch_running: bool,
    batch_current: Option<usize>,

    // stats
    frame_times: VecDeque<f32>,
    plot_max_ms: f32,
    plot_target_ms: f32,
    last_frame: Instant,
    status: Option<Status>,
    prev_mode_before_lighting: RenderMode,

    // redraw on demand
    /// Inputs of the last rendered frame; the splats are only re-rendered when these change.
    last_render: Option<RenderKey>,
    /// UI frames left to repaint after a render, so the async GPU stats land.
    settle_frames: u32,
    /// Render every frame even when nothing changed (for profiling).
    continuous_redraw: bool,
}

/// Everything a rendered frame depends on.
#[derive(PartialEq)]
struct RenderKey {
    view: Mat4,
    fov: f32,
    size: (u32, u32),
    settings: RenderSettings,
    generation: u64,
    count: u32,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, file: Option<PathBuf>) -> Result<Self> {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("eframe was not started with the wgpu renderer"))?;
        let ctx = GpuContext::from_parts(&rs.adapter, rs.device.clone(), rs.queue.clone());
        log::info!(
            "GPU: {} ({:?})",
            ctx.adapter_info.name,
            ctx.adapter_info.backend
        );
        let (tx, rx) = flume::unbounded();
        let mut app = Self {
            renderer: Renderer::new(&ctx),
            converter: Converter::new(&ctx),
            gaussians: GaussianBuffer::new_empty(&ctx),
            ctx,
            scene: None,
            scene_bbox: BBox::EMPTY,
            loaded_path: None,
            output_texture: None,
            camera: Camera::default(),
            camera_controls: CameraControls::Fly,
            settings: RenderSettings::default(),
            quality: 0.5,
            max_res: 1024,
            bbox_mode: BBoxMode::Scene,
            merge_enabled: false,
            merge_strength: MergeSettings::DEFAULT_STRENGTH,
            detail_enabled: false,
            detail_strength: DetailSettings::DEFAULT_STRENGTH,
            needs_conversion: false,
            last_conversion: None,
            output_folder: String::new(),
            output_name: "output.ply".into(),
            format: PlyFormat::Standard,
            path_input: String::new(),
            gizmo: Gizmo::default(),
            gizmo_target: GizmoTarget::Model,
            gizmo_op: GizmoOp::Translate,
            gizmo_orientation: GizmoOrientation::Local,
            tx,
            rx,
            loading: None,
            batch_folder: String::new(),
            batch_recursive: false,
            batch: Vec::new(),
            batch_running: false,
            batch_current: None,
            frame_times: VecDeque::with_capacity(512),
            plot_max_ms: 33.3,
            plot_target_ms: 16.6,
            last_frame: Instant::now(),
            status: None,
            prev_mode_before_lighting: RenderMode::Final,
            last_render: None,
            settle_frames: 0,
            continuous_redraw: false,
        };
        if let Some(f) = file {
            app.open(f, &cc.egui_ctx);
        }
        Ok(app)
    }

    fn set_status(&mut self, text: impl Into<String>, error: bool) {
        let text = text.into();
        if error {
            log::error!("{text}");
        } else {
            log::info!("{text}");
        }
        self.status = Some(Status {
            text,
            error,
            at: Instant::now(),
        });
    }

    fn open(&mut self, path: PathBuf, egui_ctx: &egui::Context) {
        self.path_input = path.display().to_string();
        self.loading = Some(path.clone());
        spawn_load(path, false, self.tx.clone(), egui_ctx.clone());
    }

    fn resolution(&self) -> u32 {
        resolution_from_quality(self.quality, self.max_res)
    }

    fn convert_settings(&self) -> ConvertSettings {
        ConvertSettings {
            resolution: self.resolution(),
            bbox_mode: self.bbox_mode,
            merge: self
                .merge_enabled
                .then(|| MergeSettings::from_strength(self.merge_strength)),
            detail: self
                .detail_enabled
                .then(|| DetailSettings::from_strength(self.detail_strength)),
        }
    }

    fn run_conversion(&mut self) {
        if let Some(scene) = &self.scene {
            let stats = self.converter.convert(
                &self.ctx,
                scene,
                self.convert_settings(),
                &mut self.gaussians,
            );
            if stats.fragments > stats.capacity {
                self.set_status(
                    format!(
                        "{} splats dropped: GPU buffer limit reached",
                        stats.fragments - stats.capacity
                    ),
                    true,
                );
            }
            self.last_conversion = Some(stats);
        }
        self.needs_conversion = false;
    }

    fn handle_loaded(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Loaded::Mesh { path, scene, batch } => {
                    let gpu = GpuScene::upload(&self.ctx, &scene);
                    self.scene_bbox = scene.bbox;
                    self.scene = Some(gpu);
                    self.settings.model_transform = Mat4::IDENTITY;
                    self.run_conversion();
                    if batch {
                        self.finish_batch_item(None);
                    } else {
                        self.loading = None;
                        self.camera.frame_bbox(&self.scene_bbox);
                        self.place_light_default();
                        if let Some(stem) = path.file_stem() {
                            self.output_name = format!("{}.ply", stem.to_string_lossy());
                        }
                        self.set_status(
                            format!(
                                "Converted {} ({} meshes, {} triangles) into {} gaussians",
                                path.file_name().unwrap_or_default().to_string_lossy(),
                                scene.meshes.len(),
                                scene.triangle_count(),
                                self.gaussians.count
                            ),
                            false,
                        );
                    }
                    self.loaded_path = Some(path);
                }
                Loaded::Ply { path, ply } => {
                    self.loading = None;
                    self.scene = None;
                    self.settings.model_transform = Mat4::IDENTITY;
                    self.settings.depth_test = false;
                    let mut bbox = BBox::EMPTY;
                    for g in &ply.gaussians {
                        bbox.grow(Vec3::from_slice(&g.position[..3]));
                    }
                    self.scene_bbox = bbox;
                    self.gaussians
                        .upload_ply(&self.ctx, &ply.gaussians, ply.has_pbr);
                    if let Some(stem) = path.file_stem() {
                        // Never default to overwriting the file that was just opened.
                        self.output_name = format!("{}_export.ply", stem.to_string_lossy());
                    }
                    self.camera.frame_bbox(&bbox);
                    self.place_light_default();
                    self.set_status(
                        format!(
                            "Loaded {} gaussians from {} (PBR: {})",
                            ply.gaussians.len(),
                            path.display(),
                            ply.has_pbr
                        ),
                        false,
                    );
                    self.loaded_path = Some(path);
                }
                Loaded::Failed { path, error, batch } => {
                    if batch {
                        self.finish_batch_item(Some(error));
                    } else {
                        self.loading = None;
                        self.set_status(
                            format!("Failed to load {}: {error}", path.display()),
                            true,
                        );
                    }
                }
            }
        }
    }

    fn place_light_default(&mut self) {
        if self.scene_bbox.is_valid() {
            let r = (self.scene_bbox.size().length() * 0.5).max(1e-3);
            self.settings.light_transform =
                Mat4::from_translation(self.scene_bbox.center() + Vec3::new(1.0, 1.2, 1.5) * r);
        }
    }

    fn output_path(&self) -> PathBuf {
        let mut name = self.output_name.trim().to_string();
        if name.is_empty() {
            name = "output.ply".into();
        }
        if !name.to_ascii_lowercase().ends_with(".ply") {
            name.push_str(".ply");
        }
        let folder = if self.output_folder.trim().is_empty() {
            self.loaded_path
                .as_ref()
                .and_then(|p| p.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("."))
        } else {
            PathBuf::from(self.output_folder.trim())
        };
        folder.join(name)
    }

    fn save(&mut self, path: PathBuf) {
        if self.gaussians.count == 0 {
            self.set_status("Nothing to save", true);
            return;
        }
        let data = self.gaussians.download(&self.ctx);
        let mult = self.gaussians.scale_multiplier(self.settings.gaussian_std);
        let format = self.format;
        // The file is written on a worker thread, as in the original.
        let (tx, rx) = flume::bounded(1);
        std::thread::spawn(move || {
            let _ = tx.send(ply::write_ply(&path, &data, format, mult).map(|_| path));
        });
        match rx.recv() {
            Ok(Ok(p)) => self.set_status(format!("Saved {}", p.display()), false),
            Ok(Err(e)) => self.set_status(format!("Save failed: {e:#}"), true),
            Err(_) => self.set_status("Save thread crashed", true),
        }
    }

    // --- batch ---------------------------------------------------------------

    fn scan_batch_folder(&mut self) {
        let dir = PathBuf::from(self.batch_folder.trim());
        let mut found = Vec::new();
        fn walk(d: &Path, rec: bool, out: &mut Vec<PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        if rec {
                            walk(&p, rec, out);
                        }
                    } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                        e.eq_ignore_ascii_case("glb") || e.eq_ignore_ascii_case("gltf")
                    }) {
                        out.push(p);
                    }
                }
            }
        }
        walk(&dir, self.batch_recursive, &mut found);
        found.sort();
        let out_dir = if self.output_folder.trim().is_empty() {
            dir.clone()
        } else {
            PathBuf::from(self.output_folder.trim())
        };
        for p in found {
            if self.batch.iter().any(|b| b.path == p) {
                continue;
            }
            let rel = p.strip_prefix(&dir).unwrap_or(&p).with_extension("ply");
            self.batch.push(BatchItem {
                output: out_dir.join(rel),
                path: p,
                status: BatchStatus::Queued,
                selected: false,
            });
        }
    }

    fn pump_batch(&mut self, egui_ctx: &egui::Context) {
        if !self.batch_running || self.batch_current.is_some() {
            return;
        }
        match self
            .batch
            .iter()
            .position(|b| b.status == BatchStatus::Queued)
        {
            Some(i) => {
                self.batch[i].status = BatchStatus::Processing;
                self.batch_current = Some(i);
                spawn_load(
                    self.batch[i].path.clone(),
                    true,
                    self.tx.clone(),
                    egui_ctx.clone(),
                );
            }
            None => {
                self.batch_running = false;
                let done = self
                    .batch
                    .iter()
                    .filter(|b| b.status == BatchStatus::Done)
                    .count();
                self.set_status(
                    format!("Batch finished: {done}/{} converted", self.batch.len()),
                    false,
                );
            }
        }
    }

    fn finish_batch_item(&mut self, error: Option<String>) {
        let Some(i) = self.batch_current.take() else {
            return;
        };
        let status = match error {
            Some(e) => BatchStatus::Failed(e),
            None => {
                let data = self.gaussians.download(&self.ctx);
                let out = self.batch[i].output.clone();
                if let Some(p) = out.parent() {
                    let _ = std::fs::create_dir_all(p);
                }
                match ply::write_ply(
                    &out,
                    &data,
                    self.format,
                    self.gaussians.scale_multiplier(self.settings.gaussian_std),
                ) {
                    Ok(()) => BatchStatus::Done,
                    Err(e) => BatchStatus::Failed(format!("{e:#}")),
                }
            }
        };
        self.batch[i].status = status;
    }

    // --- UI -----------------------------------------------------------------

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        let egui_ctx = ui.ctx().clone();
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Mesh2Splat");
            ui.add_space(4.0);

            egui::CollapsingHeader::new("Input").default_open(true).show(ui, |ui| {
                if ui.button("Select file to load (.glb / .gltf / .ply)").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Mesh or 3DGS", &["glb", "gltf", "ply"]).pick_file() {
                        self.open(p, &egui_ctx);
                    }
                }
                ui.horizontal(|ui| {
                    let edit = ui.add(egui::TextEdit::singleline(&mut self.path_input).hint_text("or type a path").desired_width(200.0));
                    let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if (ui.button("Load").clicked() || enter) && !self.path_input.trim().is_empty() {
                        self.open(PathBuf::from(self.path_input.trim()), &egui_ctx);
                    }
                });
                if let Some(p) = &self.loading {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("Loading {}", p.file_name().unwrap_or_default().to_string_lossy()));
                    });
                } else if let Some(p) = &self.loaded_path {
                    ui.label(format!("Loaded: {}", p.file_name().unwrap_or_default().to_string_lossy()));
                    if self.scene.is_some() && ui.button("Re-run conversion").clicked() {
                        self.needs_conversion = true;
                    }
                }
                ui.small("Tip: drag & drop a file onto the window.");
            });

            egui::CollapsingHeader::new("Output").default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Select output folder").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_folder() {
                            self.output_folder = p.display().to_string();
                        }
                    }
                });
                ui.add(egui::TextEdit::singleline(&mut self.output_folder).hint_text("folder (default: next to input)"));
                ui.add(egui::TextEdit::singleline(&mut self.output_name).hint_text("file name"));
                egui::ComboBox::from_id_salt("format").selected_text(self.format.label()).show_ui(ui, |ui| {
                    for f in PlyFormat::ALL {
                        ui.selectable_value(&mut self.format, f, f.label());
                    }
                });
                let save = egui::Button::new("Save splat").fill(Color32::from_rgb(51, 153, 51));
                if ui.add_enabled(self.gaussians.count > 0, save).clicked() {
                    let p = self.output_path();
                    self.save(p);
                }
            });

            egui::CollapsingHeader::new("Properties").default_open(true).show(ui, |ui| {
                let before = self.settings.render_mode;
                egui::ComboBox::from_label("Property visualization").selected_text(self.settings.render_mode.label()).show_ui(ui, |ui| {
                    for m in RenderMode::UI_ORDER {
                        ui.selectable_value(&mut self.settings.render_mode, m, m.label());
                    }
                });
                if before != self.settings.render_mode && self.settings.render_mode != RenderMode::Final {
                    self.prev_mode_before_lighting = self.settings.render_mode;
                }
                ui.add_enabled(
                    self.scene.is_some(),
                    egui::Checkbox::new(&mut self.settings.depth_test, "Mesh-gaussian depth test (faster)"),
                );
                ui.add(egui::Slider::new(&mut self.settings.gaussian_std, 0.1..=2.0).text("Gaussian Scale"));
                ui.separator();
                ui.label("Sampling density");
                let res_label = format!("= {} px", self.resolution());
                if ui.add(egui::Slider::new(&mut self.quality, 0.0..=1.0).text(res_label)).changed() {
                    self.needs_conversion = true;
                }
                egui::ComboBox::from_label("Max quality").selected_text(self.max_res.to_string()).show_ui(ui, |ui| {
                    for r in [1024u32, 2048, 4096] {
                        if ui.selectable_value(&mut self.max_res, r, r.to_string()).changed() {
                            self.needs_conversion = true;
                        }
                    }
                });
                let bbox_label = |m: BBoxMode| match m {
                    BBoxMode::Scene => "Scene bounds (uniform density)",
                    BBoxMode::PerMesh => "Per-mesh bounds",
                };
                egui::ComboBox::from_label("Projection box").selected_text(bbox_label(self.bbox_mode)).show_ui(ui, |ui| {
                    for m in [BBoxMode::Scene, BBoxMode::PerMesh] {
                        if ui.selectable_value(&mut self.bbox_mode, m, bbox_label(m)).changed() {
                            self.needs_conversion = true;
                        }
                    }
                });
                if ui
                    .checkbox(&mut self.detail_enabled, "Detail-aware density")
                    .on_hover_text("Sample triangles whose textures barely vary on a coarser grid (down to 1/8 density), so detail decides where the splats go.")
                    .changed()
                {
                    self.needs_conversion = true;
                }
                if self.detail_enabled {
                    let r = ui.add(egui::Slider::new(&mut self.detail_strength, 0.0..=1.0).text("Detail tolerance"));
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        self.needs_conversion = true;
                    }
                }
                if ui
                    .checkbox(&mut self.merge_enabled, "Merge similar splats")
                    .on_hover_text("Replace blocks of neighbouring splats with the same colour, normal and material by one larger splat (up to 16x16). Fewer splats: faster rendering and smaller files.")
                    .changed()
                {
                    self.needs_conversion = true;
                }
                if self.merge_enabled {
                    // Merging runs on the CPU, so only re-convert when the slider is released.
                    let r = ui.add(egui::Slider::new(&mut self.merge_strength, 0.0..=1.0).text("Merge strength"));
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        self.needs_conversion = true;
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Background");
                    ui.color_edit_button_rgba_unmultiplied(&mut self.settings.background);
                });
                ui.add_enabled(self.scene.is_some(), egui::Checkbox::new(&mut self.settings.split_screen, "Split-screen (mesh | splat)"));
                if self.settings.split_screen && self.scene.is_some() {
                    ui.add(egui::Slider::new(&mut self.settings.split_position, 0.0..=1.0).text("Split position"));
                }
            });

            egui::CollapsingHeader::new("Lighting").default_open(false).show(ui, |ui| {
                let was = self.settings.lighting;
                ui.checkbox(&mut self.settings.lighting, "Enable lighting");
                if self.settings.lighting && !was {
                    self.prev_mode_before_lighting = self.settings.render_mode;
                    self.settings.render_mode = RenderMode::Final;
                } else if !self.settings.lighting && was {
                    self.settings.render_mode = self.prev_mode_before_lighting;
                    self.gizmo_target = GizmoTarget::Model;
                }
                if self.settings.lighting {
                    ui.add(egui::Slider::new(&mut self.settings.light_intensity, 0.0..=1000.0).logarithmic(true).text("Light intensity"));
                    let mut c = self.settings.light_color.to_array();
                    ui.horizontal(|ui| {
                        ui.label("Light color");
                        ui.color_edit_button_rgb(&mut c);
                    });
                    self.settings.light_color = Vec3::from_array(c);
                    if self.gaussians.format == SourceFormat::Ply && !self.gaussians.ply_has_pbr {
                        ui.small("This PLY has no normals/PBR data: normals come from the shortest splat axis.");
                    }
                }
            });

            egui::CollapsingHeader::new("Camera").default_open(false).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Controls:");
                    ui.radio_value(&mut self.camera_controls, CameraControls::Fly, "Fly");
                    ui.radio_value(&mut self.camera_controls, CameraControls::Maya, "Maya");
                });
                if ui.button("Frame model (F)").clicked() {
                    self.focus_model();
                }
            });

            egui::CollapsingHeader::new("Gizmo").default_open(false).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.gizmo_op, GizmoOp::Translate, "Translate");
                    ui.radio_value(&mut self.gizmo_op, GizmoOp::Rotate, "Rotate");
                    ui.radio_value(&mut self.gizmo_op, GizmoOp::Scale, "Scale");
                });
                if self.gizmo_op != GizmoOp::Scale {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.gizmo_orientation, GizmoOrientation::Local, "Local");
                        ui.radio_value(&mut self.gizmo_orientation, GizmoOrientation::Global, "World");
                    });
                }
                if self.settings.lighting {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.gizmo_target, GizmoTarget::Model, "Model");
                        ui.radio_value(&mut self.gizmo_target, GizmoTarget::Light, "Light");
                    });
                }
                ui.horizontal(|ui| {
                    if ui.button("Reset model").clicked() {
                        self.settings.model_transform = Mat4::IDENTITY;
                    }
                    if ui.button("Frame camera").clicked() {
                        self.camera.frame_bbox(&self.model_bbox());
                    }
                });
            });

            egui::CollapsingHeader::new("Batch conversion").default_open(false).show(ui, |ui| self.batch_ui(ui));

            egui::CollapsingHeader::new("Stats").default_open(true).show(ui, |ui| self.stats_ui(ui));

            ui.separator();
            ui.small(match self.camera_controls {
                CameraControls::Fly => "Camera: RMB drag to look, WASD move, Q/E down/up, R/T roll, Shift fast, Ctrl slow, wheel zoom, F frame.",
                CameraControls::Maya => "Camera: Alt+LMB tumble, Alt+MMB or Alt+Cmd+LMB pan, Alt+RMB / wheel / pinch dolly, F frame.",
            });
        });
    }

    fn batch_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Select input folder").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.batch_folder = p.display().to_string();
                }
            }
            ui.checkbox(&mut self.batch_recursive, "Include subfolders");
        });
        ui.add(egui::TextEdit::singleline(&mut self.batch_folder).hint_text("input folder"));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.batch_running && !self.batch_folder.trim().is_empty(),
                    egui::Button::new("Add files"),
                )
                .clicked()
            {
                self.scan_batch_folder();
            }
            if !self.batch_running {
                let start = egui::Button::new("Start batch").fill(Color32::from_rgb(51, 153, 51));
                if ui
                    .add_enabled(
                        self.batch.iter().any(|b| b.status == BatchStatus::Queued),
                        start,
                    )
                    .clicked()
                {
                    self.batch_running = true;
                }
                if ui.button("Clear list").clicked() {
                    self.batch.clear();
                }
            } else if ui
                .add(egui::Button::new("Cancel").fill(Color32::from_rgb(178, 51, 51)))
                .clicked()
            {
                self.batch_running = false;
            }
        });
        ui.small(format!(
            "Uses the current density ({} px), format and output folder.",
            self.resolution()
        ));
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .id_salt("batch list")
            .show(ui, |ui| {
                for item in &mut self.batch {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut item.selected, "");
                        let (txt, col) = match &item.status {
                            BatchStatus::Queued => ("Queued".to_string(), Color32::GRAY),
                            BatchStatus::Processing => ("Processing".to_string(), Color32::YELLOW),
                            BatchStatus::Done => {
                                ("Done".to_string(), Color32::from_rgb(90, 200, 90))
                            }
                            BatchStatus::Failed(e) => {
                                (format!("Failed: {e}"), Color32::from_rgb(230, 90, 90))
                            }
                        };
                        ui.colored_label(col, txt);
                        ui.label(item.path.file_name().unwrap_or_default().to_string_lossy());
                    });
                }
            });
        if !self.batch_running && ui.button("Remove selected").clicked() {
            self.batch.retain(|b| !b.selected);
        }
    }

    fn stats_ui(&mut self, ui: &mut egui::Ui) {
        let stats = self.renderer.stats();
        ui.label(format!(
            "Total gaussian count: {}",
            fmt_thousands(self.gaussians.count as u64)
        ));
        ui.label(format!(
            "Visible gaussian count: {}",
            fmt_thousands(stats.visible_gaussians as u64)
        ));
        if let Some(c) = &self.last_conversion {
            ui.label(format!(
                "Last conversion: {:.2} ms ({} px)",
                c.duration.as_secs_f64() * 1e3,
                self.gaussians.resolution
            ));
            if let Some(m) = &c.merge {
                ui.label(format!(
                    "Merged {} -> {} splats ({:.1}x fewer, {:.0} ms on the {})",
                    fmt_thousands(m.input as u64),
                    fmt_thousands(m.output as u64),
                    m.input as f64 / m.output.max(1) as f64,
                    m.duration.as_secs_f64() * 1e3,
                    if m.gpu { "GPU" } else { "CPU" }
                ));
            }
        }
        let latest = self.frame_times.back().copied().unwrap_or(0.0);
        let label = if stats.gpu_ms.is_some() {
            "GPU frame time"
        } else {
            "Frame time"
        };
        ui.label(format!("{label}: {latest:.3} ms"));
        if let Some(st) = stats.stages {
            ui.small(format!(
                "prepass {:.2} · sort {:.2} · splat raster {:.2} ms",
                st.prepass, st.sort, st.splat
            ));
        }
        ui.small(format!(
            "{} ({:?})",
            self.ctx.adapter_info.name, self.ctx.adapter_info.backend
        ));

        // Frame-time plot
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 80.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_gray(25));
        let y_of =
            |ms: f32| rect.bottom() - (ms / self.plot_max_ms).clamp(0.0, 1.0) * rect.height();
        let ty = y_of(self.plot_target_ms);
        painter.line_segment(
            [Pos2::new(rect.left(), ty), Pos2::new(rect.right(), ty)],
            Stroke::new(1.0, Color32::from_rgb(200, 80, 80)),
        );
        let n = self.frame_times.len();
        if n > 1 {
            let pts: Vec<Pos2> = self
                .frame_times
                .iter()
                .enumerate()
                .map(|(i, &ms)| {
                    Pos2::new(
                        rect.left() + rect.width() * i as f32 / (n - 1) as f32,
                        y_of(ms),
                    )
                })
                .collect();
            painter.add(egui::Shape::line(
                pts,
                Stroke::new(1.0, Color32::from_rgb(100, 200, 255)),
            ));
        }
        ui.checkbox(&mut self.continuous_redraw, "Continuous redraw")
            .on_hover_text("Re-render every frame even when nothing changed (for profiling). Off: the splats are only redrawn when the view or settings change.");
        ui.add(egui::Slider::new(&mut self.plot_max_ms, 16.6..=100.0).text("Max scale (ms)"));
        ui.add(egui::Slider::new(&mut self.plot_target_ms, 8.3..=50.0).text("Target line (ms)"));
    }

    /// World-space bounds of the model with its gizmo transform applied.
    fn model_bbox(&self) -> BBox {
        let t = self.settings.model_transform;
        let b = self.scene_bbox;
        let (mn, mx) = (t.transform_point3(b.min), t.transform_point3(b.max));
        BBox { min: mn.min(mx), max: mn.max(mx) }
    }

    /// Frame the model, keeping the current view direction (Maya `F`).
    fn focus_model(&mut self) {
        self.camera.focus_bbox(&self.model_bbox());
    }

    /// Wheel notches, like GLFW's scroll callback in the original.
    fn wheel_notches(ui: &egui::Ui) -> f32 {
        ui.input(|i| {
            i.raw
                .events
                .iter()
                .map(|e| match e {
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Line,
                        delta,
                        ..
                    } => delta.y,
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta,
                        ..
                    } => delta.y / 40.0,
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Page,
                        delta,
                        ..
                    } => delta.y * 10.0,
                    _ => 0.0,
                })
                .sum()
        })
    }

    fn camera_input(&mut self, ui: &egui::Ui, response: &egui::Response, dt: f32) {
        let wants_kb = ui.ctx().egui_wants_keyboard_input();
        if !wants_kb && response.hovered() && ui.input(|i| i.key_pressed(egui::Key::F)) {
            self.focus_model();
        }
        match self.camera_controls {
            CameraControls::Fly => self.fly_input(ui, response, dt, wants_kb),
            CameraControls::Maya => self.maya_input(ui, response),
        }
    }

    fn fly_input(&mut self, ui: &egui::Ui, response: &egui::Response, dt: f32, wants_kb: bool) {
        if !wants_kb {
            let keys = ui.input(|i| CameraKeys {
                forward: i.key_down(egui::Key::W),
                backward: i.key_down(egui::Key::S),
                left: i.key_down(egui::Key::A),
                right: i.key_down(egui::Key::D),
                up: i.key_down(egui::Key::E),
                down: i.key_down(egui::Key::Q),
                roll_left: i.key_down(egui::Key::T),
                roll_right: i.key_down(egui::Key::R),
                boost: i.modifiers.shift,
                slow: i.modifiers.ctrl || i.modifiers.command,
            });
            self.camera.process_keyboard(dt, keys);
        }
        if response.dragged_by(egui::PointerButton::Secondary) {
            let d = response.drag_delta() * ui.ctx().pixels_per_point();
            self.camera.process_mouse_movement(d.x, -d.y, true);
        }
        if response.hovered() {
            let scroll = Self::wheel_notches(ui);
            if scroll != 0.0 {
                self.camera.process_mouse_scroll(scroll / 40.0);
            }
        }
    }

    fn maya_input(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let (alt, command) = ui.input(|i| (i.modifiers.alt, i.modifiers.command));
        if alt {
            let d = response.drag_delta();
            // Alt+Cmd+LMB (Alt+Ctrl off macOS) pans without a middle button, as Maya does on Mac.
            if command && response.dragged_by(egui::PointerButton::Primary) {
                self.camera.pan(d.x, d.y, response.rect.height());
            } else if response.dragged_by(egui::PointerButton::Primary) {
                self.camera.tumble(d.x, d.y);
            } else if response.dragged_by(egui::PointerButton::Middle) {
                self.camera.pan(d.x, d.y, response.rect.height());
            } else if response.dragged_by(egui::PointerButton::Secondary) {
                // Drag right or down to dolly in, left or up to dolly out.
                self.camera.dolly((d.x + d.y) * 0.005);
            }
        }
        if response.hovered() {
            let scroll = Self::wheel_notches(ui);
            if scroll != 0.0 {
                self.camera.dolly(scroll * 0.1);
            }
            // Trackpad pinch.
            let zoom = ui.input(|i| i.zoom_delta());
            if zoom != 1.0 {
                self.camera.dolly(zoom.ln());
            }
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let rect = ui.max_rect();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let now = Instant::now();
        // UI frames only run on demand, so cap the step after an idle period.
        let dt = (now - self.last_frame).as_secs_f32().min(0.05);
        self.last_frame = now;
        self.camera_input(ui, &response, dt);

        let ppp = ui.ctx().pixels_per_point();
        let size = (
            (rect.width() * ppp).round().max(1.0) as u32,
            (rect.height() * ppp).round().max(1.0) as u32,
        );

        let key = RenderKey {
            view: self.camera.view_matrix(),
            fov: self.camera.fov,
            size,
            settings: self.settings.clone(),
            generation: self.gaussians.generation,
            count: self.gaussians.count,
        };
        if self.continuous_redraw || self.last_render.as_ref() != Some(&key) {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame"),
                });
            self.renderer.render(
                &self.ctx,
                &mut enc,
                &self.camera,
                &self.settings,
                &self.gaussians,
                self.scene.as_ref(),
                size,
            );
            self.ctx.queue.submit([enc.finish()]);
            self.last_render = Some(key);
            self.settle_frames = 3;

            let ms = self
                .renderer
                .stats()
                .gpu_ms
                .map(|v| v as f32)
                .unwrap_or(dt * 1000.0);
            if self.frame_times.len() >= 300 {
                self.frame_times.pop_front();
            }
            self.frame_times.push_back(ms);
        }
        // Non-blocking: collects the GPU stats of earlier frames.
        self.renderer.after_submit(&self.ctx);

        // Hand the output texture to egui.
        if let (Some(rs), Some((_, view))) = (frame.wgpu_render_state(), self.renderer.output()) {
            debug_assert_eq!(OUTPUT_FORMAT, wgpu::TextureFormat::Rgba8Unorm);
            let mut r = rs.renderer.write();
            match &mut self.output_texture {
                Some((id, s)) if *s == size => {
                    let _ = id;
                }
                Some((id, s)) => {
                    r.update_egui_texture_from_wgpu_texture(
                        &rs.device,
                        view,
                        wgpu::FilterMode::Nearest,
                        *id,
                    );
                    *s = size;
                }
                None => {
                    let id = r.register_native_texture(&rs.device, view, wgpu::FilterMode::Nearest);
                    self.output_texture = Some((id, size));
                }
            }
        }
        if let Some((id, _)) = self.output_texture {
            ui.painter().image(
                id,
                rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        // Light marker
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix(rect.width() / rect.height());
        if self.settings.lighting {
            let p = proj * view * self.settings.light_transform.w_axis.truncate().extend(1.0);
            if p.w > 0.0 {
                let ndc = p.truncate() / p.w;
                let pos = Pos2::new(
                    rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(),
                    rect.top() + (0.5 - 0.5 * ndc.y) * rect.height(),
                );
                let c = self.settings.light_color;
                let col = Color32::from_rgb(
                    (c.x * 255.0) as u8,
                    (c.y * 255.0) as u8,
                    (c.z * 255.0) as u8,
                );
                ui.painter()
                    .circle(pos, 6.0, col, Stroke::new(1.5, Color32::BLACK));
            }
        }

        // Transform gizmo
        if self.gaussians.count > 0 {
            let modes = match self.gizmo_op {
                GizmoOp::Translate => GizmoMode::all_translate(),
                GizmoOp::Rotate => GizmoMode::all_rotate(),
                GizmoOp::Scale => GizmoMode::all_scale(),
            };
            self.gizmo.update_config(GizmoConfig {
                view_matrix: view.as_dmat4().into(),
                projection_matrix: proj.as_dmat4().into(),
                viewport: rect,
                modes,
                orientation: self.gizmo_orientation,
                ..Default::default()
            });
            let target = match self.gizmo_target {
                GizmoTarget::Light if self.settings.lighting => &mut self.settings.light_transform,
                _ => &mut self.settings.model_transform,
            };
            let (s, r, t) = target.as_dmat4().to_scale_rotation_translation();
            let transform = Transform::from_scale_rotation_translation(s, r, t);
            // Alt belongs to the camera in Maya mode, so Alt+LMB over the gizmo tumbles.
            let camera_owns_mouse =
                self.camera_controls == CameraControls::Maya && ui.input(|i| i.modifiers.alt);
            let result = self.gizmo.interact(ui, &[transform]).filter(|_| !camera_owns_mouse);
            if let Some((_, new)) = result {
                if let Some(n) = new.first() {
                    let m = DMat4::from_scale_rotation_translation(
                        DVec3::from(n.scale),
                        DQuat::from(n.rotation),
                        DVec3::from(n.translation),
                    );
                    *target = m.as_mat4();
                }
            }
        }

        if let Some(s) = &self.status {
            if s.at.elapsed().as_secs_f32() < 6.0 {
                let col = if s.error {
                    Color32::from_rgb(255, 110, 110)
                } else {
                    Color32::from_gray(220)
                };
                ui.painter().text(
                    rect.left_bottom() + Vec2::new(10.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    &s.text,
                    egui::FontId::proportional(14.0),
                    col,
                );
            }
        }
        if self.gaussians.count == 0 && self.loading.is_none() {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Open or drop a .glb / .gltf mesh or a 3DGS .ply",
                egui::FontId::proportional(20.0),
                Color32::from_gray(160),
            );
        }
    }
}

impl App {
    /// egui repaints on input by itself; this keeps frames coming only while
    /// something is animating, so an idle viewport costs no GPU time.
    fn schedule_repaint(&mut self, ctx: &egui::Context) {
        let fly_keys_held = self.camera_controls == CameraControls::Fly
            && !ctx.egui_wants_keyboard_input()
            && ctx.input(|i| {
                use egui::Key::*;
                [W, A, S, D, Q, E, R, T].iter().any(|k| i.key_down(*k))
            });
        let busy = self.loading.is_some() || self.batch_running || self.needs_conversion;
        if self.continuous_redraw || busy || fly_keys_held || self.settle_frames > 0 {
            self.settle_frames = self.settle_frames.saturating_sub(1);
            ctx.request_repaint();
        } else if let Some(s) = &self.status {
            // Repaint once more when the status message should disappear.
            let left = 6.0 - s.at.elapsed().as_secs_f32();
            if left > 0.0 {
                ctx.request_repaint_after(std::time::Duration::from_secs_f32(left));
            }
        }
    }
}

fn fmt_thousands(v: u64) -> String {
    let s = v.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let egui_ctx = ui.ctx().clone();
        self.handle_loaded();
        self.pump_batch(&egui_ctx);
        if self.needs_conversion && !self.batch_running {
            self.run_conversion();
        }

        // Drag and drop
        let dropped: Vec<PathBuf> = egui_ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if let Some(p) = dropped.into_iter().next() {
            if p.is_dir() {
                self.batch_folder = p.display().to_string();
                self.scan_batch_folder();
            } else {
                self.open(p, &egui_ctx);
            }
        }

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(330.0)
            .show_inside(ui, |ui| self.side_panel(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| self.viewport(ui, frame));
        self.schedule_repaint(&egui_ctx);
    }
}
