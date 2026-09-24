# mesh2splat (Rust / wgpu)

A Rust port of [EA SEED's Mesh2Splat](https://github.com/electronicarts/mesh2splat):
fast conversion of textured triangle meshes (`.glb` / `.gltf`) into 3D Gaussian
Splatting `.ply` (or compressed `.spz`) files, plus the real-time splat renderer from the original
(PBR relighting, point-light shadows, mesh/splat split-screen, batch conversion).

The original is C++17 + OpenGL 4.6. This port uses **wgpu** (Vulkan / Metal / DX12),
**WGSL** shaders and **egui** for the UI, so it builds on Linux, Windows and macOS
with `cargo` alone.

## How the conversion works

It works the same way as the original, with no optimization or training.

1. Each triangle is projected onto the plane of its dominant normal axis, inside
   the mesh bounding box, giving normalized 2D coordinates.
2. All triangles are rasterized into a `resolution x resolution` target at those
   coordinates. **Every fragment becomes one gaussian**:
   * The position, UV, normal and tangent are interpolated.
   * Albedo, normal map and metallic-roughness are sampled from the glTF textures.
   * The rotation is the triangle's tangent frame (longest edge, bitangent, normal).
   * The scale is the column lengths of the Jacobian of the 2D-to-3D mapping. The
     thin axis gets 1e-7, so the result is a flat, surfel-like splat.
3. Scales are stored unscaled. At render or export time they are multiplied by
   `gaussian_std / resolution`, which is the "Gaussian Scale" slider.

The original computes the per-triangle part in a **geometry shader**, and WebGPU has
none. In this port the vertex shader instead pulls all three vertices of its
triangle from a storage buffer and computes the same values
(`src/gpu/shaders/convert.wgsl`). The fragment shader appends gaussians to a
storage buffer with an atomic counter, exactly like the original.

### Detail-aware sampling density

Optional, and not in the original. Before conversion, a compute pass gives each
triangle its own sampling level: it compares the textures at the texel footprint
of level 0 and of a coarser level (the splat spacing that level would give) and
keeps the coarsest level whose difference stays within a tolerance. Each level is
then rasterized into its own smaller target, which lands on every 2^level-th cell
of the same grid, and the splats of that level are scaled to match
(`src/gpu/shaders/detail.wgsl`). Levels are also capped so a splat never
outgrows its own triangle.

On DamagedHelmet at 520 px (PSNR against the unmerged render):

| | splats | PSNR (orbit views) | PSNR (close-up) |
|---|---|---|---|
| no merge, no detail | 1.04 M | reference | reference |
| detail 0.25 | 751 k | 43–45 dB | 38 dB |
| detail 0.5 | 606 k | 39–42 dB | 34 dB |
| merge 0.25 | 545 k | 44–45 dB | 36 dB |

Merging gives more reduction per dB than this does, and combining the two is
worse than merging alone at the same splat count, because the level is chosen
per triangle: one detailed corner keeps the whole triangle dense. What detail
levels do give is a cheaper conversion — the coarse splats are never created,
so nothing has to be merged away afterwards — which matters mainly at high
sampling density. Both are off by default; **merging is the better first
choice.**

### Hair, fur and other fibres

Optional, and not in the original. Splats converted from a mesh are flat and
roughly round, but splats that follow a strand are long and thin, and shading
them as surfaces reads as felt. Three settings (all off by default) target that:

* **Anisotropic (hair) shading** — Kajiya-Kay: both the diffuse and the two
  specular lobes work off the angle to each splat's longest axis instead of its
  normal. The tangent runs root to tip and rides through the G-buffer as a
  world-space vector in a target of its own, so the strands overlapping a pixel
  average to the direction they share. (An angle would be cheaper, but angles
  measured in each splat's own frame cannot be averaged, and averaging them is
  what the G-buffer does: highlights came out sparkly and pointing the wrong
  way.) The splat raster is bound by the bytes it blends, and the extra target
  makes it about half as slow again, so only shading that reads the tangent
  (hair, or anisotropy above 0) draws it. WebGPU guarantees only 32 bytes of
  G-buffer a pixel, which the other targets already fill; adapters that allow no
  more (Intel and AMD Macs) go without, and deferred hair highlights lose their
  direction. The two
  lobes take separate roughnesses — sharp along the strand, broad across it,
  which is what reads as hair — and the primary keeps the colour of the light
  while the secondary carries the hair's, with Fresnel from the **index of
  refraction** (1.5 for most dielectrics, ~1.55 for hair).
* **Sheen** — the soft rim a fibrous surface shows at grazing angles (Charlie
  distribution), which is most of what separates fur from plastic. Black by
  default, so it costs nothing until asked for.
* **Anisotropy** for the normal PBR model stretches the highlight along each
  splat's longest axis, for brushed metal and hair cards. 0 by default, which
  is the isotropic GGX the renderer had before.
* **Baked occlusion** — splats have no surface to trace against, so a compute
  pass voxelizes their opacity into a density grid and then, per splat, marches
  it along 16 directions. The average transmittance is the occlusion and the
  weighted mean direction is the bent normal, both stored in spare `pbr`
  channels the splats already carry, so shading gets them for free. This is what
  stops the inside of a groom reading as one solid mass, and it is the largest
  single improvement of the three. It dims ambient light only, in both paths;
  only the bent normal is forward-only, and the deferred path uses the averaged
  normal in its place. 450 k splats bake in ~150 ms.
* **Transmission** is tinted by Beer-Lambert absorption over the optical depth
  the opacity shadows measure, so light that comes through deep hair takes the
  attenuation colour with it.
* **Opacity shadows** — the shadow pass already rasterizes splats from the
  light, so instead of keeping only the nearest depth it accumulates how much
  light each splat absorbs into four distance slices. Self-shadowing becomes
  graded rather than binary, and the optical depth drives **transmission**, the
  light that reaches a splat through the ones in front of it.
* **Forward (per-splat) shading** — shade in the splat pass instead of once per
  pixel afterwards, so overlapping strands blend as lit strands rather than as
  one averaged surface, each using its own normal and tangent. Sharper
  highlights on fine geometry, at the cost of shading per fragment instead of
  per pixel; with hair's overdraw that is a real cost. On an M1 at 1600x1200:

| groom | splats | deferred | forward |
|---|---|---|---|
| straight, front | 450 k | 23 ms | 49 ms |
| straight, close | 450 k | 32 ms | 94 ms |
| curly, front | 3.4 M | 136 ms | 247 ms |
| curly, close | 3.4 M | 153 ms | 333 ms |

  So it costs 2-4x the frame, all of it in the splat raster: worth it for a
  close look at fine geometry, not for a full groom in motion. It writes only
  the colour target, since nothing reads the rest of the G-buffer after it.

  Pooling buried splats (below) changes the trade. On the curly groom at its
  full 50 k strands (6.8 M splats), forward shading of the groom pooled at 0.6
  (980 k splats) renders 2-5x faster than deferred shading of the unpooled one,
  and scores 55 dB (front) / 51 dB (close) against forward shading of the
  unpooled groom, where deferred shading scores 40 / 36 dB against it.

Strand width and opacity come from the per-point values in the `.hair` file, so
strands taper and see-through tips stay soft.

Open a `.hair` groom in the viewer like any other file (**Input**, or drag and
drop). It loads as strand-aligned splats, turns on hair shading and opacity
shadows, and a **Groom** section appears with the strand count, splats per
segment, width and opacity — each rebuilds the splats. Enable **Lighting** to
see the shading.

`examples/groom.rs` does the same from the command line and exercises all three
options, with `--bench N` for the table above:

```bash
cargo run --release --example groom -- assets/straight.hair --strands 10000 \
    --per-segment 3 --light --forward
cargo run --release --example groom -- assets/wCurly.hair --light --bench 30
```

It builds splats two ways. By default each strand segment becomes a splat
oriented by its tangent and shaped like it; that is what "strand aligned" means,
and it needs about 30x fewer splats than the other route. `--ribbons` instead
builds ribbon geometry and pushes it through the normal mesh pipeline, which
samples it on the conversion grid: the splats come out round and cell-sized, the
strand direction is lost, and 10k strands blow past the 7M splat budget. Grooms are also modelled at their own scale (tens of
units), so the example normalizes one into a 2-unit box — the renderer's near
and far planes, shadow bias and gaussian scale all assume a unit-ish model.

Hair models come from https://www.cemyuksel.com/research/hairmodels (free for
personal and research use, attribution requested).

### Pooling buried splats

A groom's splat count is dominated by strands nobody can see. A strand buried in
the volume contributes bulk opacity and colour but no silhouette and no
highlight, and the occlusion bake already says which those are. `merge_occluded`
pools them by cell into coarse splats that fill the same volume and stop the
same amount of light — the merged splat's opacity is set so that opacity times
area is preserved — while the shell keeps its per-segment detail.

Two things decide whether the groom still looks like itself afterwards.

**What counts as buried is relative to the groom.** The occlusion a splat reads
depends on how dense the groom around it is: the straight groom leaves its
outermost splats at ~0.4, while the 50 k-strand curly one buries even its own
silhouette below 0.3 (94% of its splats sit under 0.3, and its most open 10%
only reach 0.275). An absolute threshold therefore pools the deep interior of
one groom and the whole of another — which is what made a dense groom come back
as a dark, shiny blob. `relative_openness` is measured against the groom's own
most open splats (the 95th percentile of its occlusion), so the setting means
the same thing at any density and the shell is always spared. Both the CPU and
GPU poolers resolve it from the same 256-bin histogram, so they agree exactly.

**Clusters follow the strands.** Splats only pool with others pointing the same
way (`direction_bins`), and a cluster may run several cells along a strand but
only one across it (`along_cells`). A cluster that spans neighbouring strands
merges them into a ribbon and the groom loses its striping; one that runs along
a strand does not.

On the curly groom at its full 50 k strands (7 M strand splats, 1600x1200, PSNR
against the unpooled render):

| interior below | splats | PSNR (front / close) |
|---|---|---|
| none | 7.00 M | reference |
| 0.4 | 2.08 M (3.4x fewer) | 56.4 / 51.9 dB |
| 0.6 | 1.48 M (4.7x fewer) | 52.6 / 49.0 dB |
| 0.9 | 983 k (7.1x fewer) | 44.5 / 39.7 dB |

For comparison, the absolute threshold this replaced scored 35.9 / 30.1 dB on
the same groom. The sparser straight groom has far less true interior, so it
pools less for the same quality: 2.1x fewer splats at 33.9 dB at the default.
Pooling needs the occlusion bake first, and the button says so.

Pooling runs on the GPU (`src/gpu/pool.rs`): rather than accumulate per cell
with atomics — WGSL has no float atomics — it keys each buried splat by its
cell, sorts with the renderer's radix sorter, and gives one thread each run of
equal keys, which accumulates its whole cluster in registers. 3.4 M splats pool
in ~300 ms against ~790 ms for the CPU version in `src/merge.rs`, which stays as
the reference and the fallback.

### Merging similar splats

Optional, and not in the original. After conversion, blocks of 2x2 neighbouring
splats on the projection grid are replaced by one splat of twice the size
when their colour, normal, metallic/roughness and flatness stay within a
tolerance. Merged blocks can merge again, up to 16x16. Surfaces that overlap
in the projection (the front and back of a closed mesh) are kept apart by
depth, and surface edges keep their fine splats (`src/merge.rs`). One
**strength** knob sets the tolerances.

On DamagedHelmet (PSNR against the unmerged render, 1920x1080):

| | splats | PSNR (orbit views) | PSNR (close-up) |
|---|---|---|---|
| 520 px, no merge | 1.04 M | reference | reference |
| 520 px, strength 0.25 (default) | 545 k | 44–45 dB | 36 dB |
| 370 px, no merge (same count) | 528 k | 39–41 dB | 32 dB |
| 520 px, strength 0.5 | 398 k | 39–41 dB | 31 dB |
| 520 px, strength 1.0 | 277 k | 36–37 dB | 28 dB |

At high strength it is no better than lowering the sampling density, so the
useful range is roughly 0.1–0.5. Converting at a higher resolution and merging
also works: 1024 px at strength 1.0 gives 771 k splats at the quality of 520 px
unmerged (1.04 M).

Merging runs on the GPU (`src/gpu/merge.rs`, `shaders/merge.wgsl`): per
quadtree level, a radix sort groups the splats by cell and depth, and one
thread per depth layer tests and builds the merged splat. On an M1 it takes
about 55 ms for 1 M splats and 0.2-0.3 s for 4 M, versus 0.2 s and 1.2 s for
the multithreaded CPU version in `src/merge.rs`. The CPU version is the
fallback for GPUs with fewer than 9 storage buffers per shader stage and for
per-mesh projection boxes with a very large number of meshes; both give the
same result up to f32 rounding.

## Building

You need Rust 1.92 or newer and a GPU with Vulkan, Metal or DX12.

```bash
cargo build --release                         # GUI + CLI
cargo build --release --no-default-features   # CLI only (no windowing deps)
```

On Linux, the file dialogs go through the XDG desktop portal. If no portal is
running, type or paste paths into the text boxes, or drag and drop files.

## Usage

### GUI

```bash
mesh2splat                       # or: mesh2splat gui path/to/model.glb
```

The side panel mirrors the original's ImGui windows:

* **Input** — pick or drop a `.glb`, `.gltf`, `.ply` or `.spz` file.
* **Output** — choose the output folder, file name and format, then press **Save splat**:

| Format | Bytes / splat | Notes |
|---|---|---|
| Standard | 248 | Original 3DGS layout, including 45 zero SH coefficients |
| Standard SH0 | 68 | Same without the SH rest: ~3.6x smaller, still widely readable |
| PBR | 76 | Adds metallic / roughness (the original's own layout) |
| Compressed PBR | 48 | The original's quantised PBR layout |
| Compressed (PlayCanvas / SuperSplat) | 16 | Chunked quantised layout, ~15x smaller than Standard |
| SPZ (Niantic) | ≤ 20 | 20 bytes before gzip, which shrinks it further depending on the scene |

  The PlayCanvas layout groups splats into chunks of 256 that store the min/max of their positions, log-scales and colours; each splat is then four 32-bit words (position and scale 11/10/11 bits inside the chunk, rotation as smallest-three, 8-bit RGBA). It drops normals and PBR, so it is an export format for viewers rather than a round-trip for relighting. mesh2splat reads it back as well; on DamagedHelmet the round trip is 46 dB PSNR.

  SPZ ([nianticlabs/spz](https://github.com/nianticlabs/spz)) is written as version 3, the last gzip one (readers of version 4 still read it): 24-bit fixed-point positions, 8-bit opacity, colour and log-scale, and a 32-bit smallest-three rotation, stored attribute by attribute so gzip can squeeze them. It also drops normals and PBR. SPZ is defined in RUB axes (y up), the same as glTF, so splats are written as-is and come out upright in SPZ viewers. Tools that convert a 3DGS `.ply` to `.spz` assume the PLY is y-down and flip y and z; the export does not, so a `.spz` shows the other way up from the `.ply` of the same splats. Log scales are clamped to [-10, 6], so the thin axis of a flat splat comes out at e^-10 rather than ~0. `.spz` files (versions 2 and 3) load back in like a `.ply`.
* **Properties** — visualization mode (Final, Albedo, Depth, Normals, Geometry, Overdraw, PBR), mesh/gaussian depth test, gaussian scale, sampling density (16 up to 1024/2048/4096 px), projection box, **Detail-aware density** and **Merge similar splats** (see below), background color and split-screen.
* **Lighting** — point light with intensity, color and a cube shadow map, plus the fibre options below.
* **Camera** — switch between the original fly controls and Maya-style controls, and frame the model.
* **Gizmo** — translate, rotate or scale the model or the light (local or world axes), and set the on-screen gizmo size (`+` / `-` / `0` in the viewport).
* **Batch conversion** — pick a folder of meshes, optionally including subfolders, and convert them all.
* **Stats** — gaussian counts, conversion time, a GPU frame-time graph and a prepass / sort / splat raster breakdown (these need timestamp query support). The viewport only re-renders when the view or settings change; tick **Continuous redraw** to profile.

Camera controls default to Maya-style navigation around a pivot:

Hold `Alt` **or** `Cmd` (`Ctrl` off macOS) and drag:

| Input | Action |
|---|---|
| modifier + left drag | Tumble |
| modifier + middle drag, or `Alt` + `Cmd` + left drag | Pan |
| modifier + right drag (horizontal), mouse wheel, or trackpad pinch | Dolly |
| `F` | Frame the model, keeping the view direction |
| `Q` `W` `E` `R` | Gizmo: none / move / rotate / scale |
| `+` / `-` / `0` | Grow / shrink / reset the gizmo (bigger handles are easier to grab) |

Maya itself only uses `Alt`; `Cmd` / `Ctrl` works too because GNOME and KDE grab
`Alt` + drag to move windows. Without a modifier, a left drag belongs to the
transform gizmo, so the gizmo can stay on screen while you navigate: holding the
modifier gives the mouse to the camera even when the drag starts on a gizmo
handle, and a drag already under way keeps it until the button is released.

The camera reads the raw pointer state rather than the viewport's egui
`Response`, because `transform-gizmo-egui` registers its own interaction widget
under the cursor every frame and would otherwise swallow every camera drag
(`tests/viewport_input.rs` pins this down).

Select **Fly** in the **Camera** section for the original controls:

| Input | Action |
|---|---|
| Right mouse drag | Look around |
| `W` `A` `S` `D` | Move |
| `Q` / `E` | Down / up |
| `R` / `T` | Roll |
| `Shift` / `Ctrl` | Fast / slow |
| Mouse wheel | Field of view |
| `F` | Frame the model |



### CLI

```bash
# single file
mesh2splat convert model.glb -o model.ply --quality 0.5 --format standard
mesh2splat convert model.glb --resolution 2048 --format pbr --std 0.65

# SH0-only standard layout (much smaller, same visual result for converted meshes)
mesh2splat convert model.glb -o model.ply --format sh0
mesh2splat convert model.glb -o model.ply --format playcanvas   # ~15x smaller
mesh2splat convert model.glb --format spz                       # writes model.spz

# sample low-detail triangles on a coarser grid (optional tolerance, default 0.25)
mesh2splat convert model.glb -o model.ply --detail

# merge alike neighbouring splats (optional strength, default 0.25)
mesh2splat convert model.glb -o model.ply --merge
mesh2splat convert model.glb -o model.ply --resolution 1024 --merge 0.5

# batch (like the original's batch window)
mesh2splat convert --batch ./meshes -o ./splats --recursive --format compressed

# offscreen render to PNG (mesh inputs are converted first; .ply / .spz loaded as-is)
mesh2splat render model.glb -o shot.png --width 1280 --height 720 --orbit 35
mesh2splat render model.glb -o lit.png --light --light-intensity 20
mesh2splat render model.glb -o cmp.png --split --mode normal
mesh2splat render scene.ply  -o ply.png --mode albedo
```

`--quality q` maps to `resolution = 16 + q * (max_res - 16)`, the same as the UI slider. The
defaults (`q = 0.5`, `max_res = 1024`, giving 520 px, and `std = 0.65`) match the original.

### Benchmark

```bash
cargo run --release --example bench -- assets/DamagedHelmet.glb --res 520,1024 --size 1920x1080
cargo run --release --example bench -- --baseline target/bench/baseline   # also print PSNR vs saved PNGs
cargo run --release --example bench -- --merge 0.25 --baseline target/bench  # merged vs unmerged
```

For each sampling resolution this converts the model, renders five views and prints the splat count, median GPU time for prepass, sort and splat raster, and the PLY size of every export format.

### Library

```rust
use mesh2splat::{gpu::*, scene, ply, PlyFormat};

let ctx = GpuContext::new_headless()?;
let s = scene::load_gltf("model.glb")?;
let gpu_scene = GpuScene::upload(&ctx, &s);
let mut gaussians = GaussianBuffer::new_empty(&ctx);
Converter::new(&ctx).convert(&ctx, &gpu_scene, ConvertSettings::default(), &mut gaussians);
ply::write_ply("model.ply", &gaussians.download(&ctx), PlyFormat::Standard, gaussians.scale_multiplier(0.65))?;
```

## Architecture

```
src/
  scene.rs          glTF loading (node transforms baked, normals/tangents like the original)
  ply.rs            PLY writers (5 layouts) and reader (standard / PBR / compressed / PlayCanvas)
  spz.rs            Niantic .spz writer and reader
  camera.rs         fly camera (port of Camera.cpp) + Maya tumble / pan / dolly
  hair.rs           .hair grooms and strand-aligned splats
  merge.rs          optional quadtree merge of alike neighbouring splats (CPU reference)
  cli.rs, main.rs   clap CLI
  app.rs            eframe/egui UI (port of ImGuiUI + GuiRendererConcreteMediator)
  gpu/
    converter.rs    ConversionPass (one pass per sampling level)
    shaders/lighting.wgsl  shading shared by the deferred and forward paths
    scene.rs        vertex buffers, textures (CPU mip chain), per-mesh bind groups
    ao.rs           bakes occlusion and a bent normal into the splats
    pool.rs         pools buried splats into coarse volume-filling ones
    sort.rs         GPU radix sort (replaces gl-radix-sort)
    merge.rs        GPU version of the splat merge
    renderer.rs     all render passes, uniforms, G-buffers, readback
    shaders/*.wgsl  ports of the GLSL shaders
```

These are the frame passes in order, the same as the original except for the shadow map:

1. Mesh depth prepass (optional).
2. Mesh G-buffer (split-screen only).
3. Point-light cube shadow map. The original drew it after the splats, which
   is fine for deferred shading, but forward shading reads it during the splat
   pass and would see the previous frame's.
4. Gaussian prepass (projection, culling, EWA 2D covariance).
5. Radix sort by view depth.
6. Instanced quads, blended front-to-back with `(ONE_MINUS_DST_ALPHA, ONE)` into the G-buffer.
7. Deferred GGX shading.

The sort is fully GPU-driven. The visible count stays on the GPU, and the sort
and draw are dispatched indirectly, so a frame never waits on a CPU readback.
Buffers are sized to the real gaussian count, not a fixed 7M reserve.

## Differences from the original

Most of the port is faithful, including the math, data layout, file formats and UI
defaults. The deliberate deviations are listed here.

**Bugs fixed in the original**

* **Metallic channel.** The deferred shader read metallic from the `.b` G-buffer
  channel, which is always 0. It now reads the channel the splat pass writes.
* **Diffuse brightness.** `#define PI 22.0f/7.0f` has no parentheses, so
  `albedo / PI` evaluated as `albedo / 22 / 7` and diffuse lighting was about 49×
  too dark. This port uses π.
* **Opacity overflow.** `invSigmoid(1.0)` overflows to `+inf` in f32, so opaque
  splats were written with infinite opacity. Opacity is now clamped to a large
  finite logit.
* **Normalized G-buffer.** Accumulated G-buffer values (position, normal, PBR) are
  divided by their accumulated alpha. Normals are premultiplied by opacity so this
  gives a proper weighted average. Before, edges were pulled toward zero.
* **Shadow quad size.** Shadow-map splat quads are sized for the shadow map's
  resolution, not the window's.
* **Multi-mesh bounding box.** The per-mesh box was accumulated over earlier meshes,
  which was an accident. You now choose **Scene** (uniform density, the default) or
  **Per-mesh**.
* **NaN footprint.** A perfectly isotropic 2D footprint produced a NaN axis. This is
  now guarded.

**Behaviour changes**

* **Background.** The background color is composited using accumulated alpha.
* **Final mode without lighting.** "Final" with lighting off shows unlit albedo.
  The original showed ambient-only shading.
* **Octahedral normals.** The compressed layout uses the reference per-component
  sign (knarkowicz). The original's joint-sign variant is lossy for lower-hemisphere
  normals with mixed signs.
* **Re-exported PLYs.** A loaded PLY is re-exported with scale multiplier 1. The
  original applied the mesh multiplier to it too.
* **Loading PLYs.** The compressed-PBR layout can be read back in.
* **Model formats.** `.gltf` is accepted as well as `.glb`, including 16-bit and
  grayscale textures.
* **Capacity.** The gaussian limit is `min(7M, the device's storage-binding limit / 96 bytes)`.
  If a conversion hits it, you get a warning.
* **Removed.** Shader hot-reload is gone, because shaders are embedded at compile time.

## Tests

```bash
cargo test --no-default-features
```

GPU tests run on any adapter, including the Mesa `lavapipe` software driver in CI.
They cover the following:

* The radix sort against a CPU stable sort, including the special sizes and stability.
* An analytic quad conversion: exact gaussian count, unit Jacobian scales, and the rotation frame.
* Both bounding-box modes and multiple meshes.
* Rendered pixel values in each mode, and culling.
* Lighting and shadows.
* A textured glTF conversion through export and reload in every export format.

The `examples/` folder has small scenes for eyeballing orientation and shadows.

## License

BSD-3-Clause, the same terms as the original. See `LICENSE.txt`. EA's copyright
notice is kept because this is a derivative port. EA's name and logos must not be
used to endorse derived products, and no EA/SEED logos are included.
