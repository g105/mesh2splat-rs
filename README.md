# mesh2splat (Rust / wgpu)

A Rust port of [EA SEED's Mesh2Splat](https://github.com/electronicarts/mesh2splat):
fast conversion of textured triangle meshes (`.glb` / `.gltf`) into 3D Gaussian
Splatting `.ply` files, plus the real-time splat renderer from the original
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

* **Input** — pick or drop a `.glb`, `.gltf` or `.ply` file.
* **Output** — choose the output folder, file name and format (Standard / PBR / Compressed PBR), then press **Save splat**.
* **Properties** — visualization mode (Final, Albedo, Depth, Normals, Geometry, Overdraw, PBR), mesh/gaussian depth test, gaussian scale, sampling density (16 up to 1024/2048/4096 px), projection box, background color and split-screen.
* **Lighting** — point light with intensity, color and a cube shadow map.
* **Gizmo** — translate, rotate or scale the model or the light (local or world axes).
* **Batch conversion** — pick a folder of meshes, optionally including subfolders, and convert them all.
* **Stats** — gaussian counts, conversion time, and a GPU frame-time graph (the graph needs timestamp query support).

Camera controls are the same as the original:

| Input | Action |
|---|---|
| Right mouse drag | Look around |
| `W` `A` `S` `D` | Move |
| `Q` / `E` | Down / up |
| `R` / `T` | Roll |
| `Shift` / `Ctrl` | Fast / slow |
| Mouse wheel | Field of view |

### CLI

```bash
# single file
mesh2splat convert model.glb -o model.ply --quality 0.5 --format standard
mesh2splat convert model.glb --resolution 2048 --format pbr --std 0.65

# batch (like the original's batch window)
mesh2splat convert --batch ./meshes -o ./splats --recursive --format compressed

# offscreen render to PNG (mesh inputs are converted first; .ply loaded as-is)
mesh2splat render model.glb -o shot.png --width 1280 --height 720 --orbit 35
mesh2splat render model.glb -o lit.png --light --light-intensity 20
mesh2splat render model.glb -o cmp.png --split --mode normal
mesh2splat render scene.ply  -o ply.png --mode albedo
```

`--quality q` maps to `resolution = 16 + q * (max_res - 16)`, the same as the UI slider. The
defaults (`q = 0.5`, `max_res = 1024`, giving 520 px, and `std = 0.65`) match the original.

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
  ply.rs            PLY writer (3 layouts) and reader (standard / PBR / compressed)
  camera.rs         fly camera (port of Camera.cpp)
  cli.rs, main.rs   clap CLI
  app.rs            eframe/egui UI (port of ImGuiUI + GuiRendererConcreteMediator)
  gpu/
    converter.rs    ConversionPass
    scene.rs        vertex buffers, textures (CPU mip chain), per-mesh bind groups
    sort.rs         GPU radix sort (replaces gl-radix-sort)
    renderer.rs     all render passes, uniforms, G-buffers, readback
    shaders/*.wgsl  ports of the GLSL shaders
```

These are the frame passes in order, the same as the original:

1. Mesh depth prepass (optional).
2. Mesh G-buffer (split-screen only).
3. Gaussian prepass (projection, culling, EWA 2D covariance).
4. Radix sort by view depth.
5. Instanced quads, blended front-to-back with `(ONE_MINUS_DST_ALPHA, ONE)` into the G-buffer.
6. Point-light cube shadow map.
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
* A textured glTF conversion through export and reload in all three PLY layouts.

The `examples/` folder has small scenes for eyeballing orientation and shadows.

## License

BSD-3-Clause, the same terms as the original. See `LICENSE.txt`. EA's copyright
notice is kept because this is a derivative port. EA's name and logos must not be
used to endorse derived products, and no EA/SEED logos are included.
