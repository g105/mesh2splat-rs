//! 3DGS `.ply` reading and writing. Port of `parsers.cpp`.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use glam::{Quat, Vec2, Vec3};

use crate::types::*;

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write gaussians to `path`.
///
/// `scale_multiplier` converts stored scales to world-space standard
/// deviations: `gaussian_std / resolution` for converted meshes, `1.0` for
/// gaussians that were loaded from a PLY file.
pub fn write_ply(
    path: impl AsRef<Path>,
    gaussians: &[GaussianVertex],
    format: PlyFormat,
    scale_multiplier: f32,
) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    match format {
        PlyFormat::Standard => write_standard(&mut w, gaussians, scale_multiplier, true)?,
        PlyFormat::StandardSh0 => write_standard(&mut w, gaussians, scale_multiplier, false)?,
        PlyFormat::Pbr => write_pbr(&mut w, gaussians, scale_multiplier)?,
        PlyFormat::CompressedPbr => write_compressed(&mut w, gaussians, scale_multiplier)?,
        PlyFormat::PlayCanvas => write_playcanvas(&mut w, gaussians, scale_multiplier)?,
    }
    w.flush()?;
    Ok(())
}

fn header(w: &mut impl Write, count: usize, props: &[(&str, &str)]) -> Result<()> {
    writeln!(w, "ply")?;
    writeln!(w, "format binary_little_endian 1.0")?;
    writeln!(w, "element vertex {count}")?;
    for (ty, name) in props {
        writeln!(w, "property {ty} {name}")?;
    }
    writeln!(w, "end_header")?;
    Ok(())
}

#[inline]
fn f(w: &mut impl Write, v: f32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn log_scales(g: &GaussianVertex, m: f32) -> [f32; 3] {
    [
        (g.scale[0] * m).ln(),
        (g.scale[1] * m).ln(),
        (g.scale[2] * m).ln(),
    ]
}

/// `sh_rest`: also write the 45 (zero) higher-order SH coefficients.
fn write_standard(w: &mut impl Write, gs: &[GaussianVertex], m: f32, sh_rest: bool) -> Result<()> {
    let mut props: Vec<(String, String)> = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .iter()
    .map(|n| ("float".to_string(), n.to_string()))
    .collect();
    let rest = if sh_rest { 45 } else { 0 };
    for i in 0..rest {
        props.push(("float".into(), format!("f_rest_{i}")));
    }
    for n in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        props.push(("float".into(), n.into()));
    }
    let props_ref: Vec<(&str, &str)> = props
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    header(w, gs.len(), &props_ref)?;
    let zeros = vec![0u8; rest * 4];
    for g in gs {
        for v in &g.position[..3] {
            f(w, *v)?;
        }
        for v in &g.normal[..3] {
            f(w, *v)?;
        }
        let sh = sh_from_color(Vec3::from_slice(&g.color[..3]));
        for v in sh.to_array() {
            f(w, v)?;
        }
        w.write_all(&zeros)?;
        f(w, inv_sigmoid(g.color[3]))?;
        for v in log_scales(g, m) {
            f(w, v)?;
        }
        for v in g.rotation {
            f(w, v)?;
        }
    }
    Ok(())
}

fn write_pbr(w: &mut impl Write, gs: &[GaussianVertex], m: f32) -> Result<()> {
    let names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "metallicFactor",
        "roughnessFactor",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ];
    let props: Vec<(&str, &str)> = names.iter().map(|n| ("float", *n)).collect();
    header(w, gs.len(), &props)?;
    for g in gs {
        for v in &g.position[..3] {
            f(w, *v)?;
        }
        for v in &g.normal[..3] {
            f(w, *v)?;
        }
        let sh = sh_from_color(Vec3::from_slice(&g.color[..3]));
        for v in sh.to_array() {
            f(w, v)?;
        }
        f(w, g.pbr[0])?;
        f(w, g.pbr[1])?;
        f(w, inv_sigmoid(g.color[3]))?;
        for v in log_scales(g, m) {
            f(w, v)?;
        }
        for v in g.rotation {
            f(w, v)?;
        }
    }
    Ok(())
}

fn to_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Octahedral normal encoding to [0,1]^2
/// (<https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/>).
pub fn encode_octa(n: Vec3) -> Vec2 {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs() + 1e-8);
    let mut r = Vec2::new(n.x, n.y);
    if n.z < 0.0 {
        let sign = Vec2::new(
            if n.x >= 0.0 { 1.0 } else { -1.0 },
            if n.y >= 0.0 { 1.0 } else { -1.0 },
        );
        r = (Vec2::ONE - Vec2::new(n.y.abs(), n.x.abs())) * sign;
    }
    r * 0.5 + Vec2::splat(0.5)
}

pub fn decode_octa(e: Vec2) -> Vec3 {
    let f = e * 2.0 - Vec2::ONE;
    let mut n = Vec3::new(f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
    let t = (-n.z).clamp(0.0, 1.0);
    n.x += if n.x >= 0.0 { -t } else { t };
    n.y += if n.y >= 0.0 { -t } else { t };
    n.normalize_or_zero()
}

// --- PlayCanvas / SuperSplat compressed PLY -----------------------------------
//
// Splats are grouped into chunks of 256; each chunk stores the min/max of the
// positions, log-scales and colours it contains, and every splat is four u32s:
// position and scale normalized inside the chunk (11/10/11 bits), the rotation
// as the smallest-three quaternion encoding (2 bits largest index + 3 x 10) and
// an 8-bit RGBA colour. ~16 bytes per splat.
// Layout mirrors playcanvas/splat-transform (compressed-chunk.ts).

/// Splats per chunk.
const PC_CHUNK: usize = 256;

fn pack_unorm(v: f32, bits: u32) -> u32 {
    let t = ((1u32 << bits) - 1) as f32;
    (v * t + 0.5).floor().clamp(0.0, t) as u32
}

fn pack111011(x: f32, y: f32, z: f32) -> u32 {
    pack_unorm(x, 11) << 21 | pack_unorm(y, 10) << 11 | pack_unorm(z, 11)
}

fn pack8888(x: f32, y: f32, z: f32, w: f32) -> u32 {
    pack_unorm(x, 8) << 24 | pack_unorm(y, 8) << 16 | pack_unorm(z, 8) << 8 | pack_unorm(w, 8)
}

/// Smallest-three quaternion encoding: 2 bits for the index of the largest
/// component, then the other three in increasing index order, 10 bits each.
/// `q` is in the file's own component order (rot_0..rot_3).
fn pack_rotation(q: [f32; 4]) -> u32 {
    let len = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let mut a = if len > 0.0 { q.map(|v| v / len) } else { [1.0, 0.0, 0.0, 0.0] };
    let largest = (0..4).max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs())).unwrap();
    if a[largest] < 0.0 {
        a = a.map(|v| -v);
    }
    let norm = std::f32::consts::SQRT_2 * 0.5;
    let mut out = largest as u32;
    for (i, v) in a.iter().enumerate() {
        if i != largest {
            out = (out << 10) | pack_unorm(v * norm + 0.5, 10);
        }
    }
    out
}

fn normalize_in(x: f32, min: f32, max: f32) -> f32 {
    if x <= min {
        0.0
    } else if x >= max {
        1.0
    } else if max - min < 1e-5 {
        0.0
    } else {
        (x - min) / (max - min)
    }
}

/// Per-chunk min/max of position, log-scale and colour (the 18 chunk floats).
fn chunk_bounds(chunk: &[GaussianVertex], m: f32) -> [f32; 18] {
    let (mut lo, mut hi) = ([f32::MAX; 9], [f32::MIN; 9]);
    for g in chunk {
        let s = log_scales(g, m);
        let vals = [
            g.position[0], g.position[1], g.position[2],
            s[0].clamp(-20.0, 20.0), s[1].clamp(-20.0, 20.0), s[2].clamp(-20.0, 20.0),
            g.color[0], g.color[1], g.color[2],
        ];
        for i in 0..9 {
            lo[i] = lo[i].min(vals[i]);
            hi[i] = hi[i].max(vals[i]);
        }
    }
    // min xyz, max xyz, min scale xyz, max scale xyz, min rgb, max rgb
    let mut out = [0f32; 18];
    for (dst, src) in out.chunks_exact_mut(3).zip([&lo[0..3], &hi[0..3], &lo[3..6], &hi[3..6], &lo[6..9], &hi[6..9]]) {
        dst.copy_from_slice(src);
    }
    out
}

fn write_playcanvas(w: &mut impl Write, gs: &[GaussianVertex], m: f32) -> Result<()> {
    let chunks: Vec<[f32; 18]> = gs.chunks(PC_CHUNK).map(|c| chunk_bounds(c, m)).collect();
    writeln!(w, "ply")?;
    writeln!(w, "format binary_little_endian 1.0")?;
    writeln!(w, "comment Generated by mesh2splat")?;
    writeln!(w, "element chunk {}", chunks.len())?;
    for n in [
        "min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
        "min_scale_x", "min_scale_y", "min_scale_z",
        "max_scale_x", "max_scale_y", "max_scale_z",
        "min_r", "min_g", "min_b", "max_r", "max_g", "max_b",
    ] {
        writeln!(w, "property float {n}")?;
    }
    writeln!(w, "element vertex {}", gs.len())?;
    for n in [
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    ] {
        writeln!(w, "property uint {n}")?;
    }
    writeln!(w, "end_header")?;

    for c in &chunks {
        for v in c {
            f(w, *v)?;
        }
    }
    for (chunk, b) in gs.chunks(PC_CHUNK).zip(&chunks) {
        for g in chunk {
            let s = log_scales(g, m);
            let words = [
                pack111011(
                    normalize_in(g.position[0], b[0], b[3]),
                    normalize_in(g.position[1], b[1], b[4]),
                    normalize_in(g.position[2], b[2], b[5]),
                ),
                pack_rotation(g.rotation),
                pack111011(
                    normalize_in(s[0], b[6], b[9]),
                    normalize_in(s[1], b[7], b[10]),
                    normalize_in(s[2], b[8], b[11]),
                ),
                pack8888(
                    normalize_in(g.color[0], b[12], b[15]),
                    normalize_in(g.color[1], b[13], b[16]),
                    normalize_in(g.color[2], b[14], b[17]),
                    g.color[3],
                ),
            ];
            for v in words {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn write_compressed(w: &mut impl Write, gs: &[GaussianVertex], m: f32) -> Result<()> {
    let props = [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("uint8", "red"),
        ("uint8", "green"),
        ("uint8", "blue"),
        ("uint8", "opacity"),
        ("float", "rot_0"),
        ("float", "rot_1"),
        ("float", "rot_2"),
        ("float", "rot_3"),
        ("float", "scale_0"),
        ("float", "scale_1"),
        ("float", "scale_2"),
        ("uint8", "octa_nx"),
        ("uint8", "octa_ny"),
        ("uint8", "roughness"),
        ("uint8", "metallic"),
    ];
    header(w, gs.len(), &props)?;
    for g in gs {
        for v in &g.position[..3] {
            f(w, *v)?;
        }
        w.write_all(&[
            to_byte(g.color[0]),
            to_byte(g.color[1]),
            to_byte(g.color[2]),
            to_byte(g.color[3]),
        ])?;
        for v in g.rotation {
            f(w, v)?;
        }
        // Same as the original: the thin axis gets min(sx, sy) so the splat has volume.
        let min_xy = g.scale[0].min(g.scale[1]);
        f(w, (g.scale[0] * m).ln())?;
        f(w, (g.scale[1] * m).ln())?;
        f(w, (min_xy * m).ln())?;
        let o = encode_octa(Vec3::from_slice(&g.normal[..3]));
        w.write_all(&[
            to_byte(o.x),
            to_byte(o.y),
            to_byte(g.pbr[1]),
            to_byte(g.pbr[0]),
        ])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    Ascii,
    BinaryLe,
    BinaryBe,
}

#[derive(Clone, Copy, Debug)]
#[derive(PartialEq, Eq)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => bail!("unknown PLY property type '{s}'"),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn read(self, b: &[u8], le: bool) -> f64 {
        macro_rules! rd {
            ($t:ty, $n:expr) => {{
                let a: [u8; $n] = b[..$n].try_into().unwrap();
                (if le {
                    <$t>::from_le_bytes(a)
                } else {
                    <$t>::from_be_bytes(a)
                }) as f64
            }};
        }
        match self {
            Self::I8 => b[0] as i8 as f64,
            Self::U8 => b[0] as f64,
            Self::I16 => rd!(i16, 2),
            Self::U16 => rd!(u16, 2),
            Self::I32 => rd!(i32, 4),
            Self::U32 => rd!(u32, 4),
            Self::F32 => rd!(f32, 4),
            Self::F64 => rd!(f64, 8),
        }
    }
}

struct Element {
    name: String,
    count: usize,
    props: Vec<(String, ScalarType)>,
}

/// Column-oriented PLY vertex data.
pub struct PlyVertexData {
    pub count: usize,
    names: Vec<String>,
    columns: Vec<Vec<f32>>,
    /// Exact values of `uint` properties: f32 columns cannot hold a packed
    /// 32-bit word without losing its low bits.
    words: Vec<Vec<u32>>,
    /// Per-chunk bounds of the PlayCanvas compressed layout.
    chunks: Option<Box<PlyVertexData>>,
}

impl PlyVertexData {
    pub fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.columns[i].as_slice())
    }

    /// Exact `uint` column, for the packed words of the compressed layout.
    fn req_u32(&self, name: &str) -> Result<&[u32]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.words[i].as_slice())
            .filter(|c| !c.is_empty())
            .ok_or_else(|| anyhow!("PLY file is missing uint property '{name}'"))
    }

    fn req(&self, name: &str) -> Result<&[f32]> {
        self.get(name)
            .ok_or_else(|| anyhow!("PLY file is missing vertex property '{name}'"))
    }
}

/// Read the `vertex` element of a PLY file into float columns.
pub fn read_ply_vertices(path: impl AsRef<Path>) -> Result<PlyVertexData> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, file);

    let mut line = String::new();
    r.read_line(&mut line)?;
    if line.trim() != "ply" {
        bail!("{} is not a PLY file", path.display());
    }
    let mut encoding = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            bail!("unexpected end of PLY header");
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        match toks.as_slice() {
            ["format", fmt, _ver] => {
                encoding = Some(match *fmt {
                    "ascii" => Encoding::Ascii,
                    "binary_little_endian" => Encoding::BinaryLe,
                    "binary_big_endian" => Encoding::BinaryBe,
                    _ => bail!("unsupported PLY format '{fmt}'"),
                })
            }
            ["element", name, count] => elements.push(Element {
                name: name.to_string(),
                count: count.parse().context("bad element count")?,
                props: vec![],
            }),
            ["property", "list", ..] => {
                let el = elements
                    .last()
                    .ok_or_else(|| anyhow!("property before element"))?;
                if el.name == "vertex" {
                    bail!("list properties in the vertex element are not supported");
                }
                bail!(
                    "PLY list properties (element '{}') are not supported",
                    el.name
                );
            }
            ["property", ty, name] => {
                let el = elements
                    .last_mut()
                    .ok_or_else(|| anyhow!("property before element"))?;
                el.props.push((name.to_string(), ScalarType::parse(ty)?));
            }
            ["end_header"] => break,
            _ => {} // comment / obj_info / blank
        }
    }
    let encoding = encoding.ok_or_else(|| anyhow!("PLY header has no format line"))?;

    let mut result = None;
    let mut chunks = None;
    for el in &elements {
        // The PlayCanvas compressed layout keeps per-chunk bounds in a second element.
        let is_vertex = el.name == "vertex" || el.name == "chunk";
        let mut columns: Vec<Vec<f32>> = if is_vertex {
            vec![Vec::with_capacity(el.count); el.props.len()]
        } else {
            vec![]
        };
        let mut words: Vec<Vec<u32>> = vec![Vec::new(); if is_vertex { el.props.len() } else { 0 }];
        match encoding {
            Encoding::Ascii => {
                for _ in 0..el.count {
                    line.clear();
                    r.read_line(&mut line)?;
                    if is_vertex {
                        for (i, tok) in line.split_whitespace().take(el.props.len()).enumerate() {
                            columns[i].push(tok.parse::<f32>().context("bad ascii PLY value")?);
                            if el.props[i].1 == ScalarType::U32 {
                                words[i].push(tok.parse::<u32>().context("bad ascii PLY value")?);
                            }
                        }
                    }
                }
            }
            Encoding::BinaryLe | Encoding::BinaryBe => {
                let le = encoding == Encoding::BinaryLe;
                let stride: usize = el.props.iter().map(|p| p.1.size()).sum();
                let offsets: Vec<usize> = el
                    .props
                    .iter()
                    .scan(0, |acc, p| {
                        let o = *acc;
                        *acc += p.1.size();
                        Some(o)
                    })
                    .collect();
                // Read in chunks to bound memory.
                let chunk_rows = (8 << 20) / stride.max(1);
                let mut remaining = el.count;
                let mut buf = vec![0u8; chunk_rows.min(el.count).max(1) * stride];
                while remaining > 0 {
                    let rows = remaining.min(chunk_rows);
                    let bytes = &mut buf[..rows * stride];
                    r.read_exact(bytes).context("PLY file is truncated")?;
                    if is_vertex {
                        for row in bytes.chunks_exact(stride) {
                            for (i, (_, ty)) in el.props.iter().enumerate() {
                                let v = ty.read(&row[offsets[i]..], le);
                                columns[i].push(v as f32);
                                if *ty == ScalarType::U32 {
                                    let b = &row[offsets[i]..offsets[i] + 4];
                                    let b: [u8; 4] = b.try_into().unwrap();
                                    words[i].push(if le {
                                        u32::from_le_bytes(b)
                                    } else {
                                        u32::from_be_bytes(b)
                                    });
                                }
                            }
                        }
                    }
                    remaining -= rows;
                }
            }
        }
        if is_vertex {
            let data = PlyVertexData {
                count: el.count,
                names: el.props.iter().map(|p| p.0.clone()).collect(),
                columns,
                words,
                chunks: None,
            };
            if el.name == "chunk" {
                chunks = Some(data);
            } else {
                result = Some(data);
                break;
            }
        }
    }
    let mut vertices = result.ok_or_else(|| anyhow!("PLY file has no 'vertex' element"))?;
    vertices.chunks = chunks.map(Box::new);
    Ok(vertices)
}

/// Result of loading a 3DGS PLY file.
pub struct LoadedPly {
    pub gaussians: Vec<GaussianVertex>,
    /// True when the file carried normals + metallic/roughness, so it can be relit.
    pub has_pbr: bool,
}

fn unpack_unorm(v: u32, bits: u32) -> f32 {
    v as f32 / ((1u32 << bits) - 1) as f32
}

fn unpack111011(v: u32) -> [f32; 3] {
    [
        unpack_unorm((v >> 21) & 0x7ff, 11),
        unpack_unorm((v >> 11) & 0x3ff, 10),
        unpack_unorm(v & 0x7ff, 11),
    ]
}

/// Inverse of [`pack_rotation`], in the file's own component order.
fn unpack_rotation(v: u32) -> [f32; 4] {
    let norm = 1.0 / (std::f32::consts::SQRT_2 * 0.5);
    let abc = [
        (unpack_unorm((v >> 20) & 0x3ff, 10) - 0.5) * norm,
        (unpack_unorm((v >> 10) & 0x3ff, 10) - 0.5) * norm,
        (unpack_unorm(v & 0x3ff, 10) - 0.5) * norm,
    ];
    let largest = (v >> 30) as usize;
    let m = (1.0 - abc.iter().map(|v| v * v).sum::<f32>()).max(0.0).sqrt();
    let mut out = [0f32; 4];
    let mut k = 0;
    for (i, o) in out.iter_mut().enumerate() {
        if i == largest {
            *o = m;
        } else {
            *o = abc[k];
            k += 1;
        }
    }
    out
}

/// Decode the PlayCanvas / SuperSplat compressed layout.
fn load_playcanvas(d: &PlyVertexData) -> Result<Vec<GaussianVertex>> {
    let chunks = d
        .chunks
        .as_ref()
        .ok_or_else(|| anyhow!("compressed PLY has no 'chunk' element"))?;
    let bound = |name: &str| -> Result<&[f32]> {
        chunks
            .get(name)
            .ok_or_else(|| anyhow!("compressed PLY chunk is missing '{name}'"))
    };
    let names = [
        "min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
        "min_scale_x", "min_scale_y", "min_scale_z",
        "max_scale_x", "max_scale_y", "max_scale_z",
        "min_r", "min_g", "min_b", "max_r", "max_g", "max_b",
    ];
    let b: Vec<&[f32]> = names.iter().map(|n| bound(n)).collect::<Result<_>>()?;
    let word = |name: &str| -> Result<&[u32]> { d.req_u32(name) };
    let (pos, rot, scale, col) = (
        word("packed_position")?,
        word("packed_rotation")?,
        word("packed_scale")?,
        word("packed_color")?,
    );

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let mut out = Vec::with_capacity(d.count);
    for i in 0..d.count {
        let c = i / 256;
        if c >= chunks.count {
            bail!("compressed PLY has too few chunks for {} splats", d.count);
        }
        let p = unpack111011(pos[i]);
        let s = unpack111011(scale[i]);
        let q = unpack_rotation(rot[i]);
        let cw = col[i];
        let rgba = [
            unpack_unorm((cw >> 24) & 0xff, 8),
            unpack_unorm((cw >> 16) & 0xff, 8),
            unpack_unorm((cw >> 8) & 0xff, 8),
            unpack_unorm(cw & 0xff, 8),
        ];
        out.push(GaussianVertex {
            position: [
                lerp(b[0][c], b[3][c], p[0]),
                lerp(b[1][c], b[4][c], p[1]),
                lerp(b[2][c], b[5][c], p[2]),
                1.0,
            ],
            color: [
                lerp(b[12][c], b[15][c], rgba[0]),
                lerp(b[13][c], b[16][c], rgba[1]),
                lerp(b[14][c], b[17][c], rgba[2]),
                rgba[3],
            ],
            scale: [
                lerp(b[6][c], b[9][c], s[0]).exp(),
                lerp(b[7][c], b[10][c], s[1]).exp(),
                lerp(b[8][c], b[11][c], s[2]).exp(),
                0.0,
            ],
            normal: [0.0; 4],
            rotation: q,
            pbr: [DEFAULT_METALLIC, DEFAULT_ROUGHNESS, 0.0, 1.0],
        });
    }
    Ok(out)
}

/// Load a 3DGS `.ply` (standard, PBR, or compressed-PBR layout).
/// Scales are exponentiated and opacities passed through a sigmoid, so the
/// result uses [`SourceFormat::Ply`] semantics.
pub fn load_gaussian_ply(path: impl AsRef<Path>) -> Result<LoadedPly> {
    let d = read_ply_vertices(path)?;
    if d.has("packed_position") {
        return Ok(LoadedPly {
            gaussians: load_playcanvas(&d)?,
            has_pbr: false,
        });
    }
    let n = d.count;
    let (x, y, z) = (d.req("x")?, d.req("y")?, d.req("z")?);
    let (s0, s1, s2) = (d.req("scale_0")?, d.req("scale_1")?, d.req("scale_2")?);
    let (r0, r1, r2, r3) = (
        d.req("rot_0")?,
        d.req("rot_1")?,
        d.req("rot_2")?,
        d.req("rot_3")?,
    );

    let mut gaussians = Vec::with_capacity(n);

    if d.has("octa_nx") {
        // Compressed PBR layout written by this tool / the original.
        let (red, green, blue, op) = (
            d.req("red")?,
            d.req("green")?,
            d.req("blue")?,
            d.req("opacity")?,
        );
        let (ox, oy) = (d.req("octa_nx")?, d.req("octa_ny")?);
        let (rough, metal) = (d.req("roughness")?, d.req("metallic")?);
        for i in 0..n {
            let q = Quat::from_xyzw(r1[i], r2[i], r3[i], r0[i]).normalize();
            let nrm = decode_octa(Vec2::new(ox[i], oy[i]) / 255.0);
            gaussians.push(GaussianVertex {
                position: [x[i], y[i], z[i], 1.0],
                color: [
                    red[i] / 255.0,
                    green[i] / 255.0,
                    blue[i] / 255.0,
                    op[i] / 255.0,
                ],
                scale: [s0[i].exp(), s1[i].exp(), s2[i].exp(), 1.0],
                normal: [nrm.x, nrm.y, nrm.z, 0.0],
                rotation: [q.w, q.x, q.y, q.z],
                pbr: [metal[i] / 255.0, rough[i] / 255.0, 0.0, 0.0],
            });
        }
        return Ok(LoadedPly {
            gaussians,
            has_pbr: true,
        });
    }

    let (dc0, dc1, dc2) = (d.req("f_dc_0")?, d.req("f_dc_1")?, d.req("f_dc_2")?);
    let op = d.req("opacity")?;
    let normals = match (d.get("nx"), d.get("ny"), d.get("nz")) {
        (Some(a), Some(b), Some(c)) => Some((a, b, c)),
        _ => None,
    };
    let pbr = match (d.get("metallicFactor"), d.get("roughnessFactor")) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    };
    let has_pbr = normals.is_some() && pbr.is_some();

    for i in 0..n {
        let c = color_from_sh(Vec3::new(dc0[i], dc1[i], dc2[i]));
        let q = Quat::from_xyzw(r1[i], r2[i], r3[i], r0[i]).normalize();
        let (normal, pbr_v) = if has_pbr {
            let (nx, ny, nz) = normals.unwrap();
            let (m, r) = pbr.unwrap();
            ([nx[i], ny[i], nz[i], 0.0], [m[i], r[i], 0.0, 0.0])
        } else {
            ([0.0; 4], [0.0; 4])
        };
        gaussians.push(GaussianVertex {
            position: [x[i], y[i], z[i], 1.0],
            color: [c.x, c.y, c.z, sigmoid(op[i])],
            scale: [s0[i].exp(), s1[i].exp(), s2[i].exp(), 1.0],
            normal,
            rotation: [q.w, q.x, q.y, q.z],
            pbr: pbr_v,
        });
    }
    Ok(LoadedPly { gaussians, has_pbr })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<GaussianVertex> {
        (0..10)
            .map(|i| {
                let t = i as f32 / 10.0;
                let q = Quat::from_rotation_y(t * 3.0) * Quat::from_rotation_x(t);
                let n = Vec3::new(t - 0.5, 0.3, -0.7).normalize();
                GaussianVertex {
                    position: [t, 2.0 * t, -t, 1.0],
                    color: [t, 1.0 - t, 0.5, 0.2 + 0.7 * t],
                    scale: [0.5 + t, 0.25 + t, 0.1 + t, 0.0],
                    normal: [n.x, n.y, n.z, 0.0],
                    rotation: [q.w, q.x, q.y, q.z],
                    pbr: [t, 1.0 - t, 0.0, 1.0],
                }
            })
            .collect()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("m2s_test_{}_{name}", std::process::id()))
    }

    #[test]
    fn standard_roundtrip() {
        let gs = sample();
        let p = tmp("std.ply");
        write_ply(&p, &gs, PlyFormat::Standard, 2.0).unwrap();
        let back = load_gaussian_ply(&p).unwrap();
        assert!(!back.has_pbr);
        assert_eq!(back.gaussians.len(), gs.len());
        for (a, b) in gs.iter().zip(&back.gaussians) {
            for k in 0..3 {
                assert!((a.position[k] - b.position[k]).abs() < 1e-6);
                assert!((a.color[k] - b.color[k]).abs() < 1e-5);
                assert!((a.scale[k] * 2.0 - b.scale[k]).abs() < 1e-5);
            }
            assert!((a.color[3] - b.color[3]).abs() < 1e-4);
            for k in 0..4 {
                assert!((a.rotation[k] - b.rotation[k]).abs() < 1e-5);
            }
        }
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn pbr_roundtrip() {
        let gs = sample();
        let p = tmp("pbr.ply");
        write_ply(&p, &gs, PlyFormat::Pbr, 1.0).unwrap();
        let back = load_gaussian_ply(&p).unwrap();
        assert!(back.has_pbr);
        for (a, b) in gs.iter().zip(&back.gaussians) {
            assert!((a.pbr[0] - b.pbr[0]).abs() < 1e-6);
            assert!((a.pbr[1] - b.pbr[1]).abs() < 1e-6);
            for k in 0..3 {
                assert!((a.normal[k] - b.normal[k]).abs() < 1e-6);
            }
        }
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn compressed_roundtrip() {
        let gs = sample();
        let p = tmp("cmp.ply");
        write_ply(&p, &gs, PlyFormat::CompressedPbr, 1.0).unwrap();
        let back = load_gaussian_ply(&p).unwrap();
        assert!(back.has_pbr);
        for (a, b) in gs.iter().zip(&back.gaussians) {
            for k in 0..4 {
                assert!((a.color[k] - b.color[k]).abs() < 1.0 / 255.0 + 1e-6);
            }
            let na = Vec3::from_slice(&a.normal[..3]);
            let nb = Vec3::from_slice(&b.normal[..3]);
            assert!(na.dot(nb) > 0.999, "normal {na} vs {nb}");
            assert!((a.pbr[0] - b.pbr[0]).abs() < 1.0 / 255.0 + 1e-6);
            // thin axis is min(sx, sy)
            assert!((b.scale[2] - a.scale[0].min(a.scale[1])).abs() < 1e-5);
        }
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn playcanvas_roundtrip() {
        // Enough splats to fill more than one chunk.
        let gs: Vec<GaussianVertex> = (0..600)
            .map(|i| {
                let mut g = sample()[i % 10];
                g.position[0] += i as f32 * 0.01;
                g
            })
            .collect();
        let p = tmp("pc.ply");
        write_ply(&p, &gs, PlyFormat::PlayCanvas, 1.0).unwrap();
        // ~16 bytes per splat plus the per-chunk bounds.
        let size = std::fs::metadata(&p).unwrap().len();
        assert!(size < gs.len() as u64 * 18 + 512, "{size} bytes");

        let back = load_gaussian_ply(&p).unwrap();
        assert_eq!(back.gaussians.len(), gs.len());
        assert!(!back.has_pbr);
        for (a, b) in gs.iter().zip(&back.gaussians) {
            // Position and scale are quantised inside the chunk's own range.
            for k in 0..3 {
                let span = 6.0f32; // chunk extent of this fixture, generously
                assert!((a.position[k] - b.position[k]).abs() < span / 1024.0, "{a:?} {b:?}");
                let rel = (a.scale[k].ln() - b.scale[k].ln()).abs();
                assert!(rel < 0.01, "scale {} vs {}", a.scale[k], b.scale[k]);
            }
            for k in 0..4 {
                assert!((a.color[k] - b.color[k]).abs() < 1.0 / 255.0 + 1e-6);
            }
            // Rotations may come back negated: q and -q are the same rotation.
            let (qa, qb) = (Quat::from_array(a.rotation), Quat::from_array(b.rotation));
            assert!(qa.dot(qb).abs() > 0.999, "{:?} vs {:?}", a.rotation, b.rotation);
        }
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn octa_all_octants() {
        for &x in &[-0.8f32, -0.1, 0.0, 0.3, 0.9] {
            for &y in &[-0.7f32, 0.0, 0.2, 0.6] {
                for &z in &[-0.9f32, -0.2, 0.0, 0.4, 1.0] {
                    let n = Vec3::new(x, y, z);
                    if n.length() < 1e-3 {
                        continue;
                    }
                    let n = n.normalize();
                    let d = decode_octa(encode_octa(n));
                    assert!(n.dot(d) > 0.9999, "{n} -> {d}");
                }
            }
        }
    }
}
