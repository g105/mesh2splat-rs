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
        PlyFormat::Standard => write_standard(&mut w, gaussians, scale_multiplier)?,
        PlyFormat::Pbr => write_pbr(&mut w, gaussians, scale_multiplier)?,
        PlyFormat::CompressedPbr => write_compressed(&mut w, gaussians, scale_multiplier)?,
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

fn write_standard(w: &mut impl Write, gs: &[GaussianVertex], m: f32) -> Result<()> {
    let mut props: Vec<(String, String)> = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .iter()
    .map(|n| ("float".to_string(), n.to_string()))
    .collect();
    for i in 0..45 {
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
    let zeros = [0u8; 45 * 4];
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
    for el in &elements {
        let is_vertex = el.name == "vertex";
        let mut columns: Vec<Vec<f32>> = if is_vertex {
            vec![Vec::with_capacity(el.count); el.props.len()]
        } else {
            vec![]
        };
        match encoding {
            Encoding::Ascii => {
                for _ in 0..el.count {
                    line.clear();
                    r.read_line(&mut line)?;
                    if is_vertex {
                        for (i, tok) in line.split_whitespace().take(el.props.len()).enumerate() {
                            columns[i].push(tok.parse::<f32>().context("bad ascii PLY value")?);
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
                                columns[i].push(ty.read(&row[offsets[i]..], le) as f32);
                            }
                        }
                    }
                    remaining -= rows;
                }
            }
        }
        if is_vertex {
            result = Some(PlyVertexData {
                count: el.count,
                names: el.props.iter().map(|p| p.0.clone()).collect(),
                columns,
            });
            break;
        }
    }
    result.ok_or_else(|| anyhow!("PLY file has no 'vertex' element"))
}

/// Result of loading a 3DGS PLY file.
pub struct LoadedPly {
    pub gaussians: Vec<GaussianVertex>,
    /// True when the file carried normals + metallic/roughness, so it can be relit.
    pub has_pbr: bool,
}

/// Load a 3DGS `.ply` (standard, PBR, or compressed-PBR layout).
/// Scales are exponentiated and opacities passed through a sigmoid, so the
/// result uses [`SourceFormat::Ply`] semantics.
pub fn load_gaussian_ply(path: impl AsRef<Path>) -> Result<LoadedPly> {
    let d = read_ply_vertices(path)?;
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
