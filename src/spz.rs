//! Niantic `.spz` reading and writing
//! (<https://github.com/nianticlabs/spz>, `load-spz.cc`).
//!
//! An `.spz` file is a gzip stream holding a 16-byte header followed by one
//! array per attribute, each covering every splat:
//!
//! | field     | bytes / splat | encoding                                        |
//! |-----------|---------------|-------------------------------------------------|
//! | position  | 9             | 24-bit signed fixed point, `fractional_bits`    |
//! | alpha     | 1             | opacity (after the sigmoid) x 255               |
//! | color     | 3             | SH DC x (0.15 x 255) + 127.5                    |
//! | scale     | 3             | (log scale + 10) x 16                           |
//! | rotation  | 4 (v3) / 3 (v2) | smallest-three (v3) / xyz with w >= 0 (v2)    |
//! | SH rest   | 0             | converted splats have no view-dependent colour  |
//!
//! 20 bytes per splat before gzip, which does well on it because the arrays
//! are grouped by attribute. Like the PlayCanvas layout, it drops normals and
//! PBR, so it is an export format for viewers.
//!
//! We write version 3, the last one on gzip (version 4 switched to zstd and a
//! new container); readers of version 4 still read 3.
//!
//! **Axes.** SPZ stores splats in RUB (x right, y up, z back), and tools that
//! convert a 3DGS `.ply` to `.spz` treat the PLY as RDF (y down, z forward)
//! and flip y and z. We do the same, so an exported `.spz` shows the same way
//! up as the `.ply` of the same splats in any viewer, and reading it back
//! undoes the flip.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use glam::Vec3;

use crate::types::*;

/// `"NGSP"`, little-endian.
const MAGIC: u32 = 0x5053_474e;
const VERSION: u32 = 3;
/// SH DC coefficients are stored as `sh * COLOR_SCALE * 255 + 127.5`.
const COLOR_SCALE: f32 = 0.15;
/// The reference writer always uses 12 (~0.25 mm at metre scale).
const FRACTIONAL_BITS: u8 = 12;
/// Largest magnitude a 24-bit signed fixed-point value holds.
const MAX_FIXED: f32 = ((1 << 23) - 1) as f32;
const FLAG_ANTIALIASED: u8 = 0x1;

/// Flip of y and z between our (PLY, RDF) frame and SPZ's RUB frame. It is a
/// 180 degree turn about x, so it is its own inverse and a proper rotation:
/// positions flip y and z, and so do the quaternion's y and z.
const FLIP: [f32; 3] = [1.0, -1.0, -1.0];

fn to_u8(x: f32) -> u8 {
    // NaN (e.g. the log of a zero scale gone wrong) lands on 0 like -inf.
    x.round().clamp(0.0, 255.0) as u8
}

/// Fractional bits for these splats: 12 like the reference, fewer when a
/// coordinate would overflow 24 bits at 12 (beyond +-2048 units). Readers take
/// the value from the header.
fn fractional_bits(gs: &[GaussianVertex]) -> Result<u8> {
    let max = gs
        .iter()
        .flat_map(|g| g.position[..3].iter())
        .fold(0f32, |m, v| m.max(v.abs()));
    if !max.is_finite() {
        bail!("cannot write SPZ: non-finite splat position");
    }
    let mut bits = FRACTIONAL_BITS;
    while bits > 0 && (max * (1u32 << bits) as f32).round() > MAX_FIXED {
        bits -= 1;
    }
    if (max * (1u32 << bits) as f32).round() > MAX_FIXED {
        bail!("cannot write SPZ: coordinate {max} does not fit 24 bits");
    }
    Ok(bits)
}

/// Smallest-three quaternion encoding of SPZ v3: 2 bits for the index of the
/// largest component, then for each other component (in xyzw order) a sign
/// bit and a 9-bit magnitude scaled by `1 / sqrt(1/2)`. `q` is xyzw.
fn pack_rotation(q: [f32; 4]) -> u32 {
    let len = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    let q = if len > 0.0 { q.map(|v| v / len) } else { [0.0, 0.0, 0.0, 1.0] };
    let largest = (0..4).fold(0, |l, i| if q[i].abs() > q[l].abs() { i } else { l });
    // q and -q are the same rotation: make the dropped component positive.
    let negate = q[largest] < 0.0;
    let mut comp = largest as u32;
    for (i, v) in q.iter().enumerate() {
        if i != largest {
            let neg = ((*v < 0.0) ^ negate) as u32;
            let mag = (511.0 * (v.abs() / std::f32::consts::FRAC_1_SQRT_2) + 0.5) as u32;
            comp = (comp << 10) | (neg << 9) | mag.min(511);
        }
    }
    comp
}

/// Inverse of [`pack_rotation`], returning xyzw.
fn unpack_rotation(comp: u32) -> [f32; 4] {
    let largest = (comp >> 30) as usize;
    let mut q = [0f32; 4];
    let mut rest = comp;
    let mut sum = 0.0;
    for i in (0..4).rev() {
        if i != largest {
            let mag = (rest & 511) as f32 / 511.0 * std::f32::consts::FRAC_1_SQRT_2;
            q[i] = if (rest >> 9) & 1 == 1 { -mag } else { mag };
            sum += q[i] * q[i];
            rest >>= 10;
        }
    }
    q[largest] = (1.0 - sum).max(0.0).sqrt();
    q
}

/// Write gaussians as an `.spz` (version 3). `scale_multiplier` as in
/// [`crate::ply::write_ply`].
pub fn write_spz(w: impl Write, gs: &[GaussianVertex], m: f32) -> Result<()> {
    let bits = fractional_bits(gs)?;
    let n = u32::try_from(gs.len()).context("too many splats for SPZ")?;
    let mut z = GzEncoder::new(w, Compression::default());

    z.write_all(&MAGIC.to_le_bytes())?;
    z.write_all(&VERSION.to_le_bytes())?;
    z.write_all(&n.to_le_bytes())?;
    // SH degree, fractional bits, flags (not trained with antialiasing), reserved.
    z.write_all(&[0, bits, 0, 0])?;

    let fixed = (1u32 << bits) as f32;
    for g in gs {
        for (p, s) in g.position.iter().zip(FLIP) {
            let v = (p * s * fixed).round() as i32;
            z.write_all(&v.to_le_bytes()[..3])?;
        }
    }
    let bytes: Vec<u8> = gs.iter().map(|g| to_u8(g.color[3] * 255.0)).collect();
    z.write_all(&bytes)?;

    let mut bytes = Vec::with_capacity(gs.len() * 4);
    for g in gs {
        let sh = sh_from_color(Vec3::from_slice(&g.color[..3]));
        bytes.extend(sh.to_array().map(|v| to_u8(v * (COLOR_SCALE * 255.0) + 127.5)));
    }
    z.write_all(&bytes)?;

    bytes.clear();
    for g in gs {
        bytes.extend((0..3).map(|k| to_u8(((g.scale[k] * m).ln() + 10.0) * 16.0)));
    }
    z.write_all(&bytes)?;

    bytes.clear();
    for g in gs {
        let [w, x, y, zz] = g.rotation;
        let q = [x * FLIP[0], y * FLIP[1], zz * FLIP[2], w];
        bytes.extend(pack_rotation(q).to_le_bytes());
    }
    z.write_all(&bytes)?;

    z.finish()?.flush()?;
    Ok(())
}

/// Read an `.spz` (versions 2 and 3). Only the SH DC colour is kept.
pub fn load_spz(path: impl AsRef<Path>) -> Result<Vec<GaussianVertex>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut raw = Vec::new();
    GzDecoder::new(BufReader::new(file))
        .read_to_end(&mut raw)
        .with_context(|| format!("{}: not a gzip-compressed SPZ", path.display()))?;
    read_spz(&raw).with_context(|| format!("cannot read {}", path.display()))
}

/// Decode an uncompressed SPZ payload (header + attribute arrays).
fn read_spz(raw: &[u8]) -> Result<Vec<GaussianVertex>> {
    if raw.len() < 16 {
        bail!("SPZ header truncated");
    }
    let word = |i: usize| u32::from_le_bytes(raw[i..i + 4].try_into().unwrap());
    if word(0) != MAGIC {
        bail!("not an SPZ file (bad magic)");
    }
    let version = word(4);
    if !(2..=3).contains(&version) {
        bail!("SPZ version {version} is not supported (2 and 3 are)");
    }
    let n = word(8) as usize;
    let (sh_degree, bits, flags) = (raw[12], raw[13], raw[14]);
    if sh_degree > 4 {
        bail!("invalid SPZ SH degree {sh_degree}");
    }
    if flags & FLAG_ANTIALIASED != 0 {
        log::info!("SPZ was trained with antialiasing; rendering without it");
    }
    let rot_bytes = if version >= 3 { 4 } else { 3 };
    let sh_bytes = ((sh_degree as usize + 1).pow(2) - 1) * 3;
    let need = n
        .checked_mul(9 + 1 + 3 + 3 + rot_bytes + sh_bytes)
        .context("SPZ splat count overflows")?;
    if raw.len() - 16 < need {
        bail!("SPZ data truncated: {} splats need {need} bytes, found {}", n, raw.len() - 16);
    }

    let mut at = 16;
    let mut take = |len: usize| {
        let s = &raw[at..at + len];
        at += len;
        s
    };
    let (pos, alpha, color, scale, rot) =
        (take(n * 9), take(n), take(n * 3), take(n * 3), take(n * rot_bytes));

    let fixed = 1.0 / (1u32 << bits) as f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut p = [0f32; 4];
        for k in 0..3 {
            let b = &pos[(i * 3 + k) * 3..][..3];
            // Sign-extend the 24-bit value.
            let v = i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8;
            p[k] = v as f32 * fixed * FLIP[k];
        }
        p[3] = 1.0;
        let sh = Vec3::from_array(
            [0, 1, 2].map(|k| (color[i * 3 + k] as f32 / 255.0 - 0.5) / COLOR_SCALE),
        );
        let c = color_from_sh(sh);
        let s = [0, 1, 2].map(|k| (scale[i * 3 + k] as f32 / 16.0 - 10.0).exp());
        let [x, y, z, w] = if version >= 3 {
            let b = &rot[i * 4..][..4];
            unpack_rotation(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        } else {
            let b = &rot[i * 3..][..3];
            let xyz = Vec3::from_array([0, 1, 2].map(|k| b[k] as f32 / 127.5 - 1.0));
            let w = (1.0 - xyz.length_squared()).max(0.0).sqrt();
            [xyz.x, xyz.y, xyz.z, w]
        };
        out.push(GaussianVertex {
            position: p,
            color: [c.x, c.y, c.z, alpha[i] as f32 / 255.0],
            scale: [s[0], s[1], s[2], 1.0],
            normal: [0.0; 4],
            rotation: [w, x * FLIP[0], y * FLIP[1], z * FLIP[2]],
            pbr: [DEFAULT_METALLIC, DEFAULT_ROUGHNESS, 0.0, 1.0],
        });
    }
    Ok(out)
}

/// True when `path` starts like a gzip stream, which every SPZ up to version 3 is.
pub fn is_spz(path: impl AsRef<Path>) -> Result<bool> {
    let path = path.as_ref();
    let mut head = [0u8; 4];
    let mut f = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let got = f.read(&mut head)?;
    if got == 4 && u32::from_le_bytes(head) == MAGIC {
        bail!(
            "{}: SPZ version 4 (zstd) is not supported yet; re-save it as version 3",
            path.display()
        );
    }
    Ok(got >= 2 && head[..2] == [0x1f, 0x8b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn sample() -> Vec<GaussianVertex> {
        (0..300)
            .map(|i| {
                let t = i as f32 / 300.0;
                let q = Quat::from_rotation_y(t * 7.0) * Quat::from_rotation_x(t * 3.0 - 1.0);
                GaussianVertex {
                    position: [t * 4.0 - 2.0, 3.0 * t, -t, 1.0],
                    color: [t, 1.0 - t, 0.5, 0.05 + 0.9 * t],
                    scale: [0.01 + t, 0.02 + t * 0.5, 0.005, 0.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    rotation: [q.w, q.x, q.y, q.z],
                    pbr: [t, 1.0 - t, 0.0, 1.0],
                }
            })
            .collect()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("m2s_spz_{}_{name}", std::process::id()))
    }

    #[test]
    fn roundtrip() {
        let gs = sample();
        let p = tmp("rt.spz");
        write_spz(File::create(&p).unwrap(), &gs, 2.0).unwrap();
        assert!(is_spz(&p).unwrap());
        let back = load_spz(&p).unwrap();
        assert_eq!(back.len(), gs.len());
        for (a, b) in gs.iter().zip(&back) {
            for k in 0..3 {
                assert!((a.position[k] - b.position[k]).abs() <= 0.5 / 4096.0 + 1e-6);
                // Colours go through the SH DC range at 0.15 x 255 steps.
                assert!((a.color[k] - b.color[k]).abs() < 0.01, "{a:?} {b:?}");
                // Log scale in steps of 1/16.
                let d = ((a.scale[k] * 2.0).ln() - b.scale[k].ln()).abs();
                assert!(d <= 1.0 / 32.0 + 1e-4, "scale {} vs {}", a.scale[k] * 2.0, b.scale[k]);
            }
            assert!((a.color[3] - b.color[3]).abs() <= 0.5 / 255.0 + 1e-6);
            let (qa, qb) = (Quat::from_array(a.rotation), Quat::from_array(b.rotation));
            assert!(qa.dot(qb).abs() > 0.9995, "{:?} vs {:?}", a.rotation, b.rotation);
        }
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn header_and_axes() {
        let mut g = sample()[0];
        g.position = [1.0, 2.0, 3.0, 1.0];
        let mut buf = Vec::new();
        write_spz(&mut buf, &[g], 1.0).unwrap();
        let mut raw = Vec::new();
        GzDecoder::new(&buf[..]).read_to_end(&mut raw).unwrap();
        assert_eq!(&raw[0..4], b"NGSP");
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), 1);
        assert_eq!(raw[12..16], [0, 12, 0, 0]);
        // y and z are flipped into SPZ's RUB frame.
        let fixed = |b: &[u8]| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 4096.0;
        assert_eq!(fixed(&raw[16..19]), 1.0);
        assert_eq!(fixed(&raw[19..22]), -2.0);
        assert_eq!(fixed(&raw[22..25]), -3.0);
        // 9 + 1 + 3 + 3 + 4 bytes per splat.
        assert_eq!(raw.len(), 16 + 20);
    }

    #[test]
    fn large_coordinates_lower_precision() {
        let mut gs = sample();
        gs[0].position[0] = 5000.0;
        assert_eq!(fractional_bits(&gs).unwrap(), 10);
        let p = tmp("big.spz");
        write_spz(File::create(&p).unwrap(), &gs, 1.0).unwrap();
        let back = load_spz(&p).unwrap();
        assert!((back[0].position[0] - 5000.0).abs() < 1e-3);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn rotation_packing() {
        for q in [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5, -0.5],
            [0.1, -0.7, 0.2, 0.6],
        ] {
            let qa = Quat::from_array(q).normalize();
            let qb = Quat::from_array(unpack_rotation(pack_rotation(q)));
            assert!(qa.dot(qb).abs() > 0.9999, "{qa} vs {qb}");
        }
    }
}
