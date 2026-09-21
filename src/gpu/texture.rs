//! Texture upload with CPU-generated mip chain.

use image::{imageops::FilterType, RgbaImage};

use super::GpuContext;
use crate::scene::TextureData;

/// Maximum mip levels (GL_TEXTURE_MAX_LEVEL = 4 in the original => 5 levels).
const MAX_MIPS: u32 = 5;

pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl GpuTexture {
    /// Upload an RGBA8 image (linear / UNORM: the original uploads GL_RGB(A),
    /// so colors stay in gamma space all the way to the PLY file).
    pub fn from_data(ctx: &GpuContext, data: &TextureData, label: &str) -> Self {
        let (w, h) = (data.width.max(1), data.height.max(1));
        let levels = (32 - w.max(h).leading_zeros()).min(MAX_MIPS);
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut img = RgbaImage::from_raw(w, h, data.rgba.clone()).expect("texture size mismatch");
        for level in 0..levels {
            if level > 0 {
                let (nw, nh) = ((img.width() / 2).max(1), (img.height() / 2).max(1));
                img = image::imageops::resize(&img, nw, nh, FilterType::Triangle);
            }
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                img.as_raw(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * img.width()),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: img.width(),
                    height: img.height(),
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }

    /// 1x1 white texture bound when a material slot is empty.
    pub fn white(ctx: &GpuContext) -> Self {
        let data = TextureData {
            width: 1,
            height: 1,
            rgba: vec![255; 4],
            id: usize::MAX,
        };
        Self::from_data(ctx, &data, "white")
    }
}

/// Repeat + trilinear, as in the original (`GL_REPEAT`, `GL_LINEAR_MIPMAP_LINEAR`).
pub fn material_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("material sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}
