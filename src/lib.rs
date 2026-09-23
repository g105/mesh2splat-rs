//! Mesh2Splat: fast mesh to 3D Gaussian Splatting conversion.
//!
//! Rust / wgpu port of Electronic Arts SEED's
//! [Mesh2Splat](https://github.com/electronicarts/mesh2splat).

#[cfg(feature = "gui")]
pub mod app;
pub mod camera;
pub mod cli;
pub mod gpu;
pub mod hair;
pub mod merge;
pub mod ply;
pub mod scene;
pub mod types;

pub use types::*;
