//! Fly camera, port of `utils/Camera.cpp` (WASD + QE, R/T roll, RMB look, wheel zoom).

use glam::{Mat4, Quat, Vec3};

use crate::types::BBox;

pub const FAST_SPEED: f32 = 3.5;
pub const SLOW_SPEED: f32 = 0.25;
pub const DEFAULT_SPEED: f32 = 1.0;

pub const NEAR_PLANE: f32 = 0.01;
pub const FAR_PLANE: f32 = 100.0;

#[derive(Clone, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub world_up: Vec3,
    /// Degrees.
    pub yaw: f32,
    /// Degrees.
    pub pitch: f32,
    /// Degrees.
    pub roll: f32,
    /// Vertical field of view, degrees.
    pub fov: f32,
    pub movement_speed: f32,
    pub mouse_sensitivity: f32,
    front: Vec3,
    right: Vec3,
    up: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 0.0, 5.0), Vec3::Y, -90.0, 0.0)
    }
}

/// Keyboard state for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraKeys {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub roll_left: bool,
    pub roll_right: bool,
    pub boost: bool,
    pub slow: bool,
}

impl Camera {
    pub fn new(position: Vec3, up: Vec3, yaw: f32, pitch: f32) -> Self {
        let mut c = Self {
            position,
            world_up: up,
            yaw,
            pitch,
            roll: 0.0,
            fov: 45.0,
            movement_speed: 2.0,
            mouse_sensitivity: 0.1,
            front: -Vec3::Z,
            right: Vec3::X,
            up: Vec3::Y,
        };
        c.update_vectors();
        c
    }

    pub fn front(&self) -> Vec3 {
        self.front
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.front, self.up)
    }

    /// Right-handed perspective with a [0, 1] depth range (wgpu convention).
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(
            self.fov.to_radians(),
            aspect.max(1e-6),
            NEAR_PLANE,
            FAR_PLANE,
        )
    }

    pub fn process_keyboard(&mut self, dt: f32, k: CameraKeys) {
        let mut mult = if k.boost { FAST_SPEED } else { DEFAULT_SPEED };
        if k.slow {
            mult = SLOW_SPEED;
        }
        let v = self.movement_speed * dt * mult;
        if k.forward {
            self.position += self.front * v;
        }
        if k.backward {
            self.position -= self.front * v;
        }
        if k.left {
            self.position -= self.right * v;
        }
        if k.right {
            self.position += self.right * v;
        }
        if k.up {
            self.position += self.world_up * v;
        }
        if k.down {
            self.position -= self.world_up * v;
        }
        if k.roll_right {
            self.roll -= 0.5;
        }
        if k.roll_left {
            self.roll += 0.5;
        }
        self.update_vectors();
    }

    pub fn process_mouse_movement(&mut self, dx: f32, dy: f32, constrain_pitch: bool) {
        self.yaw += dx * self.mouse_sensitivity;
        self.pitch += dy * self.mouse_sensitivity;
        if constrain_pitch {
            self.pitch = self.pitch.clamp(-89.0, 89.0);
        }
        self.update_vectors();
    }

    pub fn process_mouse_scroll(&mut self, dy: f32) {
        self.fov = (self.fov - dy).clamp(1.0, 90.0);
    }

    /// Place the camera so that `bbox` fills the view, looking down -Z
    /// (not in the original; handy for models that are not unit sized).
    pub fn frame_bbox(&mut self, bbox: &BBox) {
        if !bbox.is_valid() {
            return;
        }
        let radius = (bbox.size().length() * 0.5).max(1e-3);
        let dist = radius / (self.fov.to_radians() * 0.5).tan() * 1.1;
        self.yaw = -90.0;
        self.pitch = 0.0;
        self.roll = 0.0;
        self.position = bbox.center() + Vec3::Z * dist;
        self.movement_speed = (radius * 1.0).max(0.1);
        self.update_vectors();
    }

    fn update_vectors(&mut self) {
        let (yaw, pitch) = (self.yaw.to_radians(), self.pitch.to_radians());
        self.front = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
        .normalize();
        let rolled_up = Quat::from_axis_angle(self.front, self.roll.to_radians()) * Vec3::Y;
        self.right = self.front.cross(rolled_up).normalize();
        self.up = self.right.cross(self.front).normalize();
    }
}
