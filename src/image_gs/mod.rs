// src/image_gs/mod.rs

use crate::gaussian::Gaussian2D;
pub mod initialization;

/// Main ImageGS struct for 2D Gaussian Splatting
pub struct ImageGS {
    pub gaussians: Vec<Gaussian2D>,
    pub width: u32,
    pub height: u32,
}

impl ImageGS {
    /// Create a new ImageGS instance
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            gaussians: Vec::new(),
            width,
            height,
        }
    }

    /// Get the number of Gaussians
    pub fn gaussian_count(&self) -> usize {
        self.gaussians.len()
    }
}

// Include all the existing modules
pub mod implementation;
pub mod gpu_render;

pub use gpu_render::*;