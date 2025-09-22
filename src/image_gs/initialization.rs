// src/image_gs/initialization.rs

use super::ImageGS;
use crate::gaussian::Gaussian2D;
use nalgebra::Vector2;
use rand::prelude::*;

impl ImageGS {

    /// Initialize with random Gaussians
    pub fn initialize_random(&mut self, num_gaussians: usize) {
        let mut rng = thread_rng();
        self.gaussians.clear();

        for _ in 0..num_gaussians {
            let mean = Vector2::new(
                rng.gen_range(0.0..self.width as f32),
                rng.gen_range(0.0..self.height as f32),
            );
            let color = vec![
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ];

            let gaussian = Gaussian2D::new(
                mean,
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(15.0..40.0),
                    rng.gen_range(15.0..40.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }
    }
}
