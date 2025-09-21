// src/image_gs/initialization.rs

use super::ImageGS;
use crate::gaussian::Gaussian2D;
use image::RgbImage;
use nalgebra::Vector2;
use rand::prelude::*;

impl ImageGS {
    /// Hierarchical initialization with multiple levels
    pub fn hierarchical_initialize(&mut self, target: &RgbImage) {
        let mut rng = thread_rng();
        self.gaussians.clear();

        // Level 1: Large Gaussians for structure (background)
        for _ in 0..20 {
            let x = rng.gen_range(0..self.width);
            let y = rng.gen_range(0..self.height);

            let pixel = target.get_pixel(x, y);
            let color = vec![
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ];

            let gaussian = Gaussian2D::new(
                Vector2::new(x as f32, y as f32),
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(30.0..60.0), // Large for structure
                    rng.gen_range(30.0..60.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }

        // Level 2: Medium Gaussians for mid-frequency details
        for _ in 0..40 {
            let x = rng.gen_range(0..self.width);
            let y = rng.gen_range(0..self.height);

            let pixel = target.get_pixel(x, y);
            let color = vec![
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ];

            let gaussian = Gaussian2D::new(
                Vector2::new(x as f32, y as f32),
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(12.0..25.0), // Medium for details
                    rng.gen_range(12.0..25.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }

        // Level 3: Small Gaussians for high-frequency details
        for _ in 0..60 {
            let x = rng.gen_range(0..self.width);
            let y = rng.gen_range(0..self.height);

            let pixel = target.get_pixel(x, y);
            let color = vec![
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ];

            let gaussian = Gaussian2D::new(
                Vector2::new(x as f32, y as f32),
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(4.0..12.0), // Small for fine details
                    rng.gen_range(4.0..12.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }

        println!("Initialized {} Gaussians hierarchically", self.gaussians.len());
    }

    /// Initialize from target image with more Gaussians
    pub fn initialize_from_image(&mut self, target: &RgbImage) {
        let mut rng = thread_rng();
        self.gaussians.clear();

        // Start with 50 Gaussians instead of 10
        for _ in 0..50 {
            let x = rng.gen_range(0..self.width);
            let y = rng.gen_range(0..self.height);

            let pixel = target.get_pixel(x, y);
            let color = vec![
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ];

            let gaussian = Gaussian2D::new(
                Vector2::new(x as f32, y as f32),
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(8.0..25.0), // Smaller, more precise Gaussians
                    rng.gen_range(8.0..25.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }
    }

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
