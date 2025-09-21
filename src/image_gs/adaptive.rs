// src/image_gs/adaptive.rs

use super::ImageGS;
use crate::gaussian::Gaussian2D;
use image::RgbImage;
use nalgebra::Vector2;
use rand::prelude::*;

impl ImageGS {
    /// Basic adaptive Gaussian addition
    pub fn adaptive_gaussian_addition(&mut self, target: &RgbImage, rendered: &RgbImage) {
        let error_threshold = 0.1;
        let mut high_error_positions = Vec::new();

        for y in (0..self.height).step_by(10) {
            for x in (0..self.width).step_by(10) {
                let target_pixel = target.get_pixel(x, y);
                let rendered_pixel = rendered.get_pixel(x, y);

                let mut error = 0.0;
                for i in 0..3 {
                    let diff = (target_pixel[i] as f32 / 255.0) - (rendered_pixel[i] as f32 / 255.0);
                    error += diff.abs();
                }

                if error > error_threshold {
                    high_error_positions.push(Vector2::new(x as f32, y as f32));
                }
            }
        }

        let mut rng = thread_rng();
        for pos in high_error_positions.iter().take(5) {
            let target_pixel = target.get_pixel(pos.x as u32, pos.y as u32);
            let color = vec![
                target_pixel[0] as f32 / 255.0,
                target_pixel[1] as f32 / 255.0,
                target_pixel[2] as f32 / 255.0,
            ];

            let gaussian = Gaussian2D::new(
                *pos,
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(5.0..15.0),
                    rng.gen_range(5.0..15.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }
    }

    /// Enhanced adaptive addition with gradient-based criterion
    pub fn enhanced_adaptive_addition(&mut self, target: &RgbImage, rendered: &RgbImage) {
        let mut error_map = vec![vec![0.0f32; self.width as usize]; self.height as usize];

        // Compute detailed error map
        for y in 0..self.height {
            for x in 0..self.width {
                let target_pixel = target.get_pixel(x, y);
                let rendered_pixel = rendered.get_pixel(x, y);

                let mut pixel_error = 0.0;
                for i in 0..3 {
                    let diff = (target_pixel[i] as f32 / 255.0) - (rendered_pixel[i] as f32 / 255.0);
                    pixel_error += diff * diff; // L2 error instead of L1
                }

                error_map[y as usize][x as usize] = pixel_error.sqrt();
            }
        }

        // Find high-error regions with gradient-based criterion
        let mut candidates = Vec::new();
        let error_threshold = 0.08; // Lower threshold for more precision

        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let error = error_map[y as usize][x as usize];

                // Also consider gradient of error
                let grad_x = error_map[y as usize][(x + 1) as usize] - error_map[y as usize][(x - 1) as usize];
                let grad_y = error_map[(y + 1) as usize][x as usize] - error_map[(y - 1) as usize][x as usize];
                let gradient_magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();

                if error > error_threshold || gradient_magnitude > 0.05 {
                    candidates.push((Vector2::new(x as f32, y as f32), error + gradient_magnitude));
                }
            }
        }

        // Sort by error and take top candidates
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut rng = thread_rng();
        for (pos, _) in candidates.iter().take(15) { // Add more Gaussians per iteration
            let target_pixel = target.get_pixel(pos.x as u32, pos.y as u32);
            let color = vec![
                target_pixel[0] as f32 / 255.0,
                target_pixel[1] as f32 / 255.0,
                target_pixel[2] as f32 / 255.0,
            ];

            // Smaller Gaussians for detail
            let gaussian = Gaussian2D::new(
                *pos,
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(3.0..8.0), // Much smaller for fine details
                    rng.gen_range(3.0..8.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }
    }

    /// Prune ineffective Gaussians to prevent overgrowth
    pub fn prune_ineffective_gaussians(&mut self) {
        let initial_count = self.gaussians.len();
        let min_opacity_threshold = 0.005;
        let max_scale_threshold = 100.0;

        self.gaussians.retain(|gaussian| {
            let max_scale = gaussian.scale.x.max(gaussian.scale.y);
            let opacity = gaussian.color.iter().sum::<f32>() / (gaussian.color.len() as f32);

            opacity > min_opacity_threshold && max_scale < max_scale_threshold
        });

        let pruned_count = initial_count - self.gaussians.len();
        if pruned_count > 0 {
            println!("Pruned {} ineffective Gaussians", pruned_count);
        }
    }
}
