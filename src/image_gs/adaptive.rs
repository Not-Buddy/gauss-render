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
                    rng.gen_range(8.0..20.0),
                    rng.gen_range(8.0..20.0),
                ),
                color,
            );

            self.gaussians.push(gaussian);
        }
    }

    /// Enhanced adaptive addition with error analysis
    pub fn enhanced_adaptive_addition(&mut self, target: &RgbImage, rendered: &RgbImage) {
        let mut error_map = Vec::new();
        
        for y in (5..self.height - 5).step_by(5) {
            for x in (5..self.width - 5).step_by(5) {
                let target_pixel = target.get_pixel(x, y);
                let rendered_pixel = rendered.get_pixel(x, y);
                
                let mut pixel_error = 0.0;
                for i in 0..3 {
                    let diff = (target_pixel[i] as f32 / 255.0) - (rendered_pixel[i] as f32 / 255.0);
                    pixel_error += diff * diff;
                }
                
                if pixel_error > 0.05 {
                    error_map.push((Vector2::new(x as f32, y as f32), pixel_error.sqrt()));
                }
            }
        }
        
        error_map.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut rng = thread_rng();
        let add_count = error_map.len().min(20);
        
        for (pos, _) in error_map.iter().take(add_count) {
            let target_pixel = target.get_pixel(pos.x as u32, pos.y as u32);
            let color = vec![
                target_pixel[0] as f32 / 255.0,
                target_pixel[1] as f32 / 255.0,
                target_pixel[2] as f32 / 255.0,
            ];
            
            let gaussian = Gaussian2D::new(
                *pos + Vector2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(3.0..8.0),
                    rng.gen_range(3.0..8.0),
                ),
                color,
            );
            
            self.gaussians.push(gaussian);
        }
    }

    /// NEW: Moderate adaptive addition - much more efficient
    pub fn moderate_adaptive_addition(&mut self, target: &RgbImage, rendered: &RgbImage) {
        let mut candidates = Vec::new();
        
        // Sample error at lower resolution for speed
        for y in (5..self.height-5).step_by(8) {
            for x in (5..self.width-5).step_by(8) {
                let target_pixel = target.get_pixel(x, y);
                let rendered_pixel = rendered.get_pixel(x, y);
                
                let mut error = 0.0;
                for i in 0..3 {
                    let diff = (target_pixel[i] as f32 / 255.0) - (rendered_pixel[i] as f32 / 255.0);
                    error += diff * diff;
                }
                
                if error > 0.08 { // Reasonable threshold
                    candidates.push((Vector2::new(x as f32, y as f32), error));
                }
            }
        }
        
        // Sort and add reasonable number
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut rng = thread_rng();
        let add_count = candidates.len().min(10); // MAX 10 new Gaussians per iteration
        
        for (pos, _) in candidates.iter().take(add_count) {
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
                    rng.gen_range(4.0..12.0), // Reasonable detail size
                    rng.gen_range(4.0..12.0),
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
