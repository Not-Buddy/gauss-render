// src/image_gs/initialization.rs

use super::ImageGS;
use crate::gaussian::Gaussian2D;
use image::RgbImage;
use nalgebra::Vector2;
use rand::prelude::*;

impl ImageGS {

pub fn dense_content_aware_initialize(&mut self, target: &RgbImage) {
    let mut rng = thread_rng();
    self.gaussians.clear();
    
    // MUCH MORE GAUSSIANS for quality
    let base_count = 500; // Increased from ~120
    
    // 1. Dense grid initialization for coverage
    let grid_size = ((base_count as f32 * 0.6) as usize).min(300);
    let cols = (grid_size as f32).sqrt() as usize;
    let rows = (grid_size + cols - 1) / cols;
    
    for row in 0..rows {
        for col in 0..cols {
            if self.gaussians.len() >= grid_size { break; }
            
            let x = ((col as f32 + 0.5) * self.width as f32 / cols as f32) as u32;
            let y = ((row as f32 + 0.5) * self.height as f32 / rows as f32) as u32;
            
            let x = x.clamp(0, self.width - 1);
            let y = y.clamp(0, self.height - 1);
            
            let pixel = target.get_pixel(x, y);
            let color = vec![
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ];
            
            // Analyze local variance for better scale initialization
            let mut local_variance = 0.0;
            let mut sample_count = 0;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let nx = (x as i32 + dx).clamp(0, self.width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, self.height as i32 - 1) as u32;
                    let npixel = target.get_pixel(nx, ny);
                    
                    let diff = ((npixel[0] as f32 - pixel[0] as f32).abs() + 
                              (npixel[1] as f32 - pixel[1] as f32).abs() + 
                              (npixel[2] as f32 - pixel[2] as f32).abs()) / (3.0 * 255.0);
                    local_variance += diff;
                    sample_count += 1;
                }
            }
            local_variance /= sample_count as f32;
            
            // Adaptive scale based on content
            let base_scale = if local_variance > 0.1 { 
                rng.gen_range(3.0..8.0) // Small for high detail areas
            } else { 
                rng.gen_range(8.0..15.0) // Larger for smooth areas
            };
            
            let gaussian = Gaussian2D::new(
                Vector2::new(x as f32, y as f32),
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    base_scale * rng.gen_range(0.8..1.2),
                    base_scale * rng.gen_range(0.8..1.2),
                ),
                color,
            );
            
            self.gaussians.push(gaussian);
        }
    }
    
    // 2. Edge-aware random initialization
    let edge_count = base_count - self.gaussians.len();
    for _ in 0..edge_count {
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
                rng.gen_range(2.0..12.0), // Smaller for detail
                rng.gen_range(2.0..12.0),
            ),
            color,
        );
        
        self.gaussians.push(gaussian);
    }
    
    println!("Dense initialized {} Gaussians", self.gaussians.len());
}
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
