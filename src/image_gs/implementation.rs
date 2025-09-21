// src/image_gs/implementation.rs

use super::ImageGS;
use crate::gaussian::Gaussian2D;
use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::Vector2;
use rand::prelude::*;
use rayon::prelude::*;

impl ImageGS {
    
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

    pub fn smart_initialize(&mut self, target: &RgbImage) {
        let mut rng = thread_rng();
        self.gaussians.clear();
        
        // Start with REASONABLE number: 150 Gaussians
        let target_count = 150;
        
        // Grid-based initialization for good coverage
        let grid_size = (target_count as f32).sqrt() as usize;
        
        for i in 0..grid_size {
            for j in 0..grid_size {
                if self.gaussians.len() >= target_count { break; }
                
                let x = ((i as f32 + 0.5 + rng.gen_range(-0.3..0.3)) * self.width as f32 / grid_size as f32) as u32;
                let y = ((j as f32 + 0.5 + rng.gen_range(-0.3..0.3)) * self.height as f32 / grid_size as f32) as u32;
                
                let x = x.clamp(0, self.width - 1);
                let y = y.clamp(0, self.height - 1);
                
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
                        rng.gen_range(8.0..20.0), // Reasonable size
                        rng.gen_range(8.0..20.0),
                    ),
                    color,
                );
                
                self.gaussians.push(gaussian);
            }
        }
        
        println!("Smart initialized {} Gaussians", self.gaussians.len());
    }

    /// Keep existing hierarchical initialization but make it lighter
    pub fn hierarchical_initialize(&mut self, target: &RgbImage) {
        let mut rng = thread_rng();
        self.gaussians.clear();

        // REDUCED from 20+40+60 = 120 to 15+25+30 = 70 total
        for _ in 0..15 {
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
                    rng.gen_range(30.0..50.0), // Reduced from 60
                    rng.gen_range(30.0..50.0),
                ),
                color,
            );
            self.gaussians.push(gaussian);
        }

        for _ in 0..25 {
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
                    rng.gen_range(12.0..20.0), // Reduced from 25
                    rng.gen_range(12.0..20.0),
                ),
                color,
            );
            self.gaussians.push(gaussian);
        }

        for _ in 0..30 {
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
                    rng.gen_range(4.0..12.0),
                    rng.gen_range(4.0..12.0),
                ),
                color,
            );
            self.gaussians.push(gaussian);
        }

        println!("Initialized {} Gaussians hierarchically", self.gaussians.len());
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
                    rng.gen_range(8.0..20.0),
                    rng.gen_range(8.0..20.0),
                ),
                color,
            );
            self.gaussians.push(gaussian);
        }
    }


pub fn optimize_step_with_lr(&mut self, target: &RgbImage, rendered: &RgbImage, lr: f32) {
    let mut rng = thread_rng();
    
    // Optimize a reasonable subset - NOT all Gaussians!
    let gaussians_to_optimize = (self.gaussians.len() / 6).min(25).max(8);
    
    let mut gaussian_indices: Vec<usize> = (0..self.gaussians.len()).collect();
    gaussian_indices.shuffle(&mut rng);
    
    // Use REGULAR iteration instead of parallel to avoid thread-safety issues
    for &idx in gaussian_indices.iter().take(gaussians_to_optimize) {
        let gaussian = &mut self.gaussians[idx];
        
        let original_params = (gaussian.mean, gaussian.scale, gaussian.rotation, gaussian.color.clone());
        let mut best_loss = f32::INFINITY;
        let mut best_params = original_params.clone();
        
        for trial in 0..3 {
            // Reset to original
            *gaussian = Gaussian2D::new(original_params.0, original_params.2, original_params.1, original_params.3.clone());
            
            // Optimize different aspects in rotation
            match trial {
                0 => {
                    // Position only
                    gaussian.mean.x += rng.gen_range(-lr * 3.0..lr * 3.0);
                    gaussian.mean.y += rng.gen_range(-lr * 3.0..lr * 3.0);
                    gaussian.mean.x = gaussian.mean.x.clamp(0.0, target.width() as f32 - 1.0);
                    gaussian.mean.y = gaussian.mean.y.clamp(0.0, target.height() as f32 - 1.0);
                }
                1 => {
                    // Scale only
                    gaussian.scale.x *= rng.gen_range(0.9..1.1);
                    gaussian.scale.y *= rng.gen_range(0.9..1.1);
                    gaussian.scale.x = gaussian.scale.x.clamp(2.0, 30.0);
                    gaussian.scale.y = gaussian.scale.y.clamp(2.0, 30.0);
                }
                2 => {
                    // Color only
                    for i in 0..3 {
                        gaussian.color[i] += rng.gen_range(-lr * 0.1..lr * 0.1);
                        gaussian.color[i] = gaussian.color[i].clamp(0.0, 1.0);
                    }
                }
                3 => {
                    // Rotation only
                    gaussian.rotation += rng.gen_range(-lr * 0.5..lr * 0.5);
                    gaussian.rotation = gaussian.rotation % (2.0 * std::f32::consts::PI);
                }
                _ => {
                    // Combined small adjustment
                    gaussian.mean.x += rng.gen_range(-lr..lr);
                    gaussian.mean.y += rng.gen_range(-lr..lr);
                    gaussian.scale.x *= rng.gen_range(0.95..1.05);
                    gaussian.scale.y *= rng.gen_range(0.95..1.05);
                    
                    // Apply bounds
                    gaussian.mean.x = gaussian.mean.x.clamp(0.0, target.width() as f32 - 1.0);
                    gaussian.mean.y = gaussian.mean.y.clamp(0.0, target.height() as f32 - 1.0);
                    gaussian.scale.x = gaussian.scale.x.clamp(2.0, 30.0);
                    gaussian.scale.y = gaussian.scale.y.clamp(2.0, 30.0);
                }
            }
            
            // MUCH simpler local loss - just 9 samples!
            let mut local_loss = 0.0;
            let mut sample_count = 0;
            
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = (gaussian.mean.x as i32 + dx * 5).clamp(0, target.width() as i32 - 1) as u32;
                    let py = (gaussian.mean.y as i32 + dy * 3).clamp(0, target.height() as i32 - 1) as u32;
                    
                    let target_pixel = target.get_pixel(px, py);
                    let rendered_pixel = rendered.get_pixel(px, py);
                    let pixel_pos = Vector2::new(px as f32, py as f32);
                    
                    let weight = gaussian.evaluate_at(pixel_pos);
                    if weight > 0.01 {
                        for i in 0..3 {
                            let target_val = target_pixel[i] as f32 / 255.0;
                            let rendered_val = rendered_pixel[i] as f32 / 255.0;
                            let diff = target_val - rendered_val;
                            local_loss += diff * diff;
                        }
                        sample_count += 1;
                    }
                }
            }
            
            if sample_count > 0 {
                local_loss /= sample_count as f32;
                if local_loss < best_loss {
                    best_loss = local_loss;
                    best_params = (gaussian.mean, gaussian.scale, gaussian.rotation, gaussian.color.clone());
                }
            }
        }
        
        // Apply best parameters
        *gaussian = Gaussian2D::new(best_params.0, best_params.2, best_params.1, best_params.3);
    }
}


    /// Compute L1 loss between target and rendered images
    pub fn compute_loss(&self, target: &RgbImage, rendered: &RgbImage) -> f32 {
        let mut total_loss = 0.0;
        let pixel_count = (self.width * self.height) as f32;

        for (target_pixel, rendered_pixel) in target.pixels().zip(rendered.pixels()) {
            for i in 0..3 {
                let target_val = target_pixel[i] as f32 / 255.0;
                let rendered_val = rendered_pixel[i] as f32 / 255.0;
                total_loss += (target_val - rendered_val).abs();
            }
        }

        total_loss / (pixel_count * 3.0)
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

    /// EFFICIENT training loop
    pub fn fit_to_image(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        let target = image::open(target_path)?.to_rgb8();
        self.width = target.width();
        self.height = target.height();
        
        // Clear iterations directory
        let iterations_dir = "iterations";
        if std::path::Path::new(iterations_dir).exists() {
            std::fs::remove_dir_all(iterations_dir).ok();
        }
        std::fs::create_dir_all(iterations_dir)?;
        
        // Use smart initialization
        self.smart_initialize(&target);
        println!("Starting with {} Gaussians", self.gaussians.len());
        
        for iter in 0..iterations {
            let rendered = self.render();
            let loss = self.compute_loss(&target, &rendered);
            
            // Simple but effective learning rate
            let learning_rate = 1.5 * (0.995_f32).powi(iter as i32 / 20);
            
            if iter % 20 == 0 {
                println!("Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {}", 
                         iter, loss, learning_rate, self.gaussians.len());
                rendered.save(format!("iterations/progress_{:04}.png", iter))?;
            }
            
            // Optimize
            self.optimize_step_with_lr(&target, &rendered, learning_rate);
            
            // Less aggressive adaptive addition - every 25 iterations
            if iter % 40 == 0 && iter > 0 {
                let before_count = self.gaussians.len();
                self.moderate_adaptive_addition(&target, &rendered);
                let added_count = self.gaussians.len() - before_count;
                if added_count > 0 {
                    println!("Added {} Gaussians. Total: {}", added_count, self.gaussians.len());
                }
            }
            
            // Pruning every 50 iterations
            if iter % 50 == 0 && iter > 0 {
                self.prune_ineffective_gaussians();
            }
        }
        
        println!("Training complete! Final Gaussians: {}", self.gaussians.len());
        Ok(())
    }

    /// GPU training method
    pub async fn fit_to_image_gpu(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        let target = image::open(target_path)?.to_rgb8();
        self.width = target.width();
        self.height = target.height();
        
        let iterations_dir = "iterations_gpu";
        if std::path::Path::new(iterations_dir).exists() {
            std::fs::remove_dir_all(iterations_dir).ok();
        }
        std::fs::create_dir_all(iterations_dir)?;
        
        let renderer = super::gpu_render::GpuRenderer::new().await?;
        
        self.smart_initialize(&target); // Use smart_initialize instead of hierarchical
        println!("Starting GPU optimization with {} initial Gaussians", self.gaussians.len());
        
        for iter in 0..iterations {
            let rendered = renderer.render_gpu(self).await?;
            let loss = self.compute_loss(&target, &rendered);
            
            let learning_rate = 1.5 * (0.995_f32).powi(iter as i32 / 20);
            
            if iter % 10 == 0 {
                println!("GPU Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {}", 
                         iter, loss, learning_rate, self.gaussians.len());
                rendered.save(format!("iterations_gpu/gpu_progress_{:04}.png", iter))?;
            }
            
            self.optimize_step_with_lr(&target, &rendered, learning_rate);
            
            if iter % 40 == 0 && iter > 0 {
                let before_count = self.gaussians.len();
                self.moderate_adaptive_addition(&target, &rendered);
                let added_count = self.gaussians.len() - before_count;
                if added_count > 0 {
                    println!("Added {} Gaussians. Total: {}", added_count, self.gaussians.len());
                }
            }
            
            if iter % 50 == 0 && iter > 0 {
                self.prune_ineffective_gaussians();
            }
        }
        
        println!("GPU optimization complete! Final Gaussians: {}", self.gaussians.len());
        Ok(())
    }

    /// Render the Gaussians to an image
    pub fn render(&self) -> RgbImage {
        let mut buffer = vec![vec![0.0f32; 3]; (self.width * self.height) as usize];

        buffer.par_chunks_mut(self.width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel_color) in row.iter_mut().enumerate() {
                    let pixel_pos = Vector2::new(x as f32, y as f32);

                    for gaussian in &self.gaussians {
                        if gaussian.is_relevant(pixel_pos, 3.0) {
                            let weight = gaussian.evaluate_at(pixel_pos);
                            let alpha = weight.min(1.0);

                            for i in 0..3 {
                                pixel_color[i] = alpha * gaussian.color[i] + (1.0 - alpha) * pixel_color[i];
                            }
                        }
                    }
                }
            });

        let mut image = ImageBuffer::new(self.width, self.height);
        for (i, pixel_data) in buffer.iter().enumerate() {
            let x = (i % self.width as usize) as u32;
            let y = (i / self.width as usize) as u32;
            let r = (pixel_data[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (pixel_data[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (pixel_data[2].clamp(0.0, 1.0) * 255.0) as u8;
            image.put_pixel(x, y, Rgb([r, g, b]));
        }

        image
    }
}
