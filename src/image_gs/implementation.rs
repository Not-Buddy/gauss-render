// src/image_gs/implementation.rs

use super::ImageGS;
use crate::gaussian::Gaussian2D;
use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::Vector2;
use rand::prelude::*;
use rayon::prelude::*;

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

    /// Initialize from target image
    pub fn initialize_from_image(&mut self, target: &RgbImage) {
        let mut rng = thread_rng();
        self.gaussians.clear();

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
                    rng.gen_range(8.0..25.0),
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

    /// Optimization step with learning rate
    pub fn optimize_step_with_lr(&mut self, target: &RgbImage, _rendered: &RgbImage, lr: f32) {
        let mut rng = thread_rng();

        // Optimize a subset of Gaussians for performance
        for gaussian in &mut self.gaussians.iter_mut().take(5) {
            let original_mean = gaussian.mean;
            let original_color = gaussian.color.clone();

            let mut best_loss = f32::INFINITY;
            let mut best_mean = original_mean;
            let mut best_color = original_color.clone();

            // Try multiple variations and keep the best
            for _ in 0..8 {
                gaussian.mean.x = original_mean.x + rng.gen_range(-lr * 5.0..lr * 5.0);
                gaussian.mean.y = original_mean.y + rng.gen_range(-lr * 5.0..lr * 5.0);
                gaussian.mean.x = gaussian.mean.x.clamp(0.0, self.width as f32);
                gaussian.mean.y = gaussian.mean.y.clamp(0.0, self.height as f32);

                for i in 0..3 {
                    gaussian.color[i] = original_color[i] + rng.gen_range(-lr * 0.1..lr * 0.1);
                    gaussian.color[i] = gaussian.color[i].clamp(0.0, 1.0);
                }

                // Compute local loss
                let mut local_loss = 0.0;
                let sample_count = 25;

                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let px = (gaussian.mean.x as i32 + dx * 3).clamp(0, self.width as i32 - 1) as u32;
                        let py = (gaussian.mean.y as i32 + dy * 3).clamp(0, self.height as i32 - 1) as u32;

                        let target_pixel = target.get_pixel(px, py);
                        let pixel_pos = Vector2::new(px as f32, py as f32);

                        let weight = gaussian.evaluate_at(pixel_pos);
                        let alpha = weight.min(1.0);

                        for i in 0..3 {
                            let target_val = target_pixel[i] as f32 / 255.0;
                            let predicted_val = alpha * gaussian.color[i];
                            local_loss += (target_val - predicted_val).abs();
                        }
                    }
                }

                local_loss /= sample_count as f32 * 3.0;

                if local_loss < best_loss {
                    best_loss = local_loss;
                    best_mean = gaussian.mean;
                    best_color = gaussian.color.clone();
                }
            }

            // Apply best found parameters
            gaussian.mean = best_mean;
            gaussian.color = best_color;
        }
    }

    /// Main training loop with enhanced adaptive density control
    pub fn fit_to_image(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        let target = image::open(target_path)?.to_rgb8();
        let (width, height) = target.dimensions();

        // Allow larger images for better quality
        let max_size = 512;
        let (width, height) = if width > max_size || height > max_size {
            println!("Resizing large image for performance...");
            let scale = (max_size as f32) / (width.max(height) as f32);
            ((width as f32 * scale) as u32, (height as f32 * scale) as u32)
        } else {
            (width, height)
        };

        let target = image::imageops::resize(&target, width, height, image::imageops::FilterType::Lanczos3);

        self.width = width;
        self.height = height;

        // Clear and recreate iterations directory
        let iterations_dir = "iterations";
        if std::path::Path::new(iterations_dir).exists() {
            println!("Clearing previous iterations...");
            match std::fs::remove_dir_all(iterations_dir) {
                Ok(_) => println!("Successfully cleared iterations directory"),
                Err(e) => {
                    println!("Warning: Failed to clear iterations directory: {}", e);
                    println!("Continuing anyway...");
                }
            }
        }

        std::fs::create_dir_all(iterations_dir)?;
        println!("Iterations directory ready");

        // Start with hierarchical initialization
        self.hierarchical_initialize(&target);
        println!("Starting optimization with {} initial Gaussians", self.gaussians.len());

        for iter in 0..iterations {
            let rendered = self.render();
            let loss = self.compute_loss(&target, &rendered);

            // Improved learning rate schedule
            let learning_rate = if iter < 50 {
                3.0 * (0.98_f32).powi(iter as i32 / 10)
            } else {
                1.5 * (0.995_f32).powi(iter as i32 / 15)
            };

            if iter % 15 == 0 {
                println!("Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {}", 
                         iter, loss, learning_rate, self.gaussians.len());
                rendered.save(format!("iterations/progress_{:04}.png", iter))?;
            }

            // Optimization step
            self.optimize_step_with_lr(&target, &rendered, learning_rate);

            // Enhanced adaptive density control
            if iter % 15 == 0 && iter > 0 {
                println!("Running enhanced adaptive density control...");
                let before_count = self.gaussians.len();
                self.enhanced_adaptive_addition(&target, &rendered);
                let added_count = self.gaussians.len() - before_count;
                println!("Added {} Gaussians in high-error regions. Total: {}", 
                         added_count, self.gaussians.len());
            }

            // Gaussian pruning
            if iter % 50 == 0 && iter > 0 {
                self.prune_ineffective_gaussians();
            }
        }

        println!("Optimization complete! Final Gaussians: {}", self.gaussians.len());
        Ok(())
    }

    pub async fn fit_to_image_gpu(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        let target = image::open(target_path)?.to_rgb8();
        let (width, height) = target.dimensions();

        // Allow larger images with GPU acceleration
        let max_size = 1024; // Increased from 512 thanks to GPU power!
        let (width, height) = if width > max_size || height > max_size {
            println!("Resizing large image for GPU processing...");
            let scale = (max_size as f32) / (width.max(height) as f32);
            ((width as f32 * scale) as u32, (height as f32 * scale) as u32)
        } else {
            (width, height)
        };

        let target = image::imageops::resize(&target, width, height, image::imageops::FilterType::Lanczos3);

        self.width = width;
        self.height = height;

        // Clear iterations directory
        let iterations_dir = "iterations_gpu";
        if std::path::Path::new(iterations_dir).exists() {
            println!("Clearing previous GPU iterations...");
            std::fs::remove_dir_all(iterations_dir).ok();
        }
        std::fs::create_dir_all(iterations_dir)?;

        // Create GPU renderer once
        let renderer = super::gpu_render::GpuRenderer::new().await?;
        
        // Hierarchical initialization with more Gaussians for GPU
        self.hierarchical_initialize(&target);
        println!("Starting GPU optimization with {} initial Gaussians", self.gaussians.len());

        for iter in 0..iterations {
            // Use GPU for rendering!
            let rendered = renderer.render_gpu(self).await?;
            let loss = self.compute_loss(&target, &rendered);

            let learning_rate = if iter < 100 {
                4.0 * (0.99_f32).powi(iter as i32 / 20) // More aggressive with GPU
            } else {
                2.0 * (0.997_f32).powi(iter as i32 / 25)
            };

            if iter % 10 == 0 { // More frequent saves with faster GPU
                println!("GPU Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {}", 
                         iter, loss, learning_rate, self.gaussians.len());
                rendered.save(format!("iterations_gpu/gpu_progress_{:04}.png", iter))?;
            }

            // CPU optimization step (could be GPU-accelerated too in future)
            self.optimize_step_with_lr(&target, &rendered, learning_rate);

            // More frequent adaptive addition with GPU power
            if iter % 10 == 0 && iter > 0 {
                println!("Running GPU-enhanced adaptive density control...");
                let before_count = self.gaussians.len();
                self.enhanced_adaptive_addition(&target, &rendered);
                let added_count = self.gaussians.len() - before_count;
                println!("Added {} Gaussians. Total: {}", added_count, self.gaussians.len());
            }

            // Less frequent pruning
            if iter % 75 == 0 && iter > 0 {
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
