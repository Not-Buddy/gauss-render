use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::{Vector2, Matrix2, Rotation2};
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian2D {
    pub mean: Vector2<f32>,
    pub rotation: f32,
    pub scale: Vector2<f32>,
    pub color: Vec<f32>,
}

impl Gaussian2D {
    pub fn new(mean: Vector2<f32>, rotation: f32, scale: Vector2<f32>, color: Vec<f32>) -> Self {
        Self { mean, rotation, scale, color }
    }

    pub fn covariance_matrix(&self) -> Matrix2<f32> {
        let rot = Rotation2::new(self.rotation);
        let scale_matrix = Matrix2::new(
            self.scale.x, 0.0,
            0.0, self.scale.y
        );
        rot.matrix() * scale_matrix * rot.matrix().transpose()
    }

    pub fn evaluate_at(&self, x: Vector2<f32>) -> f32 {
        let diff = x - self.mean;
        let cov_inv = self.covariance_matrix().try_inverse().unwrap_or(Matrix2::identity());
        let quadratic_form = diff.transpose() * cov_inv * diff;
        let scalar_value = quadratic_form[(0, 0)];
        (-0.5 * scalar_value).exp()
    }
    
    pub fn is_relevant(&self, pixel: Vector2<f32>, threshold: f32) -> bool {
        let dist_sq = (pixel - self.mean).norm_squared();
        let max_scale = self.scale.x.max(self.scale.y);
        dist_sq < (threshold * max_scale).powi(2)
    }
}

pub struct ImageGS {
    pub gaussians: Vec<Gaussian2D>,
    pub width: u32,
    pub height: u32,
}

impl ImageGS {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            gaussians: Vec::new(),
            width,
            height,
        }
    }

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

    pub fn initialize_from_image(&mut self, target: &RgbImage) {
        let mut rng = thread_rng();
        self.gaussians.clear();
        
        for _ in 0..10 {
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
                    rng.gen_range(20.0..50.0),
                    rng.gen_range(20.0..50.0),
                ),
                color,
            );
            
            self.gaussians.push(gaussian);
        }
    }

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

    // SIMPLIFIED OPTIMIZATION - No complex gradients
    pub fn optimize_step_with_lr(&mut self, target: &RgbImage, _rendered: &RgbImage, lr: f32) {
        let mut rng = thread_rng();
        
        for gaussian in &mut self.gaussians.iter_mut().take(5) { // Only optimize first 5 for speed
            // Store original values
            let original_mean = gaussian.mean;
            let original_color = gaussian.color.clone();
            
            // Try small random adjustments and keep if they improve loss
            let mut best_loss = f32::INFINITY;
            let mut best_mean = original_mean;
            let mut best_color = original_color.clone();
            
            // Test 8 random variations
            for _ in 0..8 {
                // Random position adjustment
                gaussian.mean.x = original_mean.x + rng.gen_range(-lr * 5.0..lr * 5.0);
                gaussian.mean.y = original_mean.y + rng.gen_range(-lr * 5.0..lr * 5.0);
                gaussian.mean.x = gaussian.mean.x.clamp(0.0, self.width as f32);
                gaussian.mean.y = gaussian.mean.y.clamp(0.0, self.height as f32);
                
                // Random color adjustment
                for i in 0..3 {
                    gaussian.color[i] = original_color[i] + rng.gen_range(-lr * 0.1..lr * 0.1);
                    gaussian.color[i] = gaussian.color[i].clamp(0.0, 1.0);
                }
                
                // Quick local loss evaluation (sample a few pixels around this Gaussian)
                let mut local_loss = 0.0;
                let sample_count = 25; // 5x5 grid around Gaussian
                
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let px = (gaussian.mean.x as i32 + dx * 3).clamp(0, self.width as i32 - 1) as u32;
                        let py = (gaussian.mean.y as i32 + dy * 3).clamp(0, self.height as i32 - 1) as u32;
                        
                        let target_pixel = target.get_pixel(px, py);
                        let pixel_pos = Vector2::new(px as f32, py as f32);
                        
                        // Compute this Gaussian's contribution
                        let weight = gaussian.evaluate_at(pixel_pos);
                        let alpha = weight.min(1.0);
                        
                        for i in 0..3 {
                            let target_val = target_pixel[i] as f32 / 255.0;
                            let predicted_val = alpha * gaussian.color[i]; // Simplified prediction
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

    pub fn fit_to_image(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        let target = image::open(target_path)?.to_rgb8();
        let (width, height) = target.dimensions();
        
        let max_size = 300;
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
        
        self.initialize_from_image(&target);
        
        for iter in 0..iterations {
            let rendered = self.render();
            let loss = self.compute_loss(&target, &rendered);
            
            let learning_rate = 2.0 * (0.98_f32).powi(iter as i32 / 25);
            
            if iter % 25 == 0 {
                println!("Iteration {}: Loss = {:.6}, LR = {:.4}", iter, loss, learning_rate);
                rendered.save(format!("progress_{}.png", iter))?;
            }
            
            self.optimize_step_with_lr(&target, &rendered, learning_rate);
            
            if iter % 30 == 0 && iter > 0 {
                self.adaptive_gaussian_addition(&target, &rendered);
                println!("Added Gaussians. Total: {}", self.gaussians.len());
            }
        }
        
        Ok(())
    }

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating optimized 2D Gaussian image...");
    
    let mut image_gs = ImageGS::new(200, 200);
    
    image_gs.initialize_random(5);
    let result = image_gs.render();
    result.save("gaussian_output.png")?;
    println!("Rendered {} random Gaussians", image_gs.gaussians.len());
    
    image_gs.fit_to_image("test_images/InfinityCastle.jpeg", 100)?;
    let final_result = image_gs.render();
    final_result.save("test_images/output/InfinityCastleGauss.jpeg")?;
    
    Ok(())
}
