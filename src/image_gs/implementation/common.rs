// src/image_gs/implementation/common.rs
use super::super::ImageGS;
use crate::gaussian::Gaussian2D;
use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::Vector2;
use rand::prelude::*;
use rayon::prelude::*;

const GAUSSIAN_NUMBER: usize = 12000;

impl ImageGS {
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

    /// Dense smart initialization based on image content
    pub fn dense_smart_initialize(&mut self, target: &RgbImage) {
        let mut rng = thread_rng();
        self.gaussians.clear();
        
        let target_count = 400;
        
        // Calculate per-pixel brightness to guide initialization
        let mut pixel_importance = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = target.get_pixel(x, y);
                let brightness = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / (3.0 * 255.0);
                pixel_importance.push((x, y, brightness));
            }
        }
        
        // Sort by brightness - focus on visible areas first
        pixel_importance.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        // Dense grid initialization based on image content
        let grid_size = (target_count as f32 * 0.8) as usize; // 80% grid-based
        let cols = (grid_size as f32).sqrt() as usize;
        let rows = (grid_size + cols - 1) / cols;
        
        for row in 0..rows {
            for col in 0..cols {
                if self.gaussians.len() >= grid_size {
                    break;
                }
                
                let x = ((col as f32 + 0.5) * self.width as f32 / cols as f32) as u32;
                let y = ((row as f32 + 0.5) * self.height as f32 / rows as f32) as u32;
                let x = x.clamp(0, self.width - 1);
                let y = y.clamp(0, self.height - 1);
                
                let pixel = target.get_pixel(x, y);
                let brightness = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / (3.0 * 255.0);
                
                let color = vec![
                    (pixel[0] as f32 / 255.0).max(0.1),
                    (pixel[1] as f32 / 255.0).max(0.1),
                    (pixel[2] as f32 / 255.0).max(0.1),
                ];
                
                let base_scale = if brightness > 0.3 {
                    rng.gen_range(6.0..15.0)
                } else if brightness > 0.1 {
                    rng.gen_range(10.0..25.0)
                } else {
                    rng.gen_range(15.0..35.0)
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
        
        // Content-aware random initialization for remaining 20%
        let remaining_count = target_count - self.gaussians.len();
        let important_pixels = &pixel_importance[0..(pixel_importance.len() / 3).min(remaining_count * 2)];
        
        for i in 0..remaining_count {
            let idx = i % important_pixels.len();
            let (x, y, _brightness) = important_pixels[idx];
            
            let jitter_x = (x as f32 + rng.gen_range(-5.0..5.0)).clamp(0.0, self.width as f32 - 1.0);
            let jitter_y = (y as f32 + rng.gen_range(-5.0..5.0)).clamp(0.0, self.height as f32 - 1.0);
            
            let pixel = target.get_pixel(x, y);
            let color = vec![
                ((pixel[0] as f32 / 255.0) * 1.5).min(1.0).max(0.1),
                ((pixel[1] as f32 / 255.0) * 1.5).min(1.0).max(0.1),
                ((pixel[2] as f32 / 255.0) * 1.5).min(1.0).max(0.1),
            ];
            
            let gaussian = Gaussian2D::new(
                Vector2::new(jitter_x, jitter_y),
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(8.0..18.0),
                    rng.gen_range(8.0..18.0),
                ),
                color,
            );
            
            self.gaussians.push(gaussian);
        }
        
        println!("Dense smart initialized {} Gaussians", self.gaussians.len());
    }

    /// Moderate adaptive addition of Gaussians based on error
    pub fn moderate_adaptive_addition(&mut self, target: &RgbImage, rendered: &RgbImage) {
        let mut candidates = Vec::new();
        
        for y in (5..self.height-5).step_by(8) {
            for x in (5..self.width-5).step_by(8) {
                let target_pixel = target.get_pixel(x, y);
                let rendered_pixel = rendered.get_pixel(x, y);
                
                let mut error = 0.0;
                for i in 0..3 {
                    let diff = (target_pixel[i] as f32 / 255.0) - (rendered_pixel[i] as f32 / 255.0);
                    error += diff * diff;
                }
                
                if error > 0.08 {
                    candidates.push((Vector2::new(x as f32, y as f32), error));
                }
            }
        }
        
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut rng = thread_rng();
        let add_count = candidates.len().min(GAUSSIAN_NUMBER);
        
        for (pos, _) in candidates.iter().take(add_count) {
            let target_pixel = target.get_pixel(pos.x as u32, pos.y as u32);
            let color = vec![
                (target_pixel[0] as f32 / 255.0),
                (target_pixel[1] as f32 / 255.0),
                (target_pixel[2] as f32 / 255.0),
            ];
            
            let gaussian = Gaussian2D::new(
                *pos,
                rng.gen_range(0.0..std::f32::consts::PI),
                Vector2::new(
                    rng.gen_range(4.0..12.0),
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
            let opacity = gaussian.color.iter().sum::<f32>() / gaussian.color.len() as f32;
            opacity >= min_opacity_threshold && max_scale <= max_scale_threshold
        });
        
        let pruned_count = initial_count - self.gaussians.len();
        if pruned_count > 0 {
            println!("Pruned {} ineffective Gaussians", pruned_count);
        }
    }

    /// Optimize parameters with learning rate
    pub fn optimize_step_with_lr(&mut self, target: &RgbImage, rendered: &RgbImage, lr: f32) {
        let mut rng = thread_rng();
        let gaussians_to_optimize = (self.gaussians.len() / 6).min(25).max(8);
        
        let mut gaussian_indices: Vec<usize> = (0..self.gaussians.len()).collect();
        gaussian_indices.shuffle(&mut rng);
        
        for idx in gaussian_indices.iter().take(gaussians_to_optimize) {
            let gaussian = &mut self.gaussians[*idx];
            let original_params = (gaussian.mean, gaussian.scale, gaussian.rotation, gaussian.color.clone());
            let mut best_loss = f32::INFINITY;
            let mut best_params = original_params.clone();
            
            for trial in 0..3 {
                *gaussian = Gaussian2D::new(original_params.0, original_params.2, original_params.1, original_params.3.clone());
                
                match trial {
                    0 => { // Position only
                        gaussian.mean.x += rng.gen_range(-lr * 3.0..lr * 3.0);
                        gaussian.mean.y += rng.gen_range(-lr * 3.0..lr * 3.0);
                        gaussian.mean.x = gaussian.mean.x.clamp(0.0, target.width() as f32 - 1.0);
                        gaussian.mean.y = gaussian.mean.y.clamp(0.0, target.height() as f32 - 1.0);
                    },
                    1 => { // Scale only
                        gaussian.scale.x *= rng.gen_range(0.9..1.1);
                        gaussian.scale.y *= rng.gen_range(0.9..1.1);
                        gaussian.scale.x = gaussian.scale.x.clamp(2.0, 30.0);
                        gaussian.scale.y = gaussian.scale.y.clamp(2.0, 30.0);
                    },
                    2 => { // Color only
                        for i in 0..3 {
                            gaussian.color[i] += rng.gen_range(-lr * 0.1..lr * 0.1);
                            gaussian.color[i] = gaussian.color[i].clamp(0.0, 1.0);
                        }
                    },
                    _ => { // Combined adjustment
                        gaussian.mean.x += rng.gen_range(-lr..lr);
                        gaussian.mean.y += rng.gen_range(-lr..lr);
                        gaussian.scale.x *= rng.gen_range(0.95..1.05);
                        gaussian.scale.y *= rng.gen_range(0.95..1.05);
                        
                        gaussian.mean.x = gaussian.mean.x.clamp(0.0, target.width() as f32 - 1.0);
                        gaussian.mean.y = gaussian.mean.y.clamp(0.0, target.height() as f32 - 1.0);
                        gaussian.scale.x = gaussian.scale.x.clamp(2.0, 30.0);
                        gaussian.scale.y = gaussian.scale.y.clamp(2.0, 30.0);
                    }
                }
                
                // Simple local loss calculation
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
                                sample_count += 1;
                            }
                        }
                    }
                }
                
                if sample_count > 0 {
                    local_loss /= sample_count as f32;
                }
                
                if local_loss < best_loss {
                    best_loss = local_loss;
                    best_params = (gaussian.mean, gaussian.scale, gaussian.rotation, gaussian.color.clone());
                }
            }
            
            *gaussian = Gaussian2D::new(best_params.0, best_params.2, best_params.1, best_params.3);
        }
    }

    /// Render the Gaussians to an image
    pub fn render(&self) -> RgbImage {
        let mut buffer = vec![vec![0.1f32; 3]; (self.width * self.height) as usize];
        
        buffer.par_chunks_mut(self.width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel_color) in row.iter_mut().enumerate() {
                    let pixel_pos = Vector2::new(x as f32, y as f32);
                    let mut _total_weight = 0.0;
                    
                    for gaussian in &self.gaussians {
                        if gaussian.is_relevant(pixel_pos, 4.0) {
                            let weight = gaussian.evaluate_at(pixel_pos);
                            if weight > 0.005 {
                                _total_weight += weight;
                                let alpha = (weight * 2.0).min(0.8);
                                
                                for i in 0..3 {
                                    let gaussian_contribution = gaussian.color[i] * alpha;
                                    pixel_color[i] = gaussian_contribution + pixel_color[i] * (1.0 - alpha);
                                }
                            }
                        }
                    }
                    
                    for i in 0..3 {
                        pixel_color[i] = pixel_color[i].max(0.05);
                    }
                }
            });
        
        let mut image = ImageBuffer::new(self.width, self.height);
        for (i, pixel_data) in buffer.iter().enumerate() {
            let x = (i % self.width as usize) as u32;
            let y = (i / self.width as usize) as u32;
            
            let r = (pixel_data[0].clamp(0.0, 1.0).powf(0.8) * 255.0) as u8;
            let g = (pixel_data[1].clamp(0.0, 1.0).powf(0.8) * 255.0) as u8;
            let b = (pixel_data[2].clamp(0.0, 1.0).powf(0.8) * 255.0) as u8;
            
            image.put_pixel(x, y, Rgb([r, g, b]));
        }
        
        image
    }
}
