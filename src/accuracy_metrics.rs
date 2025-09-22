// src/accuracy_metrics.rs
use image::RgbImage;

#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub psnr: f32,
    pub ssim: f32,
    pub lpips: f32,
}

impl QualityMetrics {
    pub fn print(&self) {
        println!("\n📊 Quality Metrics:");
        println!("  🎯 PSNR (Peak Signal-to-Noise Ratio): {:.2} dB", self.psnr);
        println!("  🏗️  SSIM (Structural Similarity): {:.4}", self.ssim);
        println!("  👁️  LPIPS (Perceptual Similarity): {:.4}", self.lpips);
        
        // Quality assessment
        println!("\n📈 Quality Assessment:");
        if self.psnr > 30.0 {
            println!("  ✅ PSNR: Excellent reconstruction quality");
        } else if self.psnr > 25.0 {
            println!("  ✅ PSNR: Good reconstruction quality");
        } else if self.psnr > 20.0 {
            println!("  ⚠️  PSNR: Fair reconstruction quality");
        } else {
            println!("  ❌ PSNR: Poor reconstruction quality");
        }
        
        if self.ssim > 0.9 {
            println!("  ✅ SSIM: Excellent structural similarity");
        } else if self.ssim > 0.8 {
            println!("  ✅ SSIM: Good structural similarity");
        } else if self.ssim > 0.7 {
            println!("  ⚠️  SSIM: Fair structural similarity");
        } else {
            println!("  ❌ SSIM: Poor structural similarity");
        }
        
        if self.lpips < 0.1 {
            println!("  ✅ LPIPS: Excellent perceptual similarity");
        } else if self.lpips < 0.2 {
            println!("  ✅ LPIPS: Good perceptual similarity");
        } else if self.lpips < 0.3 {
            println!("  ⚠️  LPIPS: Fair perceptual similarity");
        } else {
            println!("  ❌ LPIPS: Poor perceptual similarity");
        }
    }
}

/// Compute PSNR (Peak Signal-to-Noise Ratio)
/// Higher values indicate better quality (typically 20-40+ dB)
pub fn compute_psnr(target: &RgbImage, rendered: &RgbImage) -> f32 {
    let mut mse = 0.0;
    let pixel_count = (target.width() * target.height() * 3) as f32;
    
    for (target_pixel, rendered_pixel) in target.pixels().zip(rendered.pixels()) {
        for i in 0..3 {
            let target_val = target_pixel[i] as f32;
            let rendered_val = rendered_pixel[i] as f32;
            let diff = target_val - rendered_val;
            mse += diff * diff;
        }
    }
    
    if mse == 0.0 {
        return f32::INFINITY; // Perfect match
    }
    
    mse /= pixel_count;
    20.0 * (255.0_f32).log10() - 10.0 * mse.log10()
}

/// Compute SSIM (Structural Similarity Index)
/// Range: 0-1, where 1.0 indicates perfect similarity
pub fn compute_ssim(target: &RgbImage, rendered: &RgbImage) -> f32 {
    let width = target.width() as i32;
    let height = target.height() as i32;
    let window_size = 11;
    let k1 = 0.01;
    let k2 = 0.03;
    let c1 = (k1 * 255.0_f32).powi(2);
    let c2 = (k2 * 255.0_f32).powi(2);

    
    let mut ssim_sum = 0.0;
    let mut valid_windows = 0;
    
    // Process overlapping windows
    for y in (window_size/2)..(height - window_size/2) {
        for x in (window_size/2)..(width - window_size/2) {
            let (mu1, mu2, sigma1_sq, sigma2_sq, sigma12) = 
                compute_window_stats(target, rendered, x, y, window_size);
            
            let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
            let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);
            
            if denominator != 0.0 {
                ssim_sum += numerator / denominator;
                valid_windows += 1;
            }
        }
    }
    
    if valid_windows > 0 {
        ssim_sum / valid_windows as f32
    } else {
        0.0
    }
}

/// Compute simplified LPIPS (Learned Perceptual Image Patch Similarity)
/// This is a simplified version without deep learning features
/// Range: 0+ (lower is better, 0 = identical)
pub fn compute_lpips(target: &RgbImage, rendered: &RgbImage) -> f32 {
    let width = target.width() as i32;
    let height = target.height() as i32;
    let patch_size = 64;
    let stride = 32;
    
    let mut total_distance = 0.0;
    let mut patch_count = 0;
    
    // Extract patches and compute perceptual features
    for y in (0..height - patch_size).step_by(stride as usize) {
        for x in (0..width - patch_size).step_by(stride as usize) {
            let target_features = extract_patch_features(target, x, y, patch_size);
            let rendered_features = extract_patch_features(rendered, x, y, patch_size);
            
            // Compute normalized feature distance
            let mut feature_distance = 0.0;
            for (t_feat, r_feat) in target_features.iter().zip(rendered_features.iter()) {
                feature_distance += (t_feat - r_feat).powi(2);
            }
            
            total_distance += feature_distance.sqrt();
            patch_count += 1;
        }
    }
    
    if patch_count > 0 {
        total_distance / patch_count as f32
    } else {
        0.0
    }
}

/// Helper function to compute window statistics for SSIM
fn compute_window_stats(
    target: &RgbImage, 
    rendered: &RgbImage, 
    center_x: i32, 
    center_y: i32, 
    window_size: i32
) -> (f32, f32, f32, f32, f32) {
    let half_window = window_size / 2;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum12 = 0.0;
    let mut count = 0;
    
    for dy in -half_window..=half_window {
        for dx in -half_window..=half_window {
            let x = center_x + dx;
            let y = center_y + dy;
            
            if x >= 0 && x < target.width() as i32 && y >= 0 && y < target.height() as i32 {
                let target_pixel = target.get_pixel(x as u32, y as u32);
                let rendered_pixel = rendered.get_pixel(x as u32, y as u32);
                
                // Convert to grayscale for SSIM computation
                let target_gray = (target_pixel[0] as f32 * 0.299 + 
                                 target_pixel[1] as f32 * 0.587 + 
                                 target_pixel[2] as f32 * 0.114) / 255.0;
                let rendered_gray = (rendered_pixel[0] as f32 * 0.299 + 
                                   rendered_pixel[1] as f32 * 0.587 + 
                                   rendered_pixel[2] as f32 * 0.114) / 255.0;
                
                sum1 += target_gray;
                sum2 += rendered_gray;
                sum1_sq += target_gray * target_gray;
                sum2_sq += rendered_gray * rendered_gray;
                sum12 += target_gray * rendered_gray;
                count += 1;
            }
        }
    }
    
    if count > 0 {
        let n = count as f32;
        let mu1 = sum1 / n;
        let mu2 = sum2 / n;
        let sigma1_sq = (sum1_sq / n) - (mu1 * mu1);
        let sigma2_sq = (sum2_sq / n) - (mu2 * mu2);
        let sigma12 = (sum12 / n) - (mu1 * mu2);
        
        (mu1, mu2, sigma1_sq, sigma2_sq, sigma12)
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    }
}

/// Extract simple perceptual features from a patch for LPIPS
fn extract_patch_features(image: &RgbImage, x: i32, y: i32, patch_size: i32) -> Vec<f32> {
    let mut features = Vec::new();
    
    // Color statistics
    let mut r_sum = 0.0;
    let mut g_sum = 0.0;
    let mut b_sum = 0.0;
    let mut brightness_sum = 0.0;
    let mut _contrast_sum = 0.0;
    let mut pixel_count = 0;
    
    for dy in 0..patch_size {
        for dx in 0..patch_size {
            let px = x + dx;
            let py = y + dy;
            
            if px >= 0 && px < image.width() as i32 && py >= 0 && py < image.height() as i32 {
                let pixel = image.get_pixel(px as u32, py as u32);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                
                r_sum += r;
                g_sum += g;
                b_sum += b;
                brightness_sum += (r + g + b) / 3.0;
                pixel_count += 1;
            }
        }
    }
    
    if pixel_count > 0 {
        let n = pixel_count as f32;
        features.push(r_sum / n);      // Mean R
        features.push(g_sum / n);      // Mean G
        features.push(b_sum / n);      // Mean B
        features.push(brightness_sum / n); // Mean brightness
        
        // Compute contrast (standard deviation of brightness)
        let mean_brightness = brightness_sum / n;
        let mut variance = 0.0;
        
        for dy in 0..patch_size {
            for dx in 0..patch_size {
                let px = x + dx;
                let py = y + dy;
                
                if px >= 0 && px < image.width() as i32 && py >= 0 && py < image.height() as i32 {
                    let pixel = image.get_pixel(px as u32, py as u32);
                    let brightness = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / (3.0 * 255.0);
                    variance += (brightness - mean_brightness).powi(2);
                }
            }
        }
        
        features.push((variance / n).sqrt()); // Contrast (std dev)
    }
    
    features
}

/// Evaluate all quality metrics at once
pub fn evaluate_all_metrics(target: &RgbImage, rendered: &RgbImage) -> QualityMetrics {
    println!("🔍 Computing quality metrics...");
    
    let psnr = compute_psnr(target, rendered);
    let ssim = compute_ssim(target, rendered);
    let lpips = compute_lpips(target, rendered);
    
    QualityMetrics { psnr, ssim, lpips }
}
