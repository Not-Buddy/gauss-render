// src/main.rs
mod gaussian;
mod image_gs;

use image_gs::ImageGS;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    loop {
        show_main_menu();
        let choice = get_user_input("Enter your choice (1-4): ");
        
        match choice.trim() {
            "1" => run_cpu_mode()?,
            "2" => {
                pollster::block_on(async {
                    run_gpu_mode().await
                })?;
            }
            "3" => {
                pollster::block_on(async {
                    run_comparison_mode().await
                })?;
            }
            "4" => {
                println!("👋 Goodbye!");
                break;
            }
            _ => {
                println!("❌ Invalid choice! Please select 1-4.");
                continue;
            }
        }

        println!("\n🎉 Task completed! Press Enter to continue...");
        get_user_input("");
    }
    
    Ok(())
}

fn show_main_menu() {
    println!("\n🚀 2D Gaussian Splatting Renderer");
    println!("=====================================");
    println!("1. 💻 CPU Mode - Traditional rendering");
    println!("2. 🚀 GPU Mode - Intel Arc accelerated");
    println!("3. ⚡ Comparison - Both CPU and GPU");
    println!("4. 🚪 Exit");
    println!("=====================================");
}

fn get_user_input(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    
    input
}

fn get_settings() -> (String, u32, u32, usize) {
    println!("\n⚙️ Configuration Settings:");
    
    // Get image path
    let default_image = "test_images/InfinityCastle.jpeg".to_string();
    print!("📁 Image path (default: {}): ", default_image);
    io::stdout().flush().unwrap();
    let mut image_path = String::new();
    io::stdin().read_line(&mut image_path).unwrap();
    let image_path = if image_path.trim().is_empty() {
        default_image
    } else {
        image_path.trim().to_string()
    };
    
    // Get image dimensions
    let width = get_number_input("📐 Image width (default: 400): ", 400);
    let height = get_number_input("📐 Image height (default: 400): ", 400);
    
    // Get iteration count
    let iterations = get_number_input("🔄 Training iterations (default: 200): ", 200) as usize;
    
    (image_path, width, height, iterations)
}

fn get_number_input(prompt: &str, default: u32) -> u32 {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    if input.trim().is_empty() {
        default
    } else {
        input.trim().parse().unwrap_or(default)
    }
}

fn run_cpu_mode() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💻 Starting CPU Mode...");
    let (image_path, width, height, iterations) = get_settings();
    
    let start_time = std::time::Instant::now();
    
    let mut image_gs = ImageGS::new(width, height);
    
    // Test rendering first
    image_gs.initialize_random(20);
    let test_result = image_gs.render();
    test_result.save("cpu_test_output.png")?;
    println!("✅ CPU test render saved as 'cpu_test_output.png'");
    
    // Training
    println!("🎯 Starting CPU training with {} iterations...", iterations);
    image_gs.fit_to_image(&image_path, iterations)?;
    
    // Final render
    let final_result = image_gs.render();
    final_result.save("cpu_final_output.png")?;
    
    let duration = start_time.elapsed();
    println!("✅ CPU training complete!");
    println!("⏱️  Total time: {:.2}s", duration.as_secs_f64());
    println!("📊 Final Gaussians: {}", image_gs.gaussian_count());
    println!("💾 Results saved as 'cpu_final_output.png'");
    
    Ok(())
}

async fn run_gpu_mode() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Starting GPU Mode with Intel Arc Graphics...");
    let (image_path, width, height, iterations) = get_settings();
    
    let start_time = std::time::Instant::now();
    
    let mut image_gs = ImageGS::new(width, height);
    
    // Create GPU renderer
    println!("🔧 Initializing Intel Arc GPU...");
    let gpu_renderer = image_gs::GpuRenderer::new().await?;
    
    // Test rendering first
    image_gs.initialize_random(50); // More Gaussians with GPU power!
    let test_result = gpu_renderer.render_gpu(&image_gs).await?;
    test_result.save("gpu_test_output.png")?;
    println!("✅ GPU test render saved as 'gpu_test_output.png'");
    
    // Training
    println!("🎯 Starting GPU-accelerated training with {} iterations...", iterations);
    image_gs.fit_to_image_gpu(&image_path, iterations).await?;
    
    // Final render
    let final_result = gpu_renderer.render_gpu(&image_gs).await?;
    final_result.save("test_images/output/gpu_final_output.png")?;
    
    let duration = start_time.elapsed();
    println!("✅ GPU training complete!");
    println!("⏱️  Total time: {:.2}s", duration.as_secs_f64());
    println!("📊 Final Gaussians: {}", image_gs.gaussian_count());
    println!("💾 Results saved as 'gpu_final_output.png'");
    
    Ok(())
}

async fn run_comparison_mode() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Starting Comparison Mode (CPU vs GPU)...");
    let (image_path, width, height, iterations) = get_settings();
    
    println!("🔄 This will run both CPU and GPU versions for comparison...");
    
    // CPU Version
    println!("\n--- 💻 CPU Phase ---");
    let cpu_start = std::time::Instant::now();
    
    let mut cpu_image_gs = ImageGS::new(width, height);
    cpu_image_gs.initialize_random(20);
    cpu_image_gs.fit_to_image(&image_path, iterations)?;
    let cpu_result = cpu_image_gs.render();
    cpu_result.save("comparison_cpu.png")?;
    
    let cpu_duration = cpu_start.elapsed();
    
    // GPU Version
    println!("\n--- 🚀 GPU Phase ---");
    let gpu_start = std::time::Instant::now();
    
    let mut gpu_image_gs = ImageGS::new(width, height);
    let gpu_renderer = image_gs::GpuRenderer::new().await?;
    gpu_image_gs.initialize_random(50); // More Gaussians for GPU
    gpu_image_gs.fit_to_image_gpu(&image_path, iterations).await?;
    let gpu_result = gpu_renderer.render_gpu(&gpu_image_gs).await?;
    gpu_result.save("comparison_gpu.png")?;
    
    let gpu_duration = gpu_start.elapsed();
    
    // Show comparison results
    println!("\n📊 === PERFORMANCE COMPARISON ===");
    println!("💻 CPU Results:");
    println!("   ⏱️  Time: {:.2}s", cpu_duration.as_secs_f64());
    println!("   📊 Gaussians: {}", cpu_image_gs.gaussian_count());
    println!("   💾 Output: comparison_cpu.png");
    
    println!("🚀 GPU Results:");
    println!("   ⏱️  Time: {:.2}s", gpu_duration.as_secs_f64());
    println!("   📊 Gaussians: {}", gpu_image_gs.gaussian_count());
    println!("   💾 Output: comparison_gpu.png");
    
    let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
    println!("⚡ GPU Speedup: {:.1}x faster", speedup);
    
    if speedup > 1.0 {
        println!("🎉 Intel Arc GPU wins! 🏆");
    } else {
        println!("🤔 CPU performed better this time");
    }
    
    Ok(())
}
