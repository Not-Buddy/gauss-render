// src/main.rs
mod gaussian;
mod image_gs;
mod accuracy_metrics;


use accuracy_metrics::{evaluate_all_metrics}; 
use image_gs::ImageGS;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    // Check if help is requested
    if args.contains(&"--help".to_string()) {
        show_help();
        return Ok(());
    }
    
    // Check if any mode flags are provided
    let mode = if args.contains(&"--cpu".to_string()) || args.contains(&"-1".to_string()) {
        Some("cpu")
    } else if args.contains(&"--gpu".to_string()) || args.contains(&"-2".to_string()) {
        Some("gpu") 
    } else if args.contains(&"--compare".to_string()) || args.contains(&"-3".to_string()) {
        Some("compare")
    } else {
        None
    };

    match mode {
        Some("cpu") => {
            println!("💻 Running CPU Mode from command line...");
            let settings = parse_settings_from_args(&args);
            run_cpu_mode_with_settings(settings)?;
        }
        Some("gpu") => {
            println!("🚀 Running GPU Mode from command line...");
            let settings = parse_settings_from_args(&args);
            pollster::block_on(async {
                run_gpu_mode_with_settings(settings).await
            })?;
        }
        Some(_) => {
            // This handles any unexpected string values
            println!("❌ Unknown mode. Use --help for usage information.");
            show_help();
        }
        None => {
            // No flags provided, show help or run interactive mode
            if args.len() == 1 {
                println!("No flags provided. Starting interactive mode...");
                run_interactive_mode()?;
            } else {
                println!("❌ Invalid arguments. Use --help for usage information.");
                show_help();
            }
        }
    }

    Ok(())
}

fn show_help() {
    println!("🚀 2D Gaussian Splatting Renderer");
    println!("=====================================");
    println!("Usage: gauss-render [MODE] [OPTIONS]");
    println!();
    println!("MODES:");
    println!("  -1, --cpu      💻 CPU Mode - Traditional rendering");
    println!("  -2, --gpu      🚀 GPU Mode - Meant to be used with Intel Arc");
    println!("  -3, --compare  ⚡ Comparison - Both CPU and GPU");
    println!();
    println!("OPTIONS:");
    println!("  --image <path>       Image path (default: test_images/InfinityCastle.jpeg)");
    println!("  --width <pixels>     Image width (default: 400)");
    println!("  --height <pixels>    Image height (default: 400)");
    println!("  --iterations <num>   Training iterations (default: 200)");
    println!("  --help               Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("  gauss-render -2                          # GPU mode with defaults");
    println!("  gauss-render --cpu --iterations 500      # CPU mode with 500 iterations");
    println!("  gauss-render -2 --width 800 --height 600 # GPU mode with custom size");
    println!("  gauss-render --compare --image my.jpg    # Compare modes with custom image");
}

struct Settings {
    image_path: String,
    width: u32,
    height: u32,
    iterations: usize,
}

fn parse_settings_from_args(args: &[String]) -> Settings {
    let mut settings = Settings {
        image_path: "test_images/sample2.jpg".to_string(),
        width: 400,
        height: 400,
        iterations: 200,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                if i + 1 < args.len() {
                    settings.image_path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("❌ --image requires a value");
                    i += 1;
                }
            }
            "--width" => {
                if i + 1 < args.len() {
                    settings.width = args[i + 1].parse().unwrap_or(400);
                    i += 2;
                } else {
                    eprintln!("❌ --width requires a value");
                    i += 1;
                }
            }
            "--height" => {
                if i + 1 < args.len() {
                    settings.height = args[i + 1].parse().unwrap_or(400);
                    i += 2;
                } else {
                    eprintln!("❌ --height requires a value");
                    i += 1;
                }
            }
            "--iterations" => {
                if i + 1 < args.len() {
                    settings.iterations = args[i + 1].parse().unwrap_or(200);
                    i += 2;
                } else {
                    eprintln!("❌ --iterations requires a value");
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    settings
}

fn run_cpu_mode_with_settings(settings: Settings) -> Result<(), Box<dyn std::error::Error>> {
    println!("💻 Starting CPU Mode...");
    println!("📁 Image: {}", settings.image_path);
    println!("📐 Size: {}x{}", settings.width, settings.height);
    println!("🔄 Iterations: {}", settings.iterations);
    
    let start_time = std::time::Instant::now();
    let mut image_gs = ImageGS::new(settings.width, settings.height);
    
    // Test rendering first
    image_gs.initialize_random(20);
    let test_result = image_gs.render();
    test_result.save("cpu_test_output.png")?;
    println!("✅ CPU test render saved as 'cpu_test_output.png'");
    
    // Training
    println!("🎯 Starting CPU training...");
    image_gs.fit_to_image(&settings.image_path, settings.iterations)?;
    
    // Final render
    let final_result = image_gs.render();
    final_result.save("cpu_final_output.png")?;
    
    let duration = start_time.elapsed();
    println!("✅ CPU training complete!");
    println!("⏱️  Total time: {:.2}s", duration.as_secs_f64());
    println!("📊 Final Gaussians: {}", image_gs.gaussian_count());
    println!("💾 Results saved as 'cpu_final_output.png'");
    
    // Evaluate quality metrics
    let target = image::open(&settings.image_path)?.to_rgb8();
    let metrics = evaluate_all_metrics(&target, &final_result);
    metrics.print();
    
    Ok(())
}

async fn run_gpu_mode_with_settings(settings: Settings) -> Result<(), Box<dyn std::error::Error>> {
    println!("📁 Image: {}", settings.image_path);
    println!("📐 Size: {}x{}", settings.width, settings.height);
    println!("🔄 Iterations: {}", settings.iterations);
    
    let start_time = std::time::Instant::now();
    let mut image_gs = ImageGS::new(settings.width, settings.height);
    
    // Create GPU renderer
    println!("🔧 Initializing Intel Arc GPU...");
    let mut gpu_renderer = image_gs::GpuRenderer::new().await?;
    
    // Test rendering first
    image_gs.initialize_random(100);
    let test_result = gpu_renderer.render_gpu(&image_gs).await?;
    test_result.save("gpu_test_output.png")?;
    println!("✅ GPU test render saved as 'gpu_test_output.png'");
    
    // Training
    println!("🎯 Starting GPU-accelerated training...");
    image_gs.fit_to_image_gpu(&settings.image_path, settings.iterations).await?;
    
    // Final render
    let final_result = gpu_renderer.render_gpu(&image_gs).await?;
    final_result.save("test_images/output/gpu_final_output.png")?;
    
    let duration = start_time.elapsed();
    println!("✅ GPU training complete!");
    println!("⏱️  Total time: {:.2}s", duration.as_secs_f64());
    println!("📊 Final Gaussians: {}", image_gs.gaussian_count());
    println!("💾 Results saved as 'gpu_final_output.png'");
    
    // Evaluate quality metrics
    let target = image::open(&settings.image_path)?.to_rgb8();
    let metrics = evaluate_all_metrics(&target, &final_result);
    metrics.print();
    
    Ok(())
}

// Fallback interactive mode (simplified for brevity)
fn run_interactive_mode() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Interactive mode not fully implemented. Use --help for command line options.");
    Ok(())
}
