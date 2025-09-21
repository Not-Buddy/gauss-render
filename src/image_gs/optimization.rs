// Add this to your existing optimization.rs file

impl ImageGS {
    /// GPU-accelerated training loop
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
        let iterations_dir = "iterations";
        if std::path::Path::new(iterations_dir).exists() {
            println!("Clearing previous GPU iterations...");
            std::fs::remove_dir_all(iterations_dir).ok();
        }
        std::fs::create_dir_all(iterations_dir)?;

        // Create GPU renderer once
        let renderer = crate::image_gs::gpu_render::GpuRenderer::new().await?;
        
        // Hierarchical initialization with more Gaussians for GPU
        self.smart_initialize(&target);
        println!("Starting GPU optimization with {} initial Gaussians", self.gaussians.len());

        for iter in 0..iterations {
            // Use GPU for rendering!
            let rendered = renderer.render_gpu(self).await?;
            let loss = self.compute_loss(&target, &rendered);

            let learning_rate = if iter < 200 {
                2.0 * (0.9995_f32).powi(iter as i32) // Much slower decay
            } else if iter < 500 {
                0.8 * (0.999_f32).powi((iter - 200) as i32) // Slower decay
            } else if iter < 800 {
                0.3 * (0.9995_f32).powi((iter - 500) as i32) // Fine-tuning
            } else {
                0.1 * (0.9998_f32).powi((iter - 800) as i32) // Ultra-fine
            };

            if iter % 10 == 0 { // More frequent saves with faster GPU
                println!("GPU Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {}", 
                         iter, loss, learning_rate, self.gaussians.len());
                rendered.save(format!("iterations/gpu_progress_{:04}.png", iter))?;
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
}
