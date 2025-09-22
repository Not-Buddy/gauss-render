// src/image_gs/implementation/cpu_impl.rs
use super::super::ImageGS;
use std::error::Error;

impl ImageGS {
    /// CPU training loop
    pub fn fit_to_image(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn Error>> {
        let target = image::open(target_path)?.to_rgb8();
        self.width = target.width();
        self.height = target.height();
        
        let iterations_dir = "iterations";
        if std::path::Path::new(iterations_dir).exists() {
            std::fs::remove_dir_all(iterations_dir).ok();
        }
        std::fs::create_dir_all(iterations_dir)?;
        
        self.dense_smart_initialize(&target);
        println!("Starting with {} Gaussians", self.gaussians.len());
        
        for iter in 0..iterations {
            let rendered = self.render();
            let loss = self.compute_loss(&target, &rendered);
            
            let learning_rate = 1.5 * (0.995_f32).powi(iter as i32 / 20);
            
            if iter % 20 == 0 {
                println!("Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {}", 
                         iter, loss, learning_rate, self.gaussians.len());
                rendered.save(format!("iterations/progress_{:04}.png", iter))?;
            }
            
            self.optimize_step_with_lr(&target, &rendered, learning_rate);
            
            if iter % 15 == 0 && iter > 0 {
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
        
        println!("Training complete! Final Gaussians: {}", self.gaussians.len());
        Ok(())
    }
}
