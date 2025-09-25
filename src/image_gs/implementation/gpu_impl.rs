// src/image_gs/implementation/gpu_impl.rs

use super::super::ImageGS;
use std::error::Error;

impl ImageGS {
    /// GPU training method - optimized with adaptive rendering and smart caching
    pub async fn fit_to_image_gpu(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn Error>> {
        let target = image::open(target_path)?.to_rgb8();
        self.width = target.width();
        self.height = target.height();

        let iterations_dir = "iterations_gpu";
        if std::path::Path::new(iterations_dir).exists() {
            std::fs::remove_dir_all(iterations_dir).ok();
        }

        std::fs::create_dir_all(iterations_dir)?;

        // Create GPU renderer (with health monitoring)
        let mut renderer = super::super::gpu_render::GpuRenderer::new().await?;

        // Same initialization as CPU
        self.dense_smart_initialize(&target);

        println!("Starting OPTIMIZED GPU training with {} initial Gaussians", self.gaussians.len());

        // SMART CACHING: Store the last rendered image
        let mut cached_rendered: Option<image::RgbImage> = None;

        // Main training loop - OPTIMIZED with adaptive rendering frequency
        for iter in 0..iterations {
            // GPU health check (less frequently for speed)
            if iter % 10 == 0 && !renderer.is_healthy() {
                println!("⚠️  GPU renderer appears unhealthy, attempting recovery...");
                match renderer.attempt_recovery().await {
                    Ok(_) => println!("✅ GPU recovery successful, continuing training"),
                    Err(e) => {
                        eprintln!("❌ GPU recovery failed: {}", e);
                        return Err(format!("GPU device recovery failed at iteration {}: {}", iter, e).into());
                    }
                }
            }

            // ADAPTIVE RENDERING FREQUENCY: Render based on training phase
            let should_render = if iter < 30 {
                // Early phase: render every iteration for fast convergence
                true
            } else if iter < 80 {
                // Mid phase: render every 2 iterations
                iter % 2 == 0
            } else {
                // Late phase: render every 3 iterations for fine-tuning
                iter % 3 == 0
            };

            // Get rendered image (fresh or cached)
            let rendered = if should_render {
                // Render fresh image
                let fresh_rendered = renderer.render_gpu(self).await?;
                cached_rendered = Some(fresh_rendered.clone());
                fresh_rendered
            } else {
                // Use cached image if available, otherwise force a render
                if let Some(ref cached) = cached_rendered {
                    cached.clone()
                } else {
                    // First iteration or cache miss - force render
                    let fresh_rendered = renderer.render_gpu(self).await?;
                    cached_rendered = Some(fresh_rendered.clone());
                    fresh_rendered
                }
            };

            let loss = self.compute_loss(&target, &rendered);

            // ADAPTIVE LEARNING RATE: Higher early, standard late
            let learning_rate = if iter < 40 {
                // Higher learning rate early to compensate for less frequent rendering
                2.0 * (0.99_f32).powi(iter as i32 / 15)
            } else {
                // Standard learning rate for fine-tuning
                1.5 * (0.995_f32).powi((iter - 40) as i32 / 20)
            };

            // Less frequent logging for speed
            if iter % 20 == 0 {
                println!("GPU Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {} [Rendered: {}]",
                         iter, loss, learning_rate, self.gaussians.len(), 
                         if should_render {"FRESH"} else {"cached"});

                // Only save progress when we have a fresh render
                if should_render {
                    rendered.save(format!("iterations_gpu/gpu_progress_{:04}.png", iter))?;
                }
            }

            // Use the rendered image (fresh or cached) for optimization
            self.optimize_step_with_lr(&target, &rendered, learning_rate);

            // OPTIMIZED DENSIFICATION: Adaptive frequency and smart caching
            let densification_freq = if iter < 40 {
                6  // More frequent early on
            } else if iter < 100 {
                8  // Standard frequency
            } else {
                12 // Less frequent in late stage
            };

            if iter % densification_freq == 0 && iter > 0 {
                let before_count = self.gaussians.len();

                // Use the current rendered image for densification
                self.moderate_adaptive_addition(&target, &rendered);

                let added_count = self.gaussians.len() - before_count;
                if added_count > 0 {
                    println!("Added {} Gaussians (freq: {}). Total: {}", 
                            added_count, densification_freq, self.gaussians.len());

                    // Invalidate cache after adding Gaussians (significant change)
                    cached_rendered = None;
                }
            }

            // SAME pruning frequency but with cache invalidation
            if iter % 50 == 0 && iter > 0 {
                let before_count = self.gaussians.len();
                self.prune_ineffective_gaussians();
                let pruned_count = before_count - self.gaussians.len();

                if pruned_count > 0 {
                    println!("Pruned {} Gaussians. Total: {}", pruned_count, self.gaussians.len());
                    // Invalidate cache after pruning (significant change)
                    cached_rendered = None;
                }
            }

            // EARLY STOPPING: Exit if converged well
            if iter > 50 && loss < 0.01 && iter % 10 == 0 {
                // Force a fresh render to verify convergence
                let fresh_rendered = renderer.render_gpu(self).await?;
                let fresh_loss = self.compute_loss(&target, &fresh_rendered);
                if fresh_loss < 0.008 {
                    println!("Early stopping at iteration {} due to excellent convergence (loss: {:.6})", 
                            iter, fresh_loss);
                    break;
                }
            }
        }

        println!("GPU training complete! Final Gaussians: {}", self.gaussians.len());

        // Final health check
        if renderer.is_healthy() {
            println!("✅ GPU renderer remained healthy throughout training");
        } else {
            println!("⚠️  GPU renderer had {} failures during training", renderer.get_failure_count());
        }

        Ok(())
    }
}