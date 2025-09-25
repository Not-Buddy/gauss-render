// src/image_gs/implementation/gpu_impl.rs

use super::super::ImageGS;
use std::error::Error;

impl ImageGS {
    /// GPU training method with adaptive learning rates
    pub async fn fit_to_image_gpu(&mut self, target_path: &str, iterations: usize) -> Result<(), Box<dyn Error>> {
        let target = image::open(target_path)?.to_rgb8();
        self.width = target.width();
        self.height = target.height();

        let iterations_dir = "iterations_gpu";
        if std::path::Path::new(iterations_dir).exists() {
            std::fs::remove_dir_all(iterations_dir).ok();
        }

        std::fs::create_dir_all(iterations_dir)?;

        // FIXED: Make renderer mutable
        let mut renderer = super::super::gpu_render::GpuRenderer::new().await?;

        self.dense_smart_initialize(&target);

        println!("Starting GPU optimization with {} initial Gaussians", self.gaussians.len());

        // Loss tracking for adaptive scheduling
        let mut loss_history: Vec<f32> = Vec::new();
        let mut prev_loss = f32::MAX;
        let mut stagnation_count = 0;
        let mut best_loss = f32::MAX;

        for iter in 0..iterations {
            // Check renderer health before each iteration
            if !renderer.is_healthy() {
                println!("⚠️  GPU renderer appears unhealthy, attempting recovery...");
                match renderer.attempt_recovery().await {
                    Ok(_) => println!("✅ GPU recovery successful, continuing training"),
                    Err(e) => {
                        eprintln!("❌ GPU recovery failed: {}", e);
                        return Err(format!("GPU device recovery failed at iteration {}: {}", iter, e).into());
                    }
                }
            }

            // FIXED: renderer.render_gpu now takes &mut self
            let rendered = renderer.render_gpu(self).await?;
            let loss = self.compute_loss(&target, &rendered);

            // Track loss history and convergence patterns
            loss_history.push(loss);
            if loss_history.len() > 20 {
                loss_history.remove(0);
            }

            // Detect stagnation and improvement patterns
            let loss_improvement = prev_loss - loss;
            if loss_improvement.abs() < 0.001 {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
            }

            if loss < best_loss {
                best_loss = loss;
            }

            // Adaptive learning rate calculation based on loss behavior
            let loss_momentum = if loss_history.len() >= 5 {
                let recent_avg = loss_history.iter().rev().take(5).sum::<f32>() / 5.0;
                let older_avg = loss_history.iter().take(5).sum::<f32>() / 5.0;
                (older_avg - recent_avg).max(0.0)
            } else {
                0.0
            };

            // Base learning rate with exponential decay
            let base_lr = 1.5 * (0.995_f32).powi(iter as i32 / 20);

            // Adaptive scaling factors based on loss characteristics
            let loss_adaptive_factor = if loss > 0.15 {
                2.5 // High loss - aggressive learning
            } else if loss > 0.08 {
                1.8 // Medium loss - moderate learning
            } else if loss > 0.04 {
                1.2 // Low loss - conservative learning
            } else {
                0.8 // Very low loss - fine-tuning mode
            };

            // Stagnation boost - increase LR when stuck
            let stagnation_boost = if stagnation_count > 15 {
                2.0 + (stagnation_count as f32 - 15.0) * 0.1
            } else if stagnation_count > 5 {
                1.3
            } else {
                1.0
            };

            // Momentum-based adaptation
            let momentum_factor = if loss_momentum > 0.01 {
                1.0 // Good progress - maintain current rates
            } else if loss_momentum > 0.001 {
                1.2 // Slow progress - slight boost
            } else {
                1.5 // No progress - significant boost
            };

            // Final adaptive learning rate
            let learning_rate = base_lr * loss_adaptive_factor * stagnation_boost * momentum_factor;

            if iter % 10 == 0 {
                println!("GPU Iteration {}: Loss = {:.6}, LR = {:.4}, Gaussians = {}",
                         iter, loss, learning_rate, self.gaussians.len());
                println!("  Stagnation = {}, Momentum = {:.4}, Loss_Factor = {:.2}, GPU_Failures = {}",
                         stagnation_count, loss_momentum, loss_adaptive_factor, renderer.get_failure_count());
                rendered.save(format!("iterations_gpu/gpu_progress_{:04}.png", iter))?;
            }

            // Use existing optimization method with adaptive learning rate
            self.optimize_step_with_lr(&target, &rendered, learning_rate);

            // Adaptive densification based on loss and training stage
            let densification_frequency = if loss > 0.08 {
                10 // More frequent when loss is high
            } else if loss > 0.04 {
                15 // Standard frequency
            } else {
                25 // Less frequent when converging
            };

            if iter % densification_frequency == 0 && iter > 0 {
                let before_count = self.gaussians.len();
                self.moderate_adaptive_addition(&target, &rendered);
                let added_count = self.gaussians.len() - before_count;
                if added_count > 0 {
                    println!("Added {} Gaussians (freq: {}). Total: {}",
                             added_count, densification_frequency, self.gaussians.len());
                }
            }

            // Adaptive pruning based on convergence stage
            let pruning_frequency = if loss > 0.05 {
                40 // Less frequent pruning when loss is high
            } else if loss > 0.02 {
                50 // Standard frequency
            } else {
                75 // More frequent pruning when converging
            };

            if iter % pruning_frequency == 0 && iter > 0 {
                let before_count = self.gaussians.len();
                self.prune_ineffective_gaussians();
                let pruned_count = before_count - self.gaussians.len();
                if pruned_count > 0 {
                    println!("Pruned {} ineffective Gaussians (freq: {}). Total: {}",
                             pruned_count, pruning_frequency, self.gaussians.len());
                }
            }

            // Early stopping for very low losses
            if loss < 0.001 && stagnation_count > 30 {
                println!("Early stopping due to convergence at iteration {}", iter);
                break;
            }

            // Reset stagnation if we've boosted learning rate significantly
            if stagnation_boost > 1.5 && loss_improvement > 0.005 {
                stagnation_count = 0;
                println!("Learning rate boost successful, resetting stagnation counter");
            }

            prev_loss = loss;
        }

        println!("GPU optimization complete! Final Gaussians: {}, Best Loss: {:.6}",
                 self.gaussians.len(), best_loss);

        // Final health check
        if renderer.is_healthy() {
            println!("✅ GPU renderer remained healthy throughout training");
        } else {
            println!("⚠️  GPU renderer had {} failures during training", renderer.get_failure_count());
        }

        Ok(())
    }
}