// src/image_gs/gpu_render.rs

use super::ImageGS;
use crate::gaussian::Gaussian2D;
use image::{ImageBuffer, Rgb, RgbImage};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct GpuGaussian {
    mean_x: f32,
    mean_y: f32,
    rotation: f32,
    scale_x: f32,
    scale_y: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
}

impl From<&Gaussian2D> for GpuGaussian {
    fn from(g: &Gaussian2D) -> Self {
        Self {
            mean_x: g.mean.x,
            mean_y: g.mean.y,
            rotation: g.rotation,
            scale_x: g.scale.x,
            scale_y: g.scale.y,
            color_r: g.color.get(0).copied().unwrap_or(0.0),
            color_g: g.color.get(1).copied().unwrap_or(0.0),
            color_b: g.color.get(2).copied().unwrap_or(0.0),
        }
    }
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    instance: wgpu::Instance,
    // Store adapter info for recovery
    adapter_info: Option<wgpu::AdapterInfo>,
    // Track operation failures to detect device issues
    consecutive_failures: u32,
}

impl GpuRenderer {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::create_with_fallback().await
    }

    async fn create_with_fallback() -> Result<Self, Box<dyn std::error::Error>> {
        // 1. ENABLE FALLBACK BACKENDS - Try multiple backends in order of preference
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL,
            ..Default::default()
        });

        // Try different power preferences for better compatibility
        let power_preferences = [
            wgpu::PowerPreference::HighPerformance,
            wgpu::PowerPreference::LowPower,
        ];

        let mut adapter = None;
        let mut adapter_info = None;

        for power_pref in power_preferences.iter() {
            if let Some(found_adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: *power_pref,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await {
                let info = found_adapter.get_info();
                println!("Found adapter with {:?}: {:?}", power_pref, info);
                adapter_info = Some(info);
                adapter = Some(found_adapter);
                break;
            }
        }

        // Fallback to any available adapter
        let adapter = match adapter {
            Some(adapter) => adapter,
            None => {
                println!("Trying fallback adapter...");
                instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                }).await.ok_or("No suitable adapter found after trying all fallbacks")?
            }
        };

        if adapter_info.is_none() {
            adapter_info = Some(adapter.get_info());
        }

        println!("Using GPU: {:?}", adapter_info);

        // Create device and pipeline
        let (device, queue, compute_pipeline) = Self::create_device_and_pipeline(&adapter).await?;

        Ok(Self {
            device,
            queue,
            compute_pipeline,
            instance,
            adapter_info,
            consecutive_failures: 0,
        })
    }

    async fn create_device_and_pipeline(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue, wgpu::ComputePipeline), Box<dyn std::error::Error>> {
        // Use more conservative limits to avoid device loss
        let mut limits = wgpu::Limits::default();
        let adapter_limits = adapter.limits();

        // Use 75% of available limits to leave headroom
        limits.max_storage_buffer_binding_size = (adapter_limits.max_storage_buffer_binding_size as f32 * 0.75) as u32;
        limits.max_buffer_size = (adapter_limits.max_buffer_size as f32 * 0.75) as u64;

        println!("Using buffer limits: max_storage_buffer_binding_size={}, max_buffer_size={}", 
                limits.max_storage_buffer_binding_size, limits.max_buffer_size);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Gaussian Renderer Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
        }, None).await.map_err(|e| {
            eprintln!("Failed to create device: {}", e);
            e
        })?;

        // Create compute shader and pipeline
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gaussian Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gaussian_compute.wgsl").into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gaussian Pipeline"),
            layout: None,
            module: &compute_shader,
            entry_point: "main",
        });

        Ok((device, queue, compute_pipeline))
    }

    // Recovery method for device loss
    pub async fn attempt_recovery(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔄 Attempting GPU device recovery...");

        // Try to recreate the adapter with the same instance
        let adapter = self.instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower, // Use more stable preference for recovery
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await;

        match adapter {
            Some(adapter) => {
                let info = adapter.get_info();
                println!("Recovery adapter found: {:?}", info);

                match Self::create_device_and_pipeline(&adapter).await {
                    Ok((device, queue, pipeline)) => {
                        self.device = device;
                        self.queue = queue;
                        self.compute_pipeline = pipeline;
                        self.adapter_info = Some(info);
                        self.consecutive_failures = 0;

                        println!("✅ GPU device recovery successful!");
                        Ok(())
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to create device during recovery: {}", e);
                        self.consecutive_failures += 1;
                        Err(e)
                    }
                }
            }
            None => {
                eprintln!("❌ No adapter found during recovery");
                self.consecutive_failures += 1;
                Err("No adapter available for recovery".into())
            }
        }
    }

    // Check if device seems healthy based on operation history
    fn is_device_healthy(&self) -> bool {
        self.consecutive_failures < 3
    }

    // Mark an operation as failed
    fn mark_operation_failed(&mut self) {
        self.consecutive_failures += 1;
        if self.consecutive_failures == 1 {
            eprintln!("⚠️  GPU operation failed - device may be having issues");
        } else if self.consecutive_failures >= 3 {
            eprintln!("🚨 Multiple GPU failures detected - device likely lost");
        }
    }

    // Mark an operation as successful
    fn mark_operation_success(&mut self) {
        if self.consecutive_failures > 0 {
            println!("✅ GPU operation succeeded - device appears stable");
            self.consecutive_failures = 0;
        }
    }

    pub async fn render_gpu(&mut self, image_gs: &ImageGS) -> Result<RgbImage, Box<dyn std::error::Error>> {
        // Check device health before rendering
        if !self.is_device_healthy() {
            return Err("Device appears unhealthy - recovery recommended".into());
        }

        let width = image_gs.width;
        let height = image_gs.height;
        let total_gaussians = image_gs.gaussians.len();

        if total_gaussians == 0 {
            return Ok(ImageBuffer::new(width, height));
        }

        // Use more conservative batch sizing based on actual device limits
        const GAUSSIAN_SIZE_BYTES: usize = std::mem::size_of::<GpuGaussian>();
        let safe_buffer_limit = (self.device.limits().max_storage_buffer_binding_size as f32 * 0.8) as usize; // 80% of limit
        let max_gaussians_per_batch = safe_buffer_limit / GAUSSIAN_SIZE_BYTES;

        println!("Total Gaussians: {}, Max per batch: {} (buffer limit: {} bytes)", 
                 total_gaussians, max_gaussians_per_batch, safe_buffer_limit);

        let result = if total_gaussians <= max_gaussians_per_batch {
            // Single batch - use original method
            self.render_single_batch(&image_gs.gaussians, width, height).await
        } else {
            // Multi-batch processing
            self.render_multi_batch(&image_gs.gaussians, width, height, max_gaussians_per_batch).await
        };

        match result {
            Ok(image) => {
                self.mark_operation_success();
                Ok(image)
            }
            Err(e) => {
                self.mark_operation_failed();
                Err(e)
            }
        }
    }

    async fn render_multi_batch(&mut self, gaussians: &[Gaussian2D], width: u32, height: u32, max_gaussians_per_batch: usize) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let mut final_image = self.create_blank_image(width, height);
        let mut batch_count = 0;
        let total_gaussians = gaussians.len();

        for batch_start in (0..total_gaussians).step_by(max_gaussians_per_batch) {
            let batch_end = (batch_start + max_gaussians_per_batch).min(total_gaussians);
            let batch_gaussians = &gaussians[batch_start..batch_end];

            batch_count += 1;
            println!("Processing batch {}: {} Gaussians ({}-{})",
                     batch_count, batch_gaussians.len(), batch_start, batch_end - 1);

            // Check device health before each batch
            if !self.is_device_healthy() {
                return Err("Device became unhealthy during batch processing".into());
            }

            // Render this batch
            let batch_result = self.render_single_batch(batch_gaussians, width, height).await?;

            // Blend with final image using additive blending
            self.blend_images_additive(&mut final_image, &batch_result);
        }

        // Normalize the final image (since we're doing additive blending)
        self.normalize_image(&mut final_image, batch_count as f32);
        Ok(final_image)
    }

    // Enhanced version of render_single_batch with better error handling
    async fn render_single_batch(&mut self, gaussians: &[Gaussian2D], width: u32, height: u32) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let num_gaussians = gaussians.len() as u32;

        // Convert Gaussians to GPU format
        let gpu_gaussians: Vec<GpuGaussian> = gaussians.iter().map(|g| g.into()).collect();

        // Validate buffer sizes before creating them
        let gaussian_buffer_size = gpu_gaussians.len() * std::mem::size_of::<GpuGaussian>();
        if gaussian_buffer_size as u64 > self.device.limits().max_buffer_size {
            return Err(format!("Gaussian buffer size {} exceeds device limit {}", 
                              gaussian_buffer_size, self.device.limits().max_buffer_size).into());
        }

        let output_buffer_size = (width * height * 4 * std::mem::size_of::<f32>() as u32) as u64;
        if output_buffer_size > self.device.limits().max_buffer_size {
            return Err(format!("Output buffer size {} exceeds device limit {}", 
                              output_buffer_size, self.device.limits().max_buffer_size).into());
        }

        // Create GPU buffers
        let gaussian_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Buffer"),
            contents: bytemuck::cast_slice(&gpu_gaussians),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[width, height, num_gaussians, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Atomic buffer for performance tracking
        let atomic_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atomic Counter Buffer"),
            size: (std::mem::size_of::<u32>() as u64) * 4, // 4 atomic counters
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Initialize atomic counters to zero
        let initial_counters = [0u32; 4];
        self.queue.write_buffer(&atomic_buffer, 0, bytemuck::cast_slice(&initial_counters));

        let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Atomic Buffer"),
            size: atomic_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gaussian_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: atomic_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gaussian Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 8;
            let workgroups_x = (width + workgroup_size - 1) / workgroup_size;
            let workgroups_y = (height + workgroup_size - 1) / workgroup_size;

            println!("Dispatching workgroups: {}x{}x1", workgroups_x, workgroups_y);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy buffers for reading back results
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        encoder.copy_buffer_to_buffer(&atomic_buffer, 0, &debug_buffer, 0, atomic_buffer.size());

        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffers for reading results
        let debug_slice = debug_buffer.slice(..);
        let staging_slice = staging_buffer.slice(..);

        debug_slice.map_async(wgpu::MapMode::Read, |_| {});
        staging_slice.map_async(wgpu::MapMode::Read, |_| {});

        // Poll device until mapping is complete with timeout protection
        use std::time::{Duration, Instant};
        let start_time = Instant::now();
        let timeout = Duration::from_secs(30);

        loop {
            self.device.poll(wgpu::Maintain::Wait);

            if start_time.elapsed() > timeout {
                eprintln!("⏰ GPU operation timed out after 30 seconds");
                return Err("GPU operation timed out - device may be lost".into());
            }

            // Break when polling completes successfully
            break;
        }

        // Read back atomic counters for debugging
        let atomic_stats = {
            let atomic_data = debug_slice.get_mapped_range();
            let atomic_counters: &[u32] = bytemuck::cast_slice(&atomic_data);
            (atomic_counters[0], atomic_counters[1], atomic_counters[2], atomic_counters[3])
        };

        println!("GPU Stats - Relevant Gaussians: {}, Shared Cache Hits: {}, Processed Pixels: {}, Debug Counter: {}",
                 atomic_stats.0, atomic_stats.1, atomic_stats.2, atomic_stats.3);

        // Read back image results
        let mut image = ImageBuffer::new(width, height);

        {
            let data = staging_slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);

            for y in 0..height {
                for x in 0..width {
                    let idx = ((y * width + x) * 4) as usize;
                    let r = (result[idx].clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (result[idx + 1].clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (result[idx + 2].clamp(0.0, 1.0) * 255.0) as u8;
                    image.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
        }

        // Ensure proper cleanup
        self.device.poll(wgpu::Maintain::Wait);

        Ok(image)
    }

    fn create_blank_image(&self, width: u32, height: u32) -> RgbImage {
        ImageBuffer::from_fn(width, height, |_, _| Rgb([0, 0, 0]))
    }

    fn blend_images_additive(&self, base: &mut RgbImage, overlay: &RgbImage) {
        for (base_pixel, overlay_pixel) in base.pixels_mut().zip(overlay.pixels()) {
            for i in 0..3 {
                let base_val = base_pixel[i] as u16;
                let overlay_val = overlay_pixel[i] as u16;
                base_pixel[i] = (base_val + overlay_val).min(255) as u8;
            }
        }
    }

    fn normalize_image(&self, image: &mut RgbImage, batch_count: f32) {
        if batch_count <= 1.0 {
            return;
        }

        let normalization_factor = 1.0 / batch_count.sqrt(); // Use sqrt to preserve brightness better
        for pixel in image.pixels_mut() {
            for i in 0..3 {
                let normalized = (pixel[i] as f32 * normalization_factor).min(255.0);
                pixel[i] = normalized as u8;
            }
        }
    }

    // Helper method to check if the renderer is healthy
    pub fn is_healthy(&self) -> bool {
        self.is_device_healthy()
    }

    // Get current adapter info
    pub fn get_adapter_info(&self) -> Option<&wgpu::AdapterInfo> {
        self.adapter_info.as_ref()
    }

    // Get failure count for diagnostics
    pub fn get_failure_count(&self) -> u32 {
        self.consecutive_failures
    }
}