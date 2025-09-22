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
}

impl GpuRenderer {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN, // Intel Arc works best with Vulkan
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.ok_or("Failed to find suitable adapter")?;

        println!("Using GPU: {:?}", adapter.get_info());

        // Request higher limits to handle more Gaussians
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 268435456; // 256MB (double the default)
        limits.max_buffer_size = 536870912; // 512MB
        
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: limits,
        }, None).await?;

        // Load the compute shader
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

        Ok(Self {
            device,
            queue,
            compute_pipeline,
        })
    }

    pub async fn render_gpu(&self, image_gs: &ImageGS) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let width = image_gs.width;
        let height = image_gs.height;
        let total_gaussians = image_gs.gaussians.len();

        if total_gaussians == 0 {
            return Ok(ImageBuffer::new(width, height));
        }

        // Calculate safe batch size (leave some headroom)
        const GAUSSIAN_SIZE_BYTES: usize = std::mem::size_of::<GpuGaussian>();
        const SAFE_BUFFER_LIMIT: usize = 200_000_000; // 200MB - safe margin under 256MB limit
        const MAX_GAUSSIANS_PER_BATCH: usize = SAFE_BUFFER_LIMIT / GAUSSIAN_SIZE_BYTES;

        println!("Total Gaussians: {}, Max per batch: {}", total_gaussians, MAX_GAUSSIANS_PER_BATCH);

        if total_gaussians <= MAX_GAUSSIANS_PER_BATCH {
            // Single batch - use original method
            return self.render_single_batch(&image_gs.gaussians, width, height).await;
        }

        // Multi-batch processing
        let mut final_image = self.create_blank_image(width, height);
        let mut batch_count = 0;

        for batch_start in (0..total_gaussians).step_by(MAX_GAUSSIANS_PER_BATCH) {
            let batch_end = (batch_start + MAX_GAUSSIANS_PER_BATCH).min(total_gaussians);
            let batch_gaussians = &image_gs.gaussians[batch_start..batch_end];
            
            batch_count += 1;
            println!("Processing batch {}: {} Gaussians ({}-{})", 
                     batch_count, batch_gaussians.len(), batch_start, batch_end - 1);

            // Render this batch
            let batch_result = self.render_single_batch(batch_gaussians, width, height).await?;
            
            // Blend with final image using additive blending
            self.blend_images_additive(&mut final_image, &batch_result);
        }

        // Normalize the final image (since we're doing additive blending)
        self.normalize_image(&mut final_image, batch_count as f32);

        Ok(final_image)
    }

    // Original render method for single batch
    async fn render_single_batch(&self, gaussians: &[Gaussian2D], width: u32, height: u32) -> Result<RgbImage, Box<dyn std::error::Error>> {
        let num_gaussians = gaussians.len() as u32;

        // Convert Gaussians to GPU format
        let gpu_gaussians: Vec<GpuGaussian> = gaussians.iter().map(|g| g.into()).collect();

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
            size: (width * height * 4 * std::mem::size_of::<f32>() as u32) as u64,
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

            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy buffers for reading back results
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        encoder.copy_buffer_to_buffer(&atomic_buffer, 0, &debug_buffer, 0, atomic_buffer.size());

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back atomic counters for debugging
        let debug_slice = debug_buffer.slice(..);
        debug_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let atomic_stats = {
            let atomic_data = debug_slice.get_mapped_range();
            let atomic_counters: &[u32] = bytemuck::cast_slice(&atomic_data);
            (atomic_counters[0], atomic_counters[1], atomic_counters[2], atomic_counters[3])
        };

        println!("GPU Stats - Relevant Gaussians: {}, Shared Cache Hits: {}, Processed Pixels: {}, Debug Counter: {}", 
                 atomic_stats.0, atomic_stats.1, atomic_stats.2, atomic_stats.3);

        // Read back image results
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mut image = ImageBuffer::new(width, height);
        {
            let data = buffer_slice.get_mapped_range();
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
}
