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

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await?;

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
    let num_gaussians = image_gs.gaussians.len() as u32;

    if num_gaussians == 0 {
        return Ok(ImageBuffer::new(width, height));
    }

    // Convert Gaussians to GPU format
    let gpu_gaussians: Vec<GpuGaussian> = image_gs.gaussians.iter().map(|g| g.into()).collect();

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

    // ATOMIC BUFFERS - New addition for advanced optimizations
    let atomic_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Atomic Counter Buffer"),
        size: std::mem::size_of::<u32>() as u64 * 4, // 4 atomic counters
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Initialize atomic counters to zero
    let initial_counters: [u32; 4] = [0, 0, 0, 0];
    self.queue.write_buffer(&atomic_buffer, 0, bytemuck::cast_slice(&initial_counters));

    // Debug buffer to read back atomic values (optional)
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

    // Create bind group with atomic buffer
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
                binding: 3, // NEW: Atomic buffer binding
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

    // Read back atomic counters for debugging/optimization FIRST
    let debug_slice = debug_buffer.slice(..);
    debug_slice.map_async(wgpu::MapMode::Read, |_| {});
    self.device.poll(wgpu::Maintain::Wait);

    let atomic_stats = {
        let atomic_data = debug_slice.get_mapped_range();
        let atomic_counters: &[u32] = bytemuck::cast_slice(&atomic_data);
        // Copy the data we need before dropping the mapping
        [atomic_counters[0], atomic_counters[1], atomic_counters[2], atomic_counters[3]]
    }; // atomic_data is dropped here automatically

    // Log atomic counter values for performance analysis
    println!("GPU Stats - Relevant Gaussians: {}, Shared Cache Hits: {}, Processed Pixels: {}, Debug Counter: {}", 
        atomic_stats[0], atomic_stats[1], atomic_stats[2], atomic_stats[3]);

    // Read back image results
    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    self.device.poll(wgpu::Maintain::Wait);

    // Convert back to image - process data in scope where it's valid
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
        // data is automatically dropped here when scope ends
    }

    Ok(image)
}

}