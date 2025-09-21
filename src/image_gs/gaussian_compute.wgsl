// src/image_gs/gaussian_compute.wgsl

struct Gaussian {
    mean_x: f32,
    mean_y: f32,
    rotation: f32,
    scale_x: f32,
    scale_y: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
}

struct Params {
    width: u32,
    height: u32,
    num_gaussians: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> atomic_counters: array<atomic<u32>>;

// SHARED MEMORY OPTIMIZATION WITH PROPER ATOMIC TYPES
const SHARED_GAUSSIANS_COUNT: u32 = 64u;
const TILE_SIZE: u32 = 8u;

// Shared memory variables - using atomic types for thread-safe counters
var<workgroup> shared_gaussians: array<Gaussian, SHARED_GAUSSIANS_COUNT>;
var<workgroup> relevant_gaussians: array<u32, 256>;
var<workgroup> relevant_count: atomic<u32>;      // FIXED: Now atomic
var<workgroup> shared_gaussian_count: atomic<u32>; // FIXED: Now atomic

fn evaluate_gaussian(gaussian: Gaussian, pixel_x: f32, pixel_y: f32) -> f32 {
    let dx = pixel_x - gaussian.mean_x;
    let dy = pixel_y - gaussian.mean_y;
    
    let cos_r = cos(gaussian.rotation);
    let sin_r = sin(gaussian.rotation);
    
    // Prevent tiny scales that cause excessive blur
    let sx = max(gaussian.scale_x, 2.0);
    let sy = max(gaussian.scale_y, 2.0);
    
    let tx = (cos_r * dx + sin_r * dy) / sx;
    let ty = (-sin_r * dx + cos_r * dy) / sy;
    
    let dist_sq = tx * tx + ty * ty;
    
    return exp(-1.5 * dist_sq);
}

fn is_gaussian_relevant(gaussian: Gaussian, pixel_x: f32, pixel_y: f32) -> bool {
    let dx = pixel_x - gaussian.mean_x;
    let dy = pixel_y - gaussian.mean_y;
    let max_scale = max(gaussian.scale_x, gaussian.scale_y);
    let dist = sqrt(dx * dx + dy * dy);
    return dist <= max_scale * 3.0;
}

fn gaussian_affects_tile(gaussian: Gaussian, tile_min_x: f32, tile_min_y: f32, tile_max_x: f32, tile_max_y: f32) -> bool {
    let max_scale = max(gaussian.scale_x, gaussian.scale_y);
    let influence_radius = max_scale * 3.0;
    
    let gauss_min_x = gaussian.mean_x - influence_radius;
    let gauss_max_x = gaussian.mean_x + influence_radius;
    let gauss_min_y = gaussian.mean_y - influence_radius;
    let gauss_max_y = gaussian.mean_y + influence_radius;
    
    return !(gauss_max_x < tile_min_x || gauss_min_x > tile_max_x || 
             gauss_max_y < tile_min_y || gauss_min_y > tile_max_y);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    // PHASE 1: COOPERATIVE LOADING OF SHARED DATA
    let tile_min_x = f32(workgroup_id.x * TILE_SIZE);
    let tile_max_x = f32((workgroup_id.x + 1u) * TILE_SIZE);
    let tile_min_y = f32(workgroup_id.y * TILE_SIZE);
    let tile_max_y = f32((workgroup_id.y + 1u) * TILE_SIZE);
    
    // First thread in workgroup initializes atomic counters
    if (local_index == 0u) {
        atomicStore(&relevant_count, 0u);
        atomicStore(&shared_gaussian_count, 0u);
    }
    
    // Wait for initialization
    workgroupBarrier();
    
    // COOPERATIVE GAUSSIAN FILTERING
    let gaussians_per_thread = (params.num_gaussians + 63u) / 64u;
    let start_idx = local_index * gaussians_per_thread;
    let end_idx = min(start_idx + gaussians_per_thread, params.num_gaussians);
    
    for (var i = start_idx; i < end_idx; i = i + 1u) {
        let gaussian = gaussians[i];
        
        if (gaussian_affects_tile(gaussian, tile_min_x, tile_min_y, tile_max_x, tile_max_y)) {
            // FIXED: Now using proper atomic operations
            let old_count = atomicAdd(&relevant_count, 1u);
            if (old_count < 256u) {
                relevant_gaussians[old_count] = i;
                
                // Also copy most relevant Gaussians to shared memory
                if (old_count < SHARED_GAUSSIANS_COUNT) {
                    let shared_idx = atomicAdd(&shared_gaussian_count, 1u);
                    if (shared_idx < SHARED_GAUSSIANS_COUNT) {
                        shared_gaussians[shared_idx] = gaussian;
                    }
                }
            }
            
            // Update global atomic counters for statistics
            atomicAdd(&atomic_counters[0], 1u); // Total relevant Gaussians
        }
    }
    
    // Wait for all threads to finish cooperative loading
    workgroupBarrier();
    
    // PHASE 2: PIXEL RENDERING WITH SHARED DATA
    let pixel_idx = y * params.width + x;
    let pixel_x = f32(x);
    let pixel_y = f32(y);
    
    var final_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    
    // Load atomic counters (read-only after barrier)
    let shared_count = min(atomicLoad(&shared_gaussian_count), SHARED_GAUSSIANS_COUNT);
    let total_relevant = min(atomicLoad(&relevant_count), 256u);
    
    // First, process Gaussians from shared memory (fastest access)
    for (var i = 0u; i < shared_count; i = i + 1u) {
        let gaussian = shared_gaussians[i];
        
        if (!is_gaussian_relevant(gaussian, pixel_x, pixel_y)) {
            continue;
        }
        
        let weight = evaluate_gaussian(gaussian, pixel_x, pixel_y);
        
        if (weight > 0.01) {
            let alpha = weight * (1.0 - accumulated_alpha);
            accumulated_alpha += alpha;
            
            final_color += alpha * vec3<f32>(
                gaussian.color_r,
                gaussian.color_g,
                gaussian.color_b
            );
            
            // Update cache hit counter
            atomicAdd(&atomic_counters[1], 1u);
            
            if (accumulated_alpha > 0.99) {
                break;
            }
        }
    }
    
    // If not fully opaque, process remaining relevant Gaussians from global memory
    if (accumulated_alpha < 0.99) {
        for (var i = shared_count; i < total_relevant; i = i + 1u) {
            let gaussian_idx = relevant_gaussians[i];
            let gaussian = gaussians[gaussian_idx];
            
            if (!is_gaussian_relevant(gaussian, pixel_x, pixel_y)) {
                continue;
            }
            
            let weight = evaluate_gaussian(gaussian, pixel_x, pixel_y);
            
            if (weight > 0.01) {
                let alpha = weight * (1.0 - accumulated_alpha);
                accumulated_alpha += alpha;
                
                final_color += alpha * vec3<f32>(
                    gaussian.color_r,
                    gaussian.color_g,
                    gaussian.color_b
                );
                
                if (accumulated_alpha > 0.99) {
                    break;
                }
            }
        }
    }
    
    // Count processed pixels
    atomicAdd(&atomic_counters[2], 1u);
    
    final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0));
    output[pixel_idx] = vec4<f32>(final_color, 1.0);
}
