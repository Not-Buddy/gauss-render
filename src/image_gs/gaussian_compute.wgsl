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

fn evaluate_gaussian(gaussian: Gaussian, pixel_x: f32, pixel_y: f32) -> f32 {
    let dx = pixel_x - gaussian.mean_x;
    let dy = pixel_y - gaussian.mean_y;
    
    let cos_r = cos(gaussian.rotation);
    let sin_r = sin(gaussian.rotation);
    
    // Prevent tiny scales that cause excessive blur
    let sx = max(gaussian.scale_x, 2.0);  // Minimum scale 2.0
    let sy = max(gaussian.scale_y, 2.0);
    
    let tx = (cos_r * dx + sin_r * dy) / sx;
    let ty = (-sin_r * dx + cos_r * dy) / sy;
    
    let dist_sq = tx * tx + ty * ty;
    
    // Sharper falloff for better detail
    return exp(-1.5 * dist_sq);  // Increased from -0.5 to -1.5
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let pixel_idx = y * params.width + x;
    let pixel_x = f32(x);
    let pixel_y = f32(y);
    
    var final_color = vec3<f32>(0.0);
    
    // Improved alpha blending approach
    var accumulated_alpha = 0.0;
    
    for (var i = 0u; i < params.num_gaussians; i = i + 1u) {
        let gaussian = gaussians[i];
        let weight = evaluate_gaussian(gaussian, pixel_x, pixel_y);
        
        // Higher threshold for sharpness
        if (weight > 0.01) {
            let alpha = weight * (1.0 - accumulated_alpha);
            accumulated_alpha += alpha;
            
            final_color += alpha * vec3<f32>(
                gaussian.color_r,
                gaussian.color_g,
                gaussian.color_b
            );
            
            // Early exit when fully opaque
            if (accumulated_alpha > 0.99) {
                break;
            }
        }
    }
    
    // No gamma correction for now - keep linear
    final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0));
    
    output[pixel_idx] = vec4<f32>(final_color, 1.0);
}
