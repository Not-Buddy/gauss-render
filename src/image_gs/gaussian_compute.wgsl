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
    
    let sx = max(gaussian.scale_x, 0.1);
    let sy = max(gaussian.scale_y, 0.1);
    
    let tx = (cos_r * dx + sin_r * dy) / sx;
    let ty = (-sin_r * dx + cos_r * dy) / sy;
    
    let dist_sq = tx * tx + ty * ty;
    
    return exp(-0.5 * dist_sq);
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
    
    var final_color = vec3<f32>(0.0, 0.0, 0.0);
    var total_weight = 0.0;
    
    for (var i = 0u; i < params.num_gaussians; i = i + 1u) {
        let gaussian = gaussians[i];
        let weight = evaluate_gaussian(gaussian, pixel_x, pixel_y);
        
        if (weight > 0.001) {
            total_weight += weight;
            final_color += vec3<f32>(
                gaussian.color_r * weight,
                gaussian.color_g * weight,
                gaussian.color_b * weight
            );
        }
    }
    
    if (total_weight > 0.001) {
        final_color = final_color / total_weight;
    }
    
    final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0));
    
    output[pixel_idx] = vec4<f32>(final_color, 1.0);
}
