// src/main.rs
mod gaussian;
mod image_gs;

use image_gs::ImageGS;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating optimized 2D Gaussian image...");
    
    let mut image_gs = ImageGS::new(200, 200);
    
    // Test with random Gaussians
    image_gs.initialize_random(5);
    let result = image_gs.render();
    result.save("gaussian_output.png")?;
    println!("Rendered {} random Gaussians", image_gs.gaussians.len());
    
    // Fit to target image
    image_gs.fit_to_image("test_images/InfinityCastle.jpeg", 100)?;
    let final_result = image_gs.render();
    final_result.save("test_images/output/InfinityCastleGauss.jpeg")?;
    
    Ok(())
}
