// src/image_gs/rendering.rs

use super::ImageGS;
use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::Vector2;
use rayon::prelude::*;

impl ImageGS {
    /// Render the Gaussians to an image
    pub fn render(&self) -> RgbImage {
        let mut buffer = vec![vec![0.0f32; 3]; (self.width * self.height) as usize];

        buffer.par_chunks_mut(self.width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel_color) in row.iter_mut().enumerate() {
                    let pixel_pos = Vector2::new(x as f32, y as f32);

                    for gaussian in &self.gaussians {
                        if gaussian.is_relevant(pixel_pos, 3.0) {
                            let weight = gaussian.evaluate_at(pixel_pos);
                            let alpha = weight.min(1.0);

                            for i in 0..3 {
                                pixel_color[i] = alpha * gaussian.color[i] + (1.0 - alpha) * pixel_color[i];
                            }
                        }
                    }
                }
            });

        let mut image = ImageBuffer::new(self.width, self.height);
        for (i, pixel_data) in buffer.iter().enumerate() {
            let x = (i % self.width as usize) as u32;
            let y = (i / self.width as usize) as u32;

            let r = (pixel_data[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (pixel_data[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (pixel_data[2].clamp(0.0, 1.0) * 255.0) as u8;

            image.put_pixel(x, y, Rgb([r, g, b]));
        }

        image
    }
}
