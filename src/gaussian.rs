// src/gaussian.rs
use nalgebra::{Vector2, Matrix2, Rotation2};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian2D {
    pub mean: Vector2<f32>,
    pub rotation: f32,
    pub scale: Vector2<f32>,
    pub color: Vec<f32>,
}

impl Gaussian2D {
    pub fn new(mean: Vector2<f32>, rotation: f32, scale: Vector2<f32>, color: Vec<f32>) -> Self {
        Self { mean, rotation, scale, color }
    }

    pub fn covariance_matrix(&self) -> Matrix2<f32> {
        let rot = Rotation2::new(self.rotation);
        let scale_matrix = Matrix2::new(
            self.scale.x, 0.0,
            0.0, self.scale.y
        );
        rot.matrix() * scale_matrix * rot.matrix().transpose()
    }

    pub fn evaluate_at(&self, x: Vector2<f32>) -> f32 {
        let diff = x - self.mean;
        let cov_inv = self.covariance_matrix().try_inverse().unwrap_or(Matrix2::identity());
        let quadratic_form = diff.transpose() * cov_inv * diff;
        let scalar_value = quadratic_form[(0, 0)];
        (-0.5 * scalar_value).exp()
    }
    
    pub fn is_relevant(&self, pixel: Vector2<f32>, threshold: f32) -> bool {
        let dist_sq = (pixel - self.mean).norm_squared();
        let max_scale = self.scale.x.max(self.scale.y);
        dist_sq < (threshold * max_scale).powi(2)
    }
}
