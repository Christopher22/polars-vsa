use std::sync::Arc;

use rand::{RngExt, SeedableRng};
use rustfft::{FftNum, FftPlanner, num_complex::Complex};

use super::{
    FloatResolution, NonSelfInverseVectorSymbolicArchitecture, Storage, VectorSymbolicArchitecture,
};

/// Architecture based upon Holographic Reduced Representations.
#[derive(Debug)]
pub struct HolographicReducedRepresentation<
    R: FloatResolution = f64,
    Rng: rand::Rng = rand::rngs::StdRng,
> {
    marker: std::marker::PhantomData<fn() -> R>,
    rng: Arc<parking_lot::RwLock<Rng>>,
}

impl<R: FloatResolution, Rng: rand::Rng> Clone for HolographicReducedRepresentation<R, Rng> {
    fn clone(&self) -> Self {
        Self {
            marker: self.marker,
            rng: self.rng.clone(),
        }
    }
}

impl<R: FloatResolution, Rng: rand::Rng + SeedableRng> HolographicReducedRepresentation<R, Rng> {
    /// Create a new architecture with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            marker: std::marker::PhantomData,
            rng: Arc::new(parking_lot::RwLock::new(Rng::seed_from_u64(seed))),
        }
    }
}

impl<R: FloatResolution, Rng: rand::Rng + SeedableRng> Default
    for HolographicReducedRepresentation<R, Rng>
{
    fn default() -> Self {
        Self::new(rand::random())
    }
}

impl<R: FloatResolution, Rng: rand::Rng> HolographicReducedRepresentation<R, Rng> {
    fn norm(values: &[R]) -> R {
        values
            .iter()
            .map(|v| num_traits::pow(*v, 2))
            .sum::<R>()
            .sqrt()
    }
}

impl<R: FloatResolution + FftNum, Rng: rand::Rng> HolographicReducedRepresentation<R, Rng> {
    fn bind_values(a: &Vec<R>, b: &Vec<R>) -> Vec<R> {
        a.enforce_constraints(b);
        let len = a.len();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(len);
        let ifft = planner.plan_fft_inverse(len);

        let mut fa: Vec<Complex<R>> = a.iter().map(|&v| Complex::new(v, R::ZERO)).collect();
        let mut fb: Vec<Complex<R>> = b.iter().map(|&v| Complex::new(v, R::ZERO)).collect();

        fft.process(&mut fa);
        fft.process(&mut fb);
        for (x, y) in fa.iter_mut().zip(fb.iter()) {
            *x *= *y;
        }
        ifft.process(&mut fa);

        // rustfft's inverse transform is unnormalized.
        let scale = R::one() / R::from(len).expect("dimension fits in R");
        fa.into_iter().map(|c| c.re * scale).collect()
    }
}

impl<R: FloatResolution + FftNum, Rng: rand::Rng> VectorSymbolicArchitecture
    for HolographicReducedRepresentation<R, Rng>
{
    type Storage = Vec<R>;
    type Accumulator = Vec<R>;

    fn random(&self, size: usize) -> Self::Storage {
        let mut rng = self.rng.write();
        let mut out: Vec<R> = (0..size)
            .map(|_| rng.random_range(-R::ONE..R::ONE))
            .collect();

        let norm = Self::norm(&out);
        if norm > R::ZERO {
            for value in &mut out {
                *value /= norm;
            }
        }

        out
    }

    fn denormalize(storage: Self::Storage) -> Self::Accumulator {
        storage
    }

    fn normalize(&self, storage: Self::Accumulator) -> Self::Storage {
        let norm = Self::norm(&storage);
        if norm == R::ZERO {
            return storage;
        }

        storage.into_iter().map(|value| value / norm).collect()
    }

    fn bundle(&self, accumulator: &mut Self::Accumulator, vector: &Self::Storage) {
        accumulator
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(acc, value)| {
                *acc += *value;
            })
    }

    fn bundle_with_accumulator(
        &self,
        accumulator: &mut Self::Accumulator,
        vector: &Self::Accumulator,
    ) {
        accumulator
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(acc, value)| {
                *acc += *value;
            })
    }

    fn bind(a: &mut Self::Storage, b: &Self::Storage) {
        *a = Self::bind_values(a, b);
    }

    fn bind_with_accumulator(a: &mut Self::Accumulator, b: &Self::Storage) {
        *a = Self::bind_values(a, b);
    }

    fn permute(a: &mut Self::Storage, shifts: usize) {
        let len = a.len();
        if len == 0 {
            return;
        }

        let shift = shifts % len;
        if shift == 0 {
            return;
        }

        a.rotate_right(shift);
    }

    fn similarity(a: &Self::Storage, b: &Self::Storage) -> f64 {
        a.enforce_constraints(b);

        let dot = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum::<R>();
        let magnitude = Self::norm(a) * Self::norm(b);
        if magnitude == R::ZERO {
            return 0.0;
        }

        (dot / magnitude).as_()
    }

    fn similarity_unnormalized(a: &Self::Accumulator, b: &Self::Accumulator) -> f64 {
        Self::similarity(a, b)
    }
}

impl<R: FloatResolution + FftNum, Rng: rand::Rng> NonSelfInverseVectorSymbolicArchitecture
    for HolographicReducedRepresentation<R, Rng>
{
    fn inverse(a: &mut Self::Storage) {
        if a.is_empty() {
            return;
        }

        a.reverse();
        a.rotate_right(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::architectures::VectorSymbolicArchitecture;

    #[test]
    fn random_returns_expected_size() {
        let hrr = HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::new(7);
        let hv = hrr.random(256);
        assert_eq!(hv.len(), 256);
    }

    #[test]
    fn bind_is_circular_convolution() {
        let a = vec![0.2, -0.4, 0.1, 0.6, -0.3];
        let b = vec![-0.5, 0.7, 0.2, -0.1, 0.4];

        let mut bound = a.clone();
        HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::bind(&mut bound, &b);
        let expected = vec![-0.36, 0.26, -0.02, -0.45, 0.71];

        for (actual, expected) in bound.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_matches_stable_hrr_inverse() {
        let mut a = vec![0.2, -0.4, 0.1, 0.6, -0.3];
        HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::inverse(&mut a);
        assert_eq!(a, vec![0.2, -0.3, 0.6, 0.1, -0.4]);
    }

    #[test]
    fn cosine_similarity_matches_expected_values() {
        let a = vec![1.0, 2.0, 3.0];
        let b_same = vec![1.0, 2.0, 3.0];
        let b_opposite = vec![-1.0, -2.0, -3.0];
        let b_orthogonal = vec![2.0, -1.0, 0.0];

        assert!(
            (HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::similarity(&a, &b_same)
                - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            (HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::similarity(
                &a,
                &b_opposite
            ) + 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::similarity(
                &a,
                &b_orthogonal
            )
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn similarity_unnormalized_matches_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, -1.0, 0.0];

        assert_eq!(
            HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::similarity_unnormalized(
                &a, &b
            ),
            HolographicReducedRepresentation::<f64, rand::rngs::StdRng>::similarity(&a, &b)
        );
    }
}
