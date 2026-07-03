use std::sync::Arc;

use num_traits::ToPrimitive;
use rand::{RngExt, SeedableRng};

use super::{
    FloatResolution, NonSelfInverseVectorSymbolicArchitecture, Storage, VectorSymbolicArchitecture,
};

/// Architecture based upon Vector-Derived Transformation Binding.
#[derive(Debug)]
pub struct VectorDerivedTransformationBinding<
    R: FloatResolution = f64,
    Rng: rand::Rng = rand::rngs::StdRng,
> {
    marker: std::marker::PhantomData<fn() -> R>,
    rng: Arc<parking_lot::RwLock<Rng>>,
}

impl<R: FloatResolution, Rng: rand::Rng> Clone for VectorDerivedTransformationBinding<R, Rng> {
    fn clone(&self) -> Self {
        Self {
            marker: self.marker,
            rng: self.rng.clone(),
        }
    }
}

impl<R: FloatResolution, Rng: rand::Rng + SeedableRng> VectorDerivedTransformationBinding<R, Rng> {
    /// Create a new architecture with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            marker: std::marker::PhantomData,
            rng: Arc::new(parking_lot::RwLock::new(Rng::seed_from_u64(seed))),
        }
    }
}

impl<R: FloatResolution, Rng: rand::Rng + SeedableRng> Default
    for VectorDerivedTransformationBinding<R, Rng>
{
    fn default() -> Self {
        Self::new(rand::random())
    }
}

impl<R: FloatResolution, Rng: rand::Rng> VectorDerivedTransformationBinding<R, Rng> {
    fn norm(values: &[R]) -> R {
        values.iter().map(|v| v.powi(2)).sum::<R>().sqrt()
    }

    fn bind_values(a: &Vec<R>, b: &Vec<R>) -> Vec<R> {
        a.enforce_constraints(b);

        let sqrt_d = Self::sqrt_dimension(a.len());
        let scale = R::from(sqrt_d.to_f64().unwrap().sqrt()).unwrap();
        let mut out = vec![R::ZERO; a.len()];

        for block in 0..sqrt_d {
            let block_start = block * sqrt_d;
            let x_block = &a[block_start..block_start + sqrt_d];
            let out_block = &mut out[block_start..block_start + sqrt_d];

            for row in 0..sqrt_d {
                let mut sum = R::ZERO;
                for col in 0..sqrt_d {
                    sum += b[row * sqrt_d + col] * x_block[col];
                }
                out_block[row] = scale * sum;
            }
        }

        out
    }

    fn sqrt_dimension(len: usize) -> usize {
        let sqrt_d = (len as f64).sqrt() as usize;
        assert_eq!(
            sqrt_d * sqrt_d,
            len,
            "VTB vectors must have a perfect-square dimensionality"
        );
        sqrt_d
    }
}

impl<R: FloatResolution, Rng: rand::Rng> VectorSymbolicArchitecture
    for VectorDerivedTransformationBinding<R, Rng>
{
    type Storage = Vec<R>;
    type Accumulator = Vec<R>;

    fn valid_size<S: crate::Size>(size: S) -> bool {
        // VTB requires perfect-square dimensionality.
        let len = size.size();
        let sqrt_d = (len as f64).sqrt() as usize;
        sqrt_d * sqrt_d == len
    }

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

    fn normalize(&self, storage: Self::Accumulator) -> Self::Storage {
        let norm = Self::norm(&storage);
        if norm == R::ZERO {
            return storage;
        }

        storage.into_iter().map(|value| value / norm).collect()
    }

    fn denormalize(storage: Self::Storage) -> Self::Accumulator {
        storage
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

impl<R: FloatResolution, Rng: rand::Rng> NonSelfInverseVectorSymbolicArchitecture
    for VectorDerivedTransformationBinding<R, Rng>
{
    fn inverse(a: &mut Self::Storage) {
        let sqrt_d = Self::sqrt_dimension(a.len());
        for row in 0..sqrt_d {
            for col in (row + 1)..sqrt_d {
                a.swap(row * sqrt_d + col, col * sqrt_d + row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::architectures::VectorSymbolicArchitecture;

    #[test]
    fn random_returns_expected_size() {
        let vtb = VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::new(7);
        let hv = vtb.random(256);
        assert_eq!(hv.len(), 256);
    }

    #[test]
    fn bind_matches_reference_values() {
        let a = vec![0.2, -0.1, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
        let b = vec![-0.3, 0.5, -0.2, 0.7, -0.6, 0.1, 0.4, -0.8, 0.9];

        let mut bound = a.clone();
        VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::bind(&mut bound, &b);
        let expected = vec![
            -0.29444863728670917,
            0.3983716857408417,
            0.7447818472546173,
            -0.8487048957087499,
            1.1085125168440815,
            1.905255888325765,
            1.368320137979413,
            -1.8359738560230099,
            -2.9964478970941575,
        ];

        for (actual, expected) in bound.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_transposes_square_view() {
        let mut a = vec![0.2, -0.1, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
        VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::inverse(&mut a);
        assert_eq!(a, vec![0.2, 0.4, -0.7, -0.1, -0.5, 0.8, 0.3, 0.6, -0.9]);
    }

    #[test]
    fn cosine_similarity_matches_expected_values() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b_same = vec![1.0, 2.0, 3.0, 4.0];
        let b_opposite = vec![-1.0, -2.0, -3.0, -4.0];
        let b_orthogonal = vec![2.0, -1.0, 0.0, 0.0];

        assert!(
            (VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::similarity(
                &a, &b_same
            ) - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            (VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::similarity(
                &a,
                &b_opposite
            ) + 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::similarity(
                &a,
                &b_orthogonal
            )
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn similarity_unnormalized_matches_similarity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![-1.0, -2.0, -3.0, -4.0];

        assert_eq!(
            VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::similarity_unnormalized(
                &a, &b
            ),
            VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::similarity(&a, &b)
        );
    }

    #[test]
    #[should_panic(expected = "perfect-square dimensionality")]
    fn non_square_dimension_panics() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let mut a = a;
        VectorDerivedTransformationBinding::<f64, rand::rngs::StdRng>::bind(&mut a, &b);
    }
}
