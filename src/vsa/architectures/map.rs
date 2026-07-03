use std::sync::Arc;

use bitvec::prelude::*;
use num_traits::PrimInt;
use rand::SeedableRng;

use super::{
    IntResolution, PrimaryStorage, SelfInverseVectorSymbolicArchitecture, Storage, UIntResolution,
    VectorSymbolicArchitecture,
};

/// Vector storage where each element is either +1 or -1, represented as a bit vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlusMinusOnes<R: UIntResolution = usize>(BitVec<R, Lsb0>);

impl<R: UIntResolution> PlusMinusOnes<R> {
    const POSITIVE: i8 = 1;
    const NEGATIVE: i8 = -1;
}

impl<R: UIntResolution> Storage for PlusMinusOnes<R> {
    type Primitive = i8;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn enforce_constraints(&self, other: &Self) {
        debug_assert_eq!(
            self.0.len(),
            other.0.len(),
            "cannot operate on vectors with different sizes"
        );
        debug_assert!(!self.0.is_empty(), "cannot operate on vectors with size 0");
    }

    fn serialize(&self) -> Vec<f64> {
        self.0.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect()
    }
}

impl<R: UIntResolution> PrimaryStorage for PlusMinusOnes<R> {
    fn random<Rng: rand::Rng>(rng: &mut Rng, size: usize) -> Self {
        Self(BitVec::<R, Lsb0>::random(rng, size))
    }

    fn parse(s: &[f64]) -> Self {
        Self(s.iter().map(|v| *v >= 0.0).collect())
    }
}

impl<R: UIntResolution> std::ops::Index<usize> for PlusMinusOnes<R> {
    type Output = i8;

    fn index(&self, index: usize) -> &Self::Output {
        match self.0[index] {
            true => &Self::POSITIVE,
            false => &Self::NEGATIVE,
        }
    }
}

/// Architecture based upon multiple-add-permute.
#[derive(Debug)]
pub struct MultiplyAddPermute<
    R: UIntResolution = usize,
    RM: IntResolution = isize,
    Rng: rand::Rng = rand::rngs::StdRng,
> {
    resolution: std::marker::PhantomData<fn(RM) -> R>,
    rng: Arc<parking_lot::RwLock<Rng>>,
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng> MultiplyAddPermute<R, RM, Rng> {
    fn bit_to_resolution(bit: bool) -> RM {
        if bit { RM::ONE } else { -RM::ONE }
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng + SeedableRng>
    MultiplyAddPermute<R, RM, Rng>
{
    /// Create a new architecture with a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: Arc::new(parking_lot::RwLock::new(Rng::seed_from_u64(seed))),
        }
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng> Clone
    for MultiplyAddPermute<R, RM, Rng>
{
    fn clone(&self) -> Self {
        Self {
            resolution: std::marker::PhantomData,
            rng: self.rng.clone(),
        }
    }
}

impl<R: UIntResolution, RM: IntResolution, Rng: rand::Rng + SeedableRng> Default
    for MultiplyAddPermute<R, RM, Rng>
{
    fn default() -> Self {
        Self::new(rand::random())
    }
}

impl<R: UIntResolution + PrimInt, RM: IntResolution, Rng: rand::Rng> VectorSymbolicArchitecture
    for MultiplyAddPermute<R, RM, Rng>
{
    type Storage = PlusMinusOnes<R>;
    type Accumulator = Vec<RM>;

    fn normalize(&self, storage: Self::Accumulator) -> Self::Storage {
        // MAP keeps tie votes (0) as +1 to match TorchHD's sign(0) behavior.
        let zero = -RM::ONE + RM::ONE;
        PlusMinusOnes(storage.into_iter().map(|v| v >= zero).collect())
    }

    fn denormalize(storage: Self::Storage) -> Self::Accumulator {
        // Unpack whole words from the bit-packed storage instead of going through a per-bit iterator.
        let len = storage.0.len();
        let mut out = Vec::with_capacity(len);
        let bits_per_word = size_of::<R>() * 8;
        let mut remaining = len;
        for &word in storage.0.as_raw_slice() {
            let take = remaining.min(bits_per_word);
            let mut w = word;
            for _ in 0..take {
                out.push(Self::bit_to_resolution((w & R::one()) == R::one()));
                w = w >> 1;
            }
            remaining -= take;
        }
        out
    }

    fn random(&self, size: usize) -> Self::Storage {
        PlusMinusOnes::random(&mut self.rng.write(), size)
    }

    fn bundle(&self, accumulator: &mut Self::Accumulator, vector: &Self::Storage) {
        accumulator
            .iter_mut()
            .zip(vector.0.iter())
            .for_each(|(acc, bit)| {
                *acc += Self::bit_to_resolution(*bit);
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
        a.enforce_constraints(b);

        // Binding is XNOR, i.e. NOT(XOR). Compute it word-at-a-time via BitVec's own
        // bitwise operators (like `similarity` already does) instead of iterating bit by
        // bit, which goes through a much slower per-bit proxy reference.
        a.0 ^= b.0.as_bitslice();
        a.0 = !std::mem::take(&mut a.0);
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

        a.0.rotate_right(shift);
    }

    fn bind_with_accumulator(a: &mut Self::Accumulator, b: &Self::Storage) {
        debug_assert_eq!(
            a.len(),
            b.len(),
            "cannot operate on vectors with different sizes"
        );
        debug_assert!(!a.is_empty(), "cannot operate on vectors with size 0");
        for (acc, bit) in a.iter_mut().zip(b.0.iter()) {
            if !*bit {
                *acc = -*acc;
            }
        }
    }
    fn similarity(a: &Self::Storage, b: &Self::Storage) -> f64 {
        a.enforce_constraints(b);

        let dim = a.len() as f64;
        let mut diff = a.0.clone();
        diff ^= b.0.as_bitslice();
        let mismatches = diff.count_ones() as f64;
        let dot = dim - 2.0 * mismatches;
        dot / dim
    }

    fn similarity_unnormalized(a: &Self::Accumulator, b: &Self::Accumulator) -> f64 {
        a.enforce_constraints(b);

        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let xf: f64 = x.as_();
            let yf: f64 = y.as_();
            dot += xf * yf;
            norm_a += xf * xf;
            norm_b += yf * yf;
        }

        let magnitude = norm_a.sqrt() * norm_b.sqrt();
        if magnitude == 0.0 {
            0.0
        } else {
            dot / magnitude
        }
    }
}

impl<R: UIntResolution + PrimInt, RM: IntResolution, Rng: rand::Rng>
    SelfInverseVectorSymbolicArchitecture for MultiplyAddPermute<R, RM, Rng>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::architectures::VectorSymbolicArchitecture;

    #[test]
    fn random_returns_expected_size() {
        let map = MultiplyAddPermute::<u8>::new(7);
        let hv = map.random(256);
        assert_eq!(hv.len(), 256);
    }

    #[test]
    fn similarity_unnormalized_matches_expected_values() {
        let a: Vec<isize> = vec![3, 4, 0];
        let b_same: Vec<isize> = vec![3, 4, 0];
        let b_opposite: Vec<isize> = vec![-3, -4, 0];
        let b_orthogonal: Vec<isize> = vec![4, -3, 0];

        assert!(
            (MultiplyAddPermute::<u8>::similarity_unnormalized(&a, &b_same) - 1.0).abs() < 1e-12
        );
        assert!(
            (MultiplyAddPermute::<u8>::similarity_unnormalized(&a, &b_opposite) + 1.0).abs()
                < 1e-12
        );
        assert!(MultiplyAddPermute::<u8>::similarity_unnormalized(&a, &b_orthogonal).abs() < 1e-12);
    }
}
