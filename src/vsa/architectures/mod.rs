//! Implementation of different architectures.

mod bsc;
mod hrr;
mod map;
mod vtb;

use bitvec::vec::BitVec;
use num_traits::float::Float;
use num_traits::int::PrimInt;
use num_traits::{AsPrimitive, ConstOne, ConstZero, NumAssign, Unsigned};
use rand::RngExt;

use crate::Size;

pub use self::bsc::BinarySpatterCode;
pub use self::hrr::HolographicReducedRepresentation;
pub use self::map::{MultiplyAddPermute, PlusMinusOnes};
pub use self::vtb::VectorDerivedTransformationBinding;

/// A vector symbolic architecture.
pub trait VectorSymbolicArchitecture: Clone {
    /// The underlying storage of a single vector.
    type Storage: PrimaryStorage;
    /// The underlying storage of a multi vector.
    type Accumulator: Storage;

    /// Checks if a size of a vector is valid for the architecture.
    fn valid_size<S: Size>(_size: S) -> bool {
        true
    }

    /// Create a random vector in the architecture.
    fn random(&self, size: usize) -> Self::Storage;

    /// Normalize a accumulator.
    fn normalize(&self, storage: Self::Accumulator) -> Self::Storage;

    /// Cast a normalized vector to an unnormalized accumulator.
    fn denormalize(storage: Self::Storage) -> Self::Accumulator;

    /// Permute a vector.
    fn permute(a: &mut Self::Storage, shifts: usize);

    /// Calculate a appopiate similarity for the architecture.
    fn similarity(a: &Self::Storage, b: &Self::Storage) -> f64;

    /// Calculate a similarity directly on the un-normalized accumulator.
    fn similarity_unnormalized(a: &Self::Accumulator, b: &Self::Accumulator) -> f64;

    /// Bind two vectors.
    fn bind(a: &mut Self::Storage, b: &Self::Storage);

    /// Bind two vectors.
    fn bind_with_accumulator(a: &mut Self::Accumulator, b: &Self::Storage);

    /// Bundle a an accumulator with a vector.
    fn bundle(&self, accumulator: &mut Self::Accumulator, vector: &Self::Storage);

    /// Bundle a an accumulator with another accumulator.
    fn bundle_with_accumulator(
        &self,
        accumulator: &mut Self::Accumulator,
        vector: &Self::Accumulator,
    );
}

/// A vector symbolic architecture where the bind operation is self-inverse.
pub trait SelfInverseVectorSymbolicArchitecture: VectorSymbolicArchitecture {}

/// A vector symbolic architecture where the bind operation is not self-inverse.
pub trait NonSelfInverseVectorSymbolicArchitecture: VectorSymbolicArchitecture {
    /// Inverse a vector.
    fn inverse(a: &mut Self::Storage);
}

/// A underyling data type.
#[allow(clippy::len_without_is_empty)]
pub trait Storage:
    std::fmt::Debug + Clone + PartialEq + std::ops::Index<usize, Output = Self::Primitive>
{
    /// The underyling primitive type of the storage which can be read.
    type Primitive: std::fmt::Display + Copy + PartialEq + PartialOrd;

    /// The length of the storage.
    fn len(&self) -> usize;

    /// Checks if two storages are compatible.
    fn enforce_constraints(&self, other: &Self);

    /// Serialize the storage to a vector of f64.
    fn serialize(&self) -> Vec<f64>;
}

/// A storage type used for "standard" vector content.
pub trait PrimaryStorage: Storage {
    /// Create random data with the specific size.
    fn random<Rng: rand::Rng>(rng: &mut Rng, size: usize) -> Self;

    /// Parse from numeric data.
    fn parse(s: &[f64]) -> Self;
}

impl<R: UIntResolution> Storage for BitVec<R, bitvec::order::Lsb0> {
    type Primitive = bool;

    fn len(&self) -> usize {
        BitVec::len(self)
    }

    fn enforce_constraints(&self, other: &Self) {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "cannot operate on vectors with different sizes"
        );
        debug_assert!(!self.is_empty(), "cannot operate on vectors with size 0");
    }

    fn serialize(&self) -> Vec<f64> {
        self.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect()
    }
}

impl<R: UIntResolution> PrimaryStorage for BitVec<R, bitvec::order::Lsb0> {
    fn random<Rng: rand::Rng>(rng: &mut Rng, size: usize) -> Self {
        // Bit-by-bit on purpose: batching via `extend_from_bitslice` measured no gain here,
        // since the `u64` source word never type-matches `BitVec<R, _>`'s own store type.
        let mut out = BitVec::with_capacity(size);
        let words = size / 64;

        for _ in 0..words {
            let word: u64 = rng.random();
            for bit in 0..64 {
                out.push(((word >> bit) & 1) == 1);
            }
        }

        let remaining = size % 64;
        if remaining > 0 {
            let word: u64 = rng.random();
            for bit in 0..remaining {
                out.push(((word >> bit) & 1) == 1);
            }
        }

        out
    }

    fn parse(s: &[f64]) -> Self {
        s.iter().map(|v| *v >= 0.5).collect()
    }
}

impl<R: FloatResolution> PrimaryStorage for Vec<R> {
    fn random<Rng: rand::Rng>(rng: &mut Rng, size: usize) -> Self {
        (0..size)
            .map(|_| rng.random_range(-R::ONE..R::ONE))
            .collect()
    }

    fn parse(s: &[f64]) -> Self {
        s.iter().map(|v| R::from(*v).unwrap()).collect()
    }
}

impl<R: Resolution> Storage for Vec<R> {
    type Primitive = R;

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn enforce_constraints(&self, other: &Self) {
        debug_assert_eq!(
            self.len(),
            other.len(),
            "cannot operate on vectors with different sizes"
        );
        debug_assert!(!self.is_empty(), "cannot operate on vectors with size 0");
    }

    fn serialize(&self) -> Vec<f64> {
        self.iter().map(|v| v.as_()).collect()
    }
}

/// A resolution of a data type.
pub trait Resolution:
    std::fmt::Debug
    + std::fmt::Display
    + Copy
    + PartialOrd
    + NumAssign
    + ConstZero
    + ConstOne
    + AsPrimitive<f64>
{
}
impl<T> Resolution for T where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + PartialOrd
        + NumAssign
        + ConstZero
        + ConstOne
        + AsPrimitive<f64>
{
}

/// A resolution of a data type limited to (positive and negative) integers.
pub trait IntResolution: Resolution + PrimInt + std::ops::Neg<Output = Self> {}
impl<T> IntResolution for T where T: Resolution + PrimInt + std::ops::Neg<Output = Self> {}

/// A resolution of a data type limited to floating-point numbers.
pub trait FloatResolution:
    Resolution + Float + rand::distr::uniform::SampleUniform + std::iter::Sum + AsPrimitive<f64>
{
}
impl<T> FloatResolution for T where
    T: Resolution + Float + rand::distr::uniform::SampleUniform + std::iter::Sum + AsPrimitive<f64>
{
}

/// A resolution of a data type limited to unsigned integers.
pub trait UIntResolution: Resolution + Unsigned + bitvec::store::BitStore + Ord + Eq {}
impl<T> UIntResolution for T where T: Resolution + Unsigned + bitvec::store::BitStore + Ord + Eq {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn random_bitvec_has_expected_length_and_is_deterministic_per_seed() {
        // Spans a partial word, a single full word, and several full words plus a
        // remainder, for both a byte-sized and a multi-byte backing store type.
        for size in [0, 3, 8, 64, 100, 10_000] {
            let mut rng_a = rand::rngs::StdRng::seed_from_u64(42);
            let mut rng_b = rand::rngs::StdRng::seed_from_u64(42);

            let a: BitVec<u8, bitvec::order::Lsb0> =
                <BitVec<u8, bitvec::order::Lsb0> as PrimaryStorage>::random(&mut rng_a, size);
            let b: BitVec<u8, bitvec::order::Lsb0> =
                <BitVec<u8, bitvec::order::Lsb0> as PrimaryStorage>::random(&mut rng_b, size);

            assert_eq!(a.len(), size, "wrong length at size {size}");
            assert_eq!(a, b, "same seed must produce the same bits at size {size}");
        }
    }

    #[test]
    fn random_bitvec_is_not_degenerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let bits: BitVec<usize, bitvec::order::Lsb0> =
            <BitVec<usize, bitvec::order::Lsb0> as PrimaryStorage>::random(&mut rng, 10_000);
        let ones = bits.count_ones();
        // A fair coin over 10,000 draws should land nowhere near all-0 or all-1; this only
        // guards against a gross generation bug (e.g. every word coming out as 0), not the
        // exact distribution.
        assert!(
            ones > 3_000 && ones < 7_000,
            "suspiciously biased: {ones} ones out of 10000"
        );
    }
}
