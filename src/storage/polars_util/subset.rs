use std::collections::HashMap;

use polars::prelude::*;

use super::ColumnType;

use crate::{
    Normalized, NotNormalized, Size, Storage, Vector, VectorIndex, VectorType,
    architectures::VectorSymbolicArchitecture,
};

/// A subset of a dataframe used to derive vectors for specific rows or the entire dataset.
#[derive(Debug, Clone)]
pub struct Subset<'a, S: Size, V: VectorSymbolicArchitecture> {
    storage: &'a Storage<S, V>,
    vectors: Vec<Option<VectorIndex>>,
    columns: Vec<VectorIndex>,
    width: usize,
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Subset<'a, S, V> {
    pub(crate) fn new(storage: &'a Storage<S, V>, dataframe: &DataFrame) -> Result<Self, Error> {
        let width = dataframe.width();
        let height = dataframe.height();
        let mut vectors = Vec::with_capacity(width * height);
        let mut columns = Vec::with_capacity(width);
        for column in dataframe.columns() {
            let ref_column = storage
                .columns
                .0
                .get(column.name().as_str())
                .ok_or_else(|| Error::ColumnNotFound {
                    name: column.name().to_string(),
                })?;

            match (&ref_column.column_type, column.dtype()) {
                (ColumnType::String, DataType::String) => {
                    for value in column.str().expect("string column") {
                        vectors.push(match value.map(|v| storage.vectors.names.get(v).copied()) {
                            Some(Some(vector)) => Some(vector),
                            None => None, // Skip null values.
                            Some(None) => {
                                return Err(Error::ValueNotFound {
                                    column: column.name().to_string(),
                                    value: value.unwrap_or("").to_string(),
                                });
                            }
                        })
                    }
                }
                (
                    ColumnType::Enum {
                        categories: ref_categories,
                        values: ref_values,
                    },
                    DataType::Enum(categories, ..),
                ) if ensure_same_frozen_categories(ref_categories, categories).is_ok() => {
                    for row in 0..height {
                        vectors.push(match column.get(row)? {
                            AnyValue::Null => None,
                            AnyValue::Enum(index, _) | AnyValue::EnumOwned(index, _) => {
                                Some(ref_values[index as usize])
                            }
                            _ => {
                                return Err(Error::ColumnTypeMismatch {
                                    name: column.name().to_string(),
                                });
                            }
                        })
                    }
                }
                _ => {
                    return Err(Error::ColumnTypeMismatch {
                        name: column.name().to_string(),
                    });
                }
            }

            columns.push(ref_column.vector);
        }

        Ok(Self {
            storage,
            vectors,
            columns,
            width,
        })
    }

    /// Iterate through the rows of the subset, bundling the vectors for each row into a single vector.
    pub fn bundle_rows<T: VectorType<V>, I: IntoSampleIterator>(
        &self,
    ) -> impl Iterator<Item = Option<Vector<S, V, T>>> {
        RowIterator::<_, _, I>::from(self).map(|sample_iter: I::Iterator<'_, '_, S, V>| {
            sample_iter
                .filter_map(|x| x.map(V::denormalize))
                .fold(None, |acc, value| match acc {
                    Some(mut acc) => {
                        self.storage
                            .vectors
                            .vsa
                            .bundle_with_accumulator(&mut acc, &value);
                        Some(acc)
                    }
                    None => Some(value),
                })
                .map(|vector_unnormalized| {
                    let vsa = self.storage.vectors.vsa.clone();
                    Vector::new(vsa, self.storage.vectors.size, vector_unnormalized)
                })
        })
    }

    /// Iterate through the rows of the subset, binding the vectors for each row into a single vector.
    pub fn bind_rows<I: IntoSampleIterator>(
        &self,
    ) -> impl Iterator<Item = Option<Vector<S, V, Normalized<V>>>> {
        RowIterator::<_, _, I>::from(self).map(|sample_iter: I::Iterator<'_, '_, S, V>| {
            sample_iter
                .flatten()
                .fold(None, |acc: Option<V::Storage>, value| match acc {
                    Some(mut acc) => {
                        V::bind(&mut acc, &value);
                        Some(acc)
                    }
                    None => Some(value),
                })
                .map(|vector| {
                    let vsa = self.storage.vectors.vsa.clone();
                    Vector::from_normalized(vsa, self.storage.vectors.size, vector)
                })
        })
    }

    /// Iterate through the rows of the subset, generating combinatorial bundles of the vectors for each row.
    pub fn bundle_rows_combinatorial(
        &self,
        ways: usize,
    ) -> impl Iterator<Item = Option<Vector<S, V, NotNormalized<V>>>> {
        RowIterator::<_, _, SamplesWithColumnVector>::from(self).map(
            move |sample_iter: SampleIteratorWithColumnVector<'_, '_, S, V>| {
                // Flatten cleanly drops the `None` values (missing data slots) yielded
                // by the corrected SampleIterator, natively implementing our missing-data theory.
                let features: Vec<V::Storage> = sample_iter.flatten().collect();

                if features.is_empty() {
                    return None;
                }

                let mut row_accumulator: Option<V::Accumulator> = None;
                let n = features.len();

                // Local helper to recursively generate mathematical combinations of indices
                fn combine(
                    start: usize,
                    n: usize,
                    k: usize,
                    current: &mut Vec<usize>,
                    result: &mut Vec<Vec<usize>>,
                ) {
                    if current.len() == k {
                        result.push(current.clone());
                        return;
                    }
                    for i in start..n {
                        current.push(i);
                        combine(i + 1, n, k, current, result);
                        current.pop();
                    }
                }

                // Construct all k-way combinations up to `ways`
                for k in 1..=ways {
                    if k > n {
                        break; // Cannot generate combinations larger than available valid features
                    }

                    let mut combos = Vec::new();
                    combine(0, n, k, &mut Vec::with_capacity(k), &mut combos);

                    for combo in combos {
                        // Start the binding term with the first feature in the combination
                        let mut bound_term = features[combo[0]].clone();

                        // Bind the remaining features
                        for &idx in &combo[1..] {
                            V::bind(&mut bound_term, &features[idx]);
                        }

                        // Convert into un-normalized state so they can be summed without amplitude loss
                        let term_denorm = V::denormalize(bound_term);

                        // Add into the row's mathematical superposition
                        match &mut row_accumulator {
                            Some(acc) => {
                                self.storage
                                    .vectors
                                    .vsa
                                    .bundle_with_accumulator(acc, &term_denorm);
                            }
                            None => {
                                row_accumulator = Some(term_denorm);
                            }
                        }
                    }
                }

                row_accumulator.map(|acc| {
                    let vsa = self.storage.vectors.vsa.clone();
                    Vector::new(vsa, self.storage.vectors.size, acc)
                })
            },
        )
    }

    /// Bundle the entire dataset into a single vector.
    /// Equivalent expression (I: SamplesWithColumnVector): ((V * C1) + (V * C2)) + ((V * C1) + (V * C2))
    pub fn bundle_dataset<T1: VectorType<V>, T2: VectorType<V>, I: IntoSampleIterator>(
        &self,
    ) -> Option<Vector<S, V, T2>> {
        self.bundle_vectors(self.bundle_rows::<T1, I>())
    }

    /// Bundle combinatorial interactions of the entire dataset into a single vector.
    pub fn bundle_dataset_combinatorial(
        &self,
        ways: usize,
    ) -> Option<Vector<S, V, NotNormalized<V>>> {
        self.bundle_vectors(self.bundle_rows_combinatorial(ways))
    }

    /// Bundle the entire dataset into a single vector, while binding individual-
    /// Equivalent expression (I: SamplesWithColumnVector): ((V * C1) * (V * C2)) + ((V * C1) * (V * C2))
    pub fn bundle_dataset_with_binding<T: VectorType<V>, I: IntoSampleIterator>(
        &self,
    ) -> Option<Vector<S, V, T>> {
        self.bundle_vectors(self.bind_rows::<I>())
    }

    /// Bind the bundled rows of dataset into a single vector.
    /// Equivalent expression (I: SamplesWithColumnVector): ((V * C1) + (V * C2)) * ((V * C1) + (V * C2))
    pub fn bind_dataset<I: IntoSampleIterator>(&self) -> Option<Vector<S, V, Normalized<V>>> {
        self.bind_vectors(self.bundle_rows::<Normalized<V>, I>())
    }

    /// Bind the entire dataset into a single vector.
    /// Equivalent expression (I: SamplesWithColumnVector): ((V * C1) * (V * C2)) * ((V * C1) * (V * C2))
    pub fn bind_entire_dataset<I: IntoSampleIterator>(
        &self,
    ) -> Option<Vector<S, V, Normalized<V>>> {
        self.bind_vectors(self.bind_rows::<I>())
    }

    /// Bundle rows into per-group vectors in a single pass, grouping by an externally supplied key (one entry per row, in the same order as the subset's underlying DataFrame), binding each row's cells together via their column position (equivalent to `SamplesWithPosition`).
    pub fn bundle_groups_temporal<G: Copy + Eq + std::hash::Hash, T: VectorType<V>>(
        &self,
        groups: &[G],
    ) -> Result<HashMap<G, Vector<S, V, T>>, Error> {
        if groups.len() != self.height() {
            return Err(Error::GroupCountMismatch {
                expected: self.height(),
                found: groups.len(),
            });
        }

        let shifted: Vec<Vec<V::Storage>> = (0..self.width)
            .map(|shift| {
                self.storage
                    .vectors
                    .vectors
                    .iter()
                    .map(|v| {
                        let mut data = v.data.0.clone();
                        V::permute(&mut data, shift);
                        data
                    })
                    .collect()
            })
            .collect();

        let mut accumulators: HashMap<G, V::Accumulator> = HashMap::new();

        for (y, &group) in groups.iter().enumerate() {
            let mut row_value: Option<V::Storage> = None;
            for (x, shifted_vector) in shifted.iter().enumerate().take(self.width) {
                let index = self.calculate_index(x, y);
                let Some(vector_index) = self.vectors[index] else {
                    continue;
                };
                let source = &shifted_vector[vector_index.0];

                match row_value.as_mut() {
                    None => {
                        row_value = Some(source.clone());
                    }
                    Some(acc) => {
                        V::bind(acc, source);
                    }
                }
            }

            let Some(row_value) = row_value else {
                continue;
            };
            let denormalized = V::denormalize(row_value);
            match accumulators.entry(group) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    self.storage
                        .vectors
                        .vsa
                        .bundle_with_accumulator(e.get_mut(), &denormalized);
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(denormalized);
                }
            }
        }

        Ok(accumulators
            .into_iter()
            .map(|(g, acc)| {
                let vsa = self.storage.vectors.vsa.clone();
                (g, Vector::new(vsa, self.storage.vectors.size, acc))
            })
            .collect())
    }

    fn bundle_vectors<T1: VectorType<V>, T2: VectorType<V>>(
        &self,
        iterator: impl Iterator<Item = Option<Vector<S, V, T1>>>,
    ) -> Option<Vector<S, V, T2>> {
        iterator
            .flatten()
            .fold(
                None,
                |acc: Option<V::Accumulator>, row_vector: Vector<S, V, T1>| match acc {
                    Some(mut acc) => {
                        let row_vector_denormalized = row_vector.data.into().0;
                        self.storage
                            .vectors
                            .vsa
                            .bundle_with_accumulator(&mut acc, &row_vector_denormalized);
                        Some(acc)
                    }
                    None => Some(row_vector.data.into().0),
                },
            )
            .map(|vector_unnormalized| {
                let vsa = self.storage.vectors.vsa.clone();
                Vector::new(vsa, self.storage.vectors.size, vector_unnormalized)
            })
    }

    fn bind_vectors(
        &self,
        iterator: impl Iterator<Item = Option<Vector<S, V, Normalized<V>>>>,
    ) -> Option<Vector<S, V, Normalized<V>>> {
        iterator
            .flatten()
            .fold(
                None,
                |acc: Option<V::Storage>, row_vector: Vector<S, V, Normalized<V>>| match acc {
                    Some(mut acc) => {
                        V::bind(&mut acc, &row_vector.data.0);
                        Some(acc)
                    }
                    None => Some(row_vector.data.0),
                },
            )
            .map(|vector| {
                let vsa = self.storage.vectors.vsa.clone();
                Vector::from_normalized(vsa, self.storage.vectors.size, vector)
            })
    }

    fn height(&self) -> usize {
        self.vectors.len() / self.width
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Subset<'a, S, V> {
    fn calculate_index(&self, x: usize, y: usize) -> usize {
        y + x * self.height()
    }
}

/// An error that can occur when creating a subset.
#[derive(Debug, Clone)]
pub enum Error {
    /// A new column not previously seen in the storage was found in the dataframe.
    ColumnNotFound {
        /// The name of the column.
        name: String,
    },
    /// A column in the dataframe has an incompatible type with the storage.
    ColumnTypeMismatch {
        /// The name of the column.
        name: String,
    },
    /// A value in the dataframe was not found in the storage.
    ValueNotFound {
        /// The name of the column.
        column: String,
        /// The value that was not found.
        value: String,
    },
    /// The underlying Polars library returned an error.
    PolarsError(PolarsError),
    /// The number of supplied group keys did not match the number of rows in the subset.
    GroupCountMismatch {
        /// The number of rows in the subset.
        expected: usize,
        /// The number of group keys that were supplied.
        found: usize,
    },
}

impl From<PolarsError> for Error {
    fn from(err: PolarsError) -> Self {
        Error::PolarsError(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ColumnNotFound { name } => write!(f, "column '{}' not found", name),
            Error::ColumnTypeMismatch { name } => {
                write!(f, "column '{}' has incompatible type", name)
            }
            Error::ValueNotFound { column, value } => {
                write!(f, "value '{}' not found in column '{}'", value, column)
            }
            Error::PolarsError(err) => write!(f, "polars error: {}", err),
            Error::GroupCountMismatch { expected, found } => write!(
                f,
                "expected {} group keys (one per row), found {}",
                expected, found
            ),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone)]
struct RowIterator<
    'a,
    'b,
    S: Size,
    V: VectorSymbolicArchitecture,
    I: IntoSampleIterator = SamplesWithColumnVector,
> {
    subset: &'a Subset<'b, S, V>,
    y_iter: std::ops::Range<usize>,
    marker: I,
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture, I: IntoSampleIterator>
    From<&'a Subset<'b, S, V>> for RowIterator<'a, 'b, S, V, I>
{
    fn from(subset: &'a Subset<'b, S, V>) -> Self {
        Self {
            subset,
            y_iter: 0..subset.height(),
            marker: I::default(),
        }
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture, I: IntoSampleIterator> Iterator
    for RowIterator<'a, 'b, S, V, I>
{
    type Item = I::Iterator<'a, 'b, S, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let y = self.y_iter.next()?;
        Some(self.marker.for_row(self.subset, y, 0..self.subset.width))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.y_iter.size_hint()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture, I: IntoSampleIterator> ExactSizeIterator
    for RowIterator<'a, 'b, S, V, I>
{
    fn len(&self) -> usize {
        self.y_iter.len()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture, I: IntoSampleIterator> std::iter::FusedIterator
    for RowIterator<'a, 'b, S, V, I>
{
}

/// A trait that allows for the creation of sample iterators for each row of a subset, where the individual cells are generated based on the specific implementation of the trait.
pub trait IntoSampleIterator: Default {
    /// The type of the sample iterator that will be created for each row.
    type Iterator<'a, 'b, S: Size, V: VectorSymbolicArchitecture>: Iterator<Item = Option<V::Storage>>
        + ExactSizeIterator
        + std::iter::FusedIterator
    where
        S: 'a,
        V: 'a,
        S: 'b,
        V: 'b,
        'b: 'a;

    /// Create a new sample iterator for a specific row of the subset, iterating over the specified range of columns.
    fn for_row<'a, 'b, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        subset: &'a Subset<'b, S, V>,
        y: usize,
        x_iter: std::ops::Range<usize>,
    ) -> Self::Iterator<'a, 'b, S, V>;
}

/// Allows iteration over the samples of a subset, yielding the cell bound with the corresponding column vector for each cell in the row.
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplesWithColumnVector;

impl IntoSampleIterator for SamplesWithColumnVector {
    type Iterator<'a, 'b, S: Size, V: VectorSymbolicArchitecture>
        = SampleIteratorWithColumnVector<'a, 'b, S, V>
    where
        S: 'a,
        V: 'a,
        S: 'b,
        V: 'b,
        'b: 'a;

    fn for_row<'a, 'b, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        subset: &'a Subset<'b, S, V>,
        y: usize,
        x_iter: std::ops::Range<usize>,
    ) -> Self::Iterator<'a, 'b, S, V> {
        Self::Iterator { subset, y, x_iter }
    }
}

#[derive(Debug, Clone)]
pub struct SampleIteratorWithColumnVector<'a, 'b, S: Size, V: VectorSymbolicArchitecture> {
    subset: &'a Subset<'b, S, V>,
    y: usize,
    x_iter: std::ops::Range<usize>,
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> Iterator
    for SampleIteratorWithColumnVector<'a, 'b, S, V>
{
    type Item = Option<V::Storage>;

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.x_iter.next()?;
        let index = self.subset.calculate_index(x, self.y);
        let Some(vector_index) = self.subset.vectors[index] else {
            return Some(None);
        };

        let column_vector_index = self.subset.columns[x];
        let mut bound = self.subset.storage.vectors.vectors[column_vector_index.0]
            .data
            .0
            .clone();
        V::bind(
            &mut bound,
            &self.subset.storage.vectors.vectors[vector_index.0].data.0,
        );

        Some(Some(bound))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.x_iter.size_hint()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> ExactSizeIterator
    for SampleIteratorWithColumnVector<'a, 'b, S, V>
{
    fn len(&self) -> usize {
        self.x_iter.len()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> std::iter::FusedIterator
    for SampleIteratorWithColumnVector<'a, 'b, S, V>
{
}

/// Allows iteration over the samples of a subset, yielding the cell value permuted by the column index.
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplesWithPosition;

impl IntoSampleIterator for SamplesWithPosition {
    type Iterator<'a, 'b, S: Size, V: VectorSymbolicArchitecture>
        = SampleIteratorWithPosition<'a, 'b, S, V>
    where
        S: 'a,
        V: 'a,
        S: 'b,
        V: 'b,
        'b: 'a;

    fn for_row<'a, 'b, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        subset: &'a Subset<'b, S, V>,
        y: usize,
        x_iter: std::ops::Range<usize>,
    ) -> Self::Iterator<'a, 'b, S, V> {
        Self::Iterator { subset, y, x_iter }
    }
}

/// Unlike the SampleIterator, this iterator uses only the order of the columns but not their values.
#[derive(Debug, Clone)]
pub struct SampleIteratorWithPosition<'a, 'b, S: Size, V: VectorSymbolicArchitecture> {
    subset: &'a Subset<'b, S, V>,
    y: usize,
    x_iter: std::ops::Range<usize>,
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> Iterator
    for SampleIteratorWithPosition<'a, 'b, S, V>
{
    type Item = Option<V::Storage>;

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.x_iter.next()?;
        let index = self.subset.calculate_index(x, self.y);
        let Some(vector_index) = self.subset.vectors[index] else {
            return Some(None);
        };

        // Permute the vector based on the column index.
        let mut result = self.subset.storage.vectors.vectors[vector_index.0]
            .data
            .0
            .clone();
        V::permute(&mut result, x);

        Some(Some(result))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.x_iter.size_hint()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> ExactSizeIterator
    for SampleIteratorWithPosition<'a, 'b, S, V>
{
    fn len(&self) -> usize {
        self.x_iter.len()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> std::iter::FusedIterator
    for SampleIteratorWithPosition<'a, 'b, S, V>
{
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dollar_dataframe() -> DataFrame {
        let name_dtype = DataType::from_frozen_categories(
            FrozenCategories::new(["USA", "MEX"].into_iter()).unwrap(),
        );
        let money_dtype = DataType::from_frozen_categories(
            FrozenCategories::new(["DOL", "PES"].into_iter()).unwrap(),
        );
        let capital_dtype = DataType::from_frozen_categories(
            FrozenCategories::new(["WDC", "MXC"].into_iter()).unwrap(),
        );

        DataFrame::new(
            2,
            vec![
                Column::new("NAM".into(), &["USA", "MEX"])
                    .cast(&name_dtype)
                    .unwrap(),
                Column::new("MON".into(), &["DOL", "PES"])
                    .cast(&money_dtype)
                    .unwrap(),
                Column::new("CAP".into(), &["WDC", "MXC"])
                    .cast(&capital_dtype)
                    .unwrap(),
            ],
        )
        .expect("valid dataframe")
    }

    #[test]
    fn test_bundle_dataset() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "((NAM * USA) + (CAP * WDC) + (MON * DOL)) + ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset
                .bundle_dataset::<Normalized<crate::architectures::MultiplyAddPermute<u8>>, Normalized<crate::architectures::MultiplyAddPermute<u8>>, SamplesWithColumnVector>()
                .expect("valid dataset vector")
        );
    }

    #[test]
    fn test_bind_dataset() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "((NAM * USA) + (CAP * WDC) + (MON * DOL)) * ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset
                .bind_dataset::<SamplesWithColumnVector>()
                .expect("valid dataset vector")
        );
    }

    #[test]
    fn test_bundle_dataset_with_binding() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "((NAM * USA) * (CAP * WDC) * (MON * DOL)) + ((NAM * MEX) * (CAP * MXC) * (MON * PES))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset
                .bundle_dataset_with_binding::<_, SamplesWithColumnVector>()
                .expect("valid dataset vector")
        );
    }

    #[test]
    fn test_bind_entire_dataset() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "((NAM * USA) * (CAP * WDC) * (MON * DOL)) * ((NAM * MEX) * (CAP * MXC) * (MON * PES))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset
                .bind_entire_dataset::<SamplesWithColumnVector>()
                .expect("valid dataset vector")
        );
    }

    #[test]
    fn test_bind_dataset_temporal() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "(USA * (DOL ^ 1) * (WDC ^ 2)) + (MEX * (PES ^ 1) * (MXC ^ 2))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset
                .bundle_dataset_with_binding::<_, SamplesWithPosition>()
                .expect("valid dataset vector")
        );
    }

    #[test]
    fn test_bundle_groups_with_binding() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");

        let expected_group_0: crate::Expression = "USA * (DOL ^ 1) * (WDC ^ 2)"
            .parse()
            .expect("valid expression");
        let expected_group_1: crate::Expression = "MEX * (PES ^ 1) * (MXC ^ 2)"
            .parse()
            .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        let mut grouped = subset
            .bundle_groups_temporal::<
                usize,
                Normalized<crate::architectures::MultiplyAddPermute<u8>>,
            >(&[0, 1])
            .expect("valid grouping");

        assert_eq!(grouped.len(), 2);
        assert_eq!(
            storage
                .execute(&expected_group_0)
                .expect("valid execution")
                .into_owned(),
            grouped.remove(&0).expect("group 0 present")
        );
        assert_eq!(
            storage
                .execute(&expected_group_1)
                .expect("valid execution")
                .into_owned(),
            grouped.remove(&1).expect("group 1 present")
        );
    }

    #[test]
    fn test_bundle_groups_with_binding_length_mismatch() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let subset = storage.subset(&df).expect("valid subset");

        let result = subset.bundle_groups_temporal::<
            usize,
            Normalized<crate::architectures::MultiplyAddPermute<u8>>,
        >(&[0]);

        assert!(matches!(
            result,
            Err(Error::GroupCountMismatch {
                expected: 2,
                found: 1
            })
        ));
    }
}
