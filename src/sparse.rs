
use ndarray::{Array, ArrayView, ArrayViewMut, Dim, Ix1, Ix2, LinalgScalar};
use std::ops::Mul;

pub struct DiagonalMatrix<A> {
    data: Vec<A>,
    dim: usize,
}

/// Sparse matrix in compressed sparse row format
#[derive(Debug)]
pub struct SparseMatrix<A> {
    data: Vec<A>,
    col_indices: Vec<usize>,
    row_indices: Vec<usize>,
    dim: (usize, usize),
}

impl<A> SparseMatrix<A> {
    pub fn new(dim: (usize, usize)) -> Self {
        SparseMatrix {
            data: Vec::new(),
            col_indices: Vec::new(),
            row_indices: vec![0; dim.0 + 1],
            dim: dim,
        }
    }

    pub fn insert(&mut self, index: (usize, usize), elem: A) {
        debug_assert!(index.0 < self.dim.0 && index.1 < self.dim.1,
            "Element index out of bounds");
        let start = self.row_indices[index.0];
        let end = self.row_indices[index.0 + 1];

        match self.col_indices[start..end].binary_search(&index.1) {
            Ok(pos) => {
                // overwrite existing element
                self.data[start + pos] = elem;
            }
            Err(pos) => {
                // insert new element
                self.data.insert(start + pos, elem);
                self.col_indices.insert(start + pos, index.1);
                for i in (index.0 + 1 .. self.row_indices.len()) {
                    self.row_indices[i] += 1;
                }
            }
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
        self.col_indices.reserve(additional);
    }
}

impl<A> SparseMatrix<A> 
    where A: LinalgScalar
{
    pub fn mul_grid_simplex_0(&self, mut b: &mut (Array<A, Ix2>, Array<A, Ix2>), x: &Array<A, Ix2>) {
        let mut b0 = unsafe { ArrayViewMut::<A, Ix1>::from_shape_ptr(b.0.len(), b.0.as_mut_ptr()) };
        let mut b1 = unsafe { ArrayViewMut::<A, Ix1>::from_shape_ptr(b.1.len(), b.1.as_mut_ptr()) };
        let x = unsafe { ArrayView::<A, Ix1>::from_shape_ptr(x.len(), x.as_ptr()) };

        for i in 0 .. b0.len() {
            let start = self.row_indices[i];
            let end = self.row_indices[i + 1];

            b0[i] = self.col_indices[start..end].iter()
                        .enumerate()
                        .fold(A::zero(), |acc, (k, &j)| acc + self.data[start + k] * x[j]);
        }

        for i in b0.len() .. b0.len() + b1.len() {
            let start = self.row_indices[i];
            let end = self.row_indices[i + 1];

            b1[i - b0.len()] = self.col_indices[start..end].iter()
                        .enumerate()
                        .fold(A::zero(), |acc, (k, &j)| acc + self.data[start + k] * x[j]);
        }
    }

    pub fn mul_vec(&self, mut b: ArrayViewMut<A, Ix1>, x: ArrayView<A, Ix1>) {
        for i in 0 .. b.len() {
            let start = self.row_indices[i];
            let end = self.row_indices[i + 1];

            b[i] = self.col_indices[start..end].iter()
                        .enumerate()
                        .fold(A::zero(), |acc, (k, &j)| acc + self.data[start + k] * x[j]);
        }
    }
}

impl<'a, A> Mul for &'a SparseMatrix<A>
    where A: LinalgScalar
{
    type Output = Array<A, Ix2>;
    fn mul(self, rhs: &SparseMatrix<A>) -> Array<A, Ix2> {
        /*
        let mut matrix = Array::<A, Ix2>::new((self.dim.0, rhs.dim.1));

        for row in 0 .. matrix.dim.0 {
            for col in 0 .. matrix.dim.1 {
                let start = self.row_indices[row];
                let end = self.row_indices[row + 1];

                let val = A::zero(); 
                // TODO

                matrix.insert((row, col), val);
            }
        }

        matrix
        */

        unimplemented!()
    }
}
