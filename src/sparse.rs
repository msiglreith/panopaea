
use ndarray::{Array, ArrayView, ArrayViewMut, Ix1, Ix2, LinalgScalar, Zip};
use std::ops::{Index, IndexMut, Mul};

/// Diagonal matrix
pub struct DiagonalMatrix<A> {
    data: Vec<A>,
    dim: usize,
}

impl<A: LinalgScalar> DiagonalMatrix<A> {
    pub fn new(dim: usize) -> Self {
        DiagonalMatrix {
            data: vec![A::zero(); dim],
            dim: dim,
        }
    }
}

impl<A: LinalgScalar> DiagonalMatrix<A> {
    pub fn mul_vec(&self, mut b: ArrayViewMut<A, Ix1>, x: ArrayView<A, Ix1>) {
        let a: ArrayView<A, Ix1> = self.data.as_slice().into();
        azip_par!(mut b, x, a in { *b = x * a });
    }
}

impl<A> Index<usize> for DiagonalMatrix<A> {
    type Output = A;

    fn index(&self, elem: usize) -> &A {
        &self.data[elem]
    }
}

impl<A> IndexMut<usize> for DiagonalMatrix<A> {
    fn index_mut(&mut self, elem: usize) -> &mut A {
        &mut self.data[elem]
    }
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

    pub fn dim(&self) -> (usize, usize) {
        self.dim
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
                for i in index.0 + 1 .. self.row_indices.len() {
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

impl<A> Index<usize> for SparseMatrix<A> {
    type Output = A;
    fn index(&self, index: usize) -> &A {
        &self.data[index]
    }
}

impl<A> IndexMut<usize> for SparseMatrix<A> {
    fn index_mut(&mut self, index: usize) -> &mut A {
        &mut self.data[index]
    }
}

impl<A> Mul for DiagonalMatrix<A>
    where A: LinalgScalar
{
    type Output = DiagonalMatrix<A>;
    fn mul(mut self, rhs: DiagonalMatrix<A>) -> DiagonalMatrix<A> {
        for i in 0..self.dim {
            self[i] = self[i] * rhs[i]
        }

        self
    }
}
