
use math::AsLinearView;
use ndarray::{Array, ArrayView, ArrayViewMut, Ix1, Ix2, LinalgScalar, Zip};
use sparse::{DiagonalMatrix, SparseMatrix};
use std::ops::Neg;
use super::manifold::Manifold2d;

#[derive(Copy, Clone, Debug)]
pub struct Grid2d {
    pub dim: (usize, usize),
}

impl Grid2d {
    pub fn num_vertices(&self) -> usize {
        (self.dim.0 + 1) * (self.dim.1 + 1)
    }

    pub fn num_edges(&self) -> usize {
        (self.dim.0 + 1) * self.dim.1 + self.dim.0 * (self.dim.1 + 1)
    }

    pub fn num_faces(&self) -> usize {
        self.dim.0 * self.dim.1
    }
}

#[derive(Debug)]
pub struct Staggered2d<T> {
    pub data: Array<T, Ix1>,
    dim: (usize, usize),
}

impl<T> Staggered2d<T> {
    pub fn dim(&self) -> (usize, usize) {
        self.dim
    }

    pub fn split(&self) -> (ArrayView<T, Ix2>, ArrayView<T, Ix2>) {
        let dim_y = (self.dim.0 + 1) * self.dim.1;

        (unsafe { ArrayView::<T, Ix2>::from_shape_ptr((self.dim.1, self.dim.0 + 1), self.data.as_ptr()) },
         unsafe { ArrayView::<T, Ix2>::from_shape_ptr((self.dim.1 + 1, self.dim.0), self.data.as_ptr().offset(dim_y as isize)) })
    }

    pub fn split_mut(&mut self) -> (ArrayViewMut<T, Ix2>, ArrayViewMut<T, Ix2>) {
        let dim_y = (self.dim.0 + 1) * self.dim.1;

        (unsafe { ArrayViewMut::<T, Ix2>::from_shape_ptr((self.dim.1, self.dim.0 + 1), self.data.as_mut_ptr()) },
         unsafe { ArrayViewMut::<T, Ix2>::from_shape_ptr((self.dim.1 + 1, self.dim.0), self.data.as_mut_ptr().offset(dim_y as isize)) })
    }
}

impl<T> AsLinearView<T> for Staggered2d<T> {
    fn view_linear(&self) -> ArrayView<T, Ix1> {
        self.data.view()
    }

    fn view_linear_mut(&mut self) -> ArrayViewMut<T, Ix1> {
        self.data.view_mut()
    }
}

impl<T> Manifold2d<T> for Grid2d
    where T: LinalgScalar + Neg<Output = T>
{
    type Simplex0 = Array<T, Ix2>;
    type Simplex1 = Staggered2d<T>;
    type Simplex2 = Array<T, Ix2>;

    fn new_simplex_0(&self) -> Self::Simplex0 {
        Array::from_elem((self.dim.0 + 1, self.dim.1 + 1), T::zero()) // vertices
    }

    fn new_simplex_1(&self) -> Self::Simplex1 {
        Staggered2d {
            data: Array::from_elem((self.dim.0 + 1) * self.dim.1 + self.dim.0 * (self.dim.1 + 1), T::zero()),
            dim: self.dim,
        }
    }

    fn new_simplex_2(&self) -> Self::Simplex2 {
        Array::from_elem((self.dim.0, self.dim.1), T::zero()) // faces
    }
    
    fn derivative_0_primal(&self, edges: &mut Self::Simplex1, vertices: &Self::Simplex0) {
        let mut edges = edges.split_mut();

        azip_par!(
            mut edge (&mut edges.0),
            v0 (vertices.slice(s![.., ..-1])),
            v1 (vertices.slice(s![.., 1..]))
         in { *edge = v1 - v0; });

        azip_par!(
            mut edge (&mut edges.1),
            v0 (vertices.slice(s![.., ..-1])),
            v1 (vertices.slice(s![.., 1..]))
         in { *edge = v1 - v0; });
    }

    fn derivative_0_dual(&self, edges: &mut Self::Simplex1, vertices: &Self::Simplex2) {
        let mut edges = edges.split_mut();

        // horizontal
        azip_par!(
            mut edge (edges.0.slice_mut(s![..1, ..])),
            vertex (vertices.slice(s![..1, ..]))
         in { *edge = vertex; });

        // TODO:

        Zip::from(edges.0.slice_mut(s![-1.., ..]))
            .and(vertices.slice(s![-1.., ..]))
            .apply(|edge, &vertex| {
                *edge = -vertex;
            });

        // vertical
        // TODO:
    }

    fn derivative_1_primal(&self, faces: &mut Self::Simplex2, edges: &Self::Simplex1) {
        let edges = edges.split();

        azip_par!(
            mut face (faces),
            top    (edges.0.slice(s![..-1,   ..])),
            bottom (edges.0.slice(s![ 1..,   ..])),
            left   (edges.1.slice(s![  .., ..-1])),
            right  (edges.1.slice(s![  .., 1..]))
         in { *face = bottom - top + left - right; });
    }

    fn derivative_1_dual(&self, faces: &mut Self::Simplex0, edges: &Self::Simplex1) {
        unimplemented!()
    }

    fn hodge_0_primal(&self, dual: &mut Self::Simplex0, primal: &Self::Simplex0) {
        let two = T::one() + T::one();
        let four = two + two;
        let (h, w) = self.dim;

        // corners
        dual[(0, 0)]     = primal[(0, 0)] / four;
        dual[(0, w-1)]   = primal[(0, w-1)] / four;
        dual[(h-1, 0)]   = primal[(h-1, 0)] / four;
        dual[(h-1, w-1)] = primal[(h-1, w-1)] / four;

        // sides
        Zip::from(dual.slice_mut(s![..1, 1..-1]))
            .and(primal.slice(s![..1, 1..-1]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });

        Zip::from(dual.slice_mut(s![-1.., 1..-1]))
            .and(primal.slice(s![-1.., 1..-1]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });

        Zip::from(dual.slice_mut(s![1..-1, ..1]))
            .and(primal.slice(s![1..-1, ..1]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });

        Zip::from(dual.slice_mut(s![1..-1, -1..]))
            .and(primal.slice(s![1..-1, -1..]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });

        // inner
        Zip::from(dual.slice_mut(s![1..-1, 1..-1]))
            .and(primal.slice(s![1..-1, 1..-1]))
            .apply(|dual, &primal| {
                *dual = primal;
            });
    }

    fn hodge_2_dual(&self, primal: &mut Self::Simplex0, dual: &Self::Simplex0) {
        let two = T::one() + T::one();
        let four = two + two;
        let (h, w) = self.dim;

        // corners
        primal[(0, 0)]     = dual[(0, 0)] * four;
        primal[(0, w-1)]   = dual[(0, w-1)] * four;
        primal[(h-1, 0)]   = dual[(h-1, 0)] * four;
        primal[(h-1, w-1)] = dual[(h-1, w-1)] * four;

        // sides
        Zip::from(primal.slice_mut(s![..1, 1..-1]))
            .and(dual.slice(s![..1, 1..-1]))
            .apply(|primal, &dual| {
                *primal = dual * two;
            });

        Zip::from(primal.slice_mut(s![-1.., 1..-1]))
            .and(dual.slice(s![-1.., 1..-1]))
            .apply(|primal, &dual| {
                *primal = dual * two;
            });

        Zip::from(primal.slice_mut(s![1..-1, ..1]))
            .and(dual.slice(s![1..-1, ..1]))
            .apply(|primal, &dual| {
                *primal = dual * two;
            });

        Zip::from(primal.slice_mut(s![1..-1, -1..]))
            .and(dual.slice(s![1..-1, -1..]))
            .apply(|primal, &dual| {
                *primal = dual * two;
            });

        // inner
        Zip::from(primal.slice_mut(s![1..-1, 1..-1]))
            .and(dual.slice(s![1..-1, 1..-1]))
            .apply(|primal, &dual| {
                *primal = dual;
            });
    }

    fn hodge_1_primal(&self, dual: &mut Self::Simplex1, primal: &Self::Simplex1) {
        let primal = primal.split();
        let mut dual = dual.split_mut();

        Zip::from(&mut dual.0)
            .and(&primal.0)
            .apply(|dual, &primal| {
                *dual = primal;
            });

        Zip::from(&mut dual.1)
            .and(&primal.1)
            .apply(|dual, &primal| {
                *dual = primal;
            });
    }

    fn hodge_1_dual(&self, primal: &mut Self::Simplex1, dual: &Self::Simplex1) {
        let dual = dual.split();
        let mut primal = primal.split_mut();

        Zip::from(&mut primal.0)
            .and(&dual.0)
            .apply(|primal, &dual| {
                *primal = -dual;
            });

        Zip::from(&mut primal.1)
            .and(&dual.1)
            .apply(|primal, &dual| {
                *primal = -dual;
            });
    }

    fn hodge_2_primal(&self, dual: &mut Self::Simplex2, primal: &Self::Simplex2) {
        dual.assign(primal);
    }

    fn hodge_0_dual(&self, primal: &mut Self::Simplex2, dual: &Self::Simplex2) {
        primal.assign(dual);
    }

    fn derivative_0_primal_matrix(&self) -> SparseMatrix<T> {
        let mut matrix = SparseMatrix::<T>::new((self.num_edges(), self.num_vertices()));

        let (h, w) = self.dim;
        let one = T::one();
        let mut idx = 0;

        // horizontal edges     
        for y in 0..(h+1) {
            for x in 0..w {
                let v_idx = y*(w+1) + x;
                matrix.insert((idx, v_idx), -one);
                matrix.insert((idx, v_idx + 1), one);
                idx += 1;
            }
        }

        for y in 0..h {
            for x in 0..(w+1) {
                let v_idx = y*(w+1) + x;
                matrix.insert((idx, v_idx), -one);
                matrix.insert((idx, v_idx + w + 1), one);
                idx += 1;
            }
        }

        matrix
    }

    fn derivative_0_dual_matrix(&self) -> SparseMatrix<T> {
        unimplemented!()
    }

    fn derivative_1_primal_matrix(&self) -> SparseMatrix<T> {
        unimplemented!()
    }
    fn derivative_1_dual_matrix(&self) -> SparseMatrix<T> {
        unimplemented!()
    }

    fn hodge_0_primal_matrix(&self) -> DiagonalMatrix<T> {
        unimplemented!()
    }
    fn hodge_1_primal_matrix(&self) -> DiagonalMatrix<T> {
        unimplemented!()
    }
    fn hodge_2_primal_matrix(&self) -> DiagonalMatrix<T> {
        unimplemented!()
    }

    fn hodge_0_dual_matrix(&self) -> DiagonalMatrix<T> {
        unimplemented!()
    }
    fn hodge_1_dual_matrix(&self) -> DiagonalMatrix<T> {
        unimplemented!()
    }
    fn hodge_2_dual_matrix(&self) -> DiagonalMatrix<T> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn grid_2d_divergence() {
        // TODO: apply d1_primal * h1_dual
    }
}
