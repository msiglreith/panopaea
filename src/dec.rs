//! Discrete exterior calculus (DEC)

use ndarray::{Array, Ix2, LinalgScalar, Zip};
use sparse::{SparseMatrix};

use std::ops::{Deref, DerefMut, Neg};

pub struct Primal<T>(T);

impl<T> Deref for Primal<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Primal<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

pub struct Dual<T>(T);

impl<T> Deref for Dual<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Dual<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}


pub trait Manifold2d<T> {
    /// Storage type for 0-simplices (vertex).
    /// A differential primal 0-form is stored for each 0-simplex.
    type Simplex0;
    /// Storage type for 1-simplices (edge).
    /// A differential primal 1-form is stored for each 1-simplex.
    type Simplex1;
    /// Storage type for 2-simplices (face).
    /// A differential primal 2-form is stored for each 2-simplex.
    type Simplex2;

    ///
    fn new_simplex_0(&self) -> Self::Simplex0;

    ///
    fn new_simplex_1(&self) -> Self::Simplex1;

    ///
    fn new_simplex_2(&self) -> Self::Simplex2;

    /// Discrete exterior derivative operator for primal 0-forms.
    ///
    /// The operator maps primal 0-forms to primal 1-forms.
    fn derivative_0_primal(&self, &mut Self::Simplex1, &Self::Simplex0);

    fn derivative_0_dual(&self, &mut Self::Simplex1, &Self::Simplex2);

    fn derivative_0_matrix(&self) -> SparseMatrix<T>;

    /// Discrete exterior derivative operator for primal 1-forms.
    ///
    /// The operator maps primal 1-forms to primal 2-forms.
    fn derivative_1_primal(&self, &mut Self::Simplex2, &Self::Simplex1);

    fn derivative_1_dual(&self, &mut Self::Simplex0, &Self::Simplex1);

    fn hodge_0_primal(&self, dual: &mut Self::Simplex0, primal: &Self::Simplex0);
    fn hodge_2_dual(&self, primal: &mut Self::Simplex0, dual: &Self::Simplex0);

    fn hodge_1_primal(&self, dual: &mut Self::Simplex1, primal: &Self::Simplex1);
    fn hodge_1_dual(&self, primal: &mut Self::Simplex1, dual: &Self::Simplex1);

    fn hodge_2_primal(&self, dual: &mut Self::Simplex2, primal: &Self::Simplex2);
    fn hodge_0_dual(&self, primal: &mut Self::Simplex2, dual: &Self::Simplex2);
}

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

impl<T> Manifold2d<T> for Grid2d
    where T: LinalgScalar + Neg<Output = T>
{
    type Simplex0 = Array<T, Ix2>;
    type Simplex1 = (Array<T, Ix2>, Array<T, Ix2>);
    type Simplex2 = Array<T, Ix2>;

    fn new_simplex_0(&self) -> Self::Simplex0 {
        Array::from_elem((self.dim.0 + 1, self.dim.1 + 1), T::zero()) // vertices
    }

    fn new_simplex_1(&self) -> Self::Simplex1 {
        (
            Array::from_elem((self.dim.0 + 1, self.dim.1), T::zero()), // horizontal edges
            Array::from_elem((self.dim.0, self.dim.1 + 1), T::zero()), // vertical edges
        )
    }

    fn new_simplex_2(&self) -> Self::Simplex2 {
        Array::from_elem((self.dim.0, self.dim.1), T::zero()) // faces
    }

    fn derivative_0_matrix(&self) -> SparseMatrix<T> {
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
    
    fn derivative_0_primal(&self, edges: &mut Self::Simplex1, vertices: &Self::Simplex0) {
        Zip::from(&mut edges.0)
            .and(vertices.slice(s![.., ..-1]))
            .and(vertices.slice(s![.., 1..]))
            .apply(|edge, &v0, &v1| {
                *edge = v1 - v0;
            });

        Zip::from(&mut edges.1)
            .and(vertices.slice(s![..-1, ..]))
            .and(vertices.slice(s![1.., ..]))
            .apply(|edge, &v0, &v1| {
                *edge = v1 - v0;
            });
    }

    fn derivative_0_dual(&self, edges: &mut Self::Simplex1, vertices: &Self::Simplex2) {
        // horizontal
        Zip::from(edges.0.slice_mut(s![..1, ..]))
            .and(vertices.slice(s![..1, ..]))
            .apply(|edge, &vertex| {
                *edge = vertex;
            });

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
        Zip::from(faces)
            .and(edges.0.slice(s![..-1, ..]))
            .and(edges.0.slice(s![1.., ..]))
            .and(edges.1.slice(s![.., ..-1]))
            .and(edges.1.slice(s![.., 1..]))
            .apply(|face, &top, &bottom, &left, &right| {
                *face = bottom - top + left - right;
            });
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
        let two = T::one() + T::one();

        // boundary handling: top
        Zip::from(dual.0.slice_mut(s![..1, ..]))
            .and(primal.0.slice(s![..1, ..]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });

        // boundary handling: left
        Zip::from(dual.1.slice_mut(s![.., ..1]))
            .and(primal.1.slice(s![.., ..1]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });

        Zip::from(dual.0.slice_mut(s![1..-1, ..]))
            .and(primal.0.slice(s![1..-1, ..]))
            .apply(|dual, &primal| {
                *dual = primal;
            });

        Zip::from(dual.1.slice_mut(s![.., 1..-1]))
            .and(primal.1.slice(s![.., 1..-1]))
            .apply(|dual, &primal| {
                *dual = primal;
            });

        // boundary handling: bottom
        Zip::from(dual.0.slice_mut(s![-1.., ..]))
            .and(primal.0.slice(s![-1.., ..]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });

        // boundary handling: right
        Zip::from(dual.1.slice_mut(s![.., -1..]))
            .and(primal.1.slice(s![.., -1..]))
            .apply(|dual, &primal| {
                *dual = primal / two;
            });
    }

    fn hodge_1_dual(&self, primal: &mut Self::Simplex1, dual: &Self::Simplex1) {
        // boundary handling: top/left
        Zip::from(primal.0.slice_mut(s![..1, ..]))
            .and(dual.0.slice(s![..1, ..]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = T::zero() - dual * two;
            });

        Zip::from(primal.1.slice_mut(s![.., ..1]))
            .and(dual.1.slice(s![.., ..1]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = -dual * two;
            });

        Zip::from(primal.0.slice_mut(s![1..-1, ..]))
            .and(dual.0.slice(s![1..-1, ..]))
            .apply(|primal, &dual| {
                *primal = -dual;
            });

        Zip::from(primal.1.slice_mut(s![.., 1..-1]))
            .and(dual.1.slice(s![.., 1..-1]))
            .apply(|primal, &dual| {
                *primal = -dual;
            });

        // boundary handling: bottom/right
        Zip::from(primal.0.slice_mut(s![-1.., ..]))
            .and(dual.0.slice(s![-1.., ..]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = -dual * two;
            });

        Zip::from(primal.1.slice_mut(s![.., -1..]))
            .and(dual.1.slice(s![.., -1..]))
            .apply(|primal, &dual| {
                let two = T::one() + T::one();
                *primal = -dual * two;
            });
    }

    fn hodge_2_primal(&self, dual: &mut Self::Simplex2, primal: &Self::Simplex2) {
        dual.assign(primal);
    }

    fn hodge_0_dual(&self, primal: &mut Self::Simplex2, dual: &Self::Simplex2) {
        primal.assign(dual);
    }
}
