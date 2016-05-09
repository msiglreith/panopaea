
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, Index, IndexMut};
use std::marker::Copy;
use std::ops::{Deref, DerefMut};
use cgmath;
use std::convert::AsRef;

// TODO: hide data later on
#[derive(Clone)]
pub struct Grid2d<T> {
    pub data: Vec<T>,
    pub dim: (usize, usize),
}

impl<T> Grid2d<T> where T: Copy {
    pub fn new(dim: (usize, usize), data: Vec<T>) -> Grid2d<T> {
        Grid2d {
            data: data,
            dim: dim,
        }
    }

    pub fn fill_zero(&mut self) where T: cgmath::Zero {
        for i in 0 .. self.data.len() {
            self.data[i] = T::zero();
        }
    }

    pub fn assign(&mut self, src: Grid2d<T>) {
        self.data = src.data;
    }

    pub fn copy(&mut self, src: &Grid2d<T>) {
        for i in 0..self.data.len() {
            self.data[i] = src.data[i];
        }
    }

    pub fn view<'a>(&'a self) -> &'a Grid2d<T> {
        self
    }

    pub fn view_mut<'a>(&'a mut self) -> &'a mut Grid2d<T> {
        self
    }

    pub fn get_index(&self, _index: (usize, usize)) -> usize {
        debug_assert!(_index < self.dim, "{:?} < {:?}", _index, self.dim);
        (_index.1 * self.dim.0) + _index.0
    }
}

impl Grid2d<f64> {
    pub fn max_norm(&self) -> f64 {
        self.data.iter().map(|x| f64::abs(*x)).fold(0.0f64, |max, x| f64::max(max, x))
    }

    pub fn dot(&self, rhs: &Grid2d<f64>) -> f64 {
        self.data.iter().zip(rhs.data.iter()).fold(0.0f64, |acc, (&a, &b)| acc + a*b)
    }
}

impl<T> Index<(usize, usize)> for Grid2d<T> where T: Copy {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        let idx = self.get_index(idx);
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for Grid2d<T> where T: Copy {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        let idx = self.get_index(idx);
        &mut self.data[idx]
    }
}

impl<T> Index<usize> for Grid2d<T> where T: Copy {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<T> IndexMut<usize> for Grid2d<T> where T: Copy {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.data[idx]
    }
}

pub struct MacGrid2d<T> where T: cgmath::Zero + Copy {
    pub x: Grid2d<T>,
    pub y: Grid2d<T>,
    pub dim: (usize, usize),
}

impl<T> MacGrid2d<T> where T: cgmath::Zero + Copy {
    pub fn new(dim_x: usize, dim_y: usize) -> MacGrid2d<T> {
        MacGrid2d {
            x: Grid2d::new((dim_x + 1, dim_y), vec![T::zero(); (dim_x + 1) * dim_y]),
            y: Grid2d::new((dim_x, dim_y + 1), vec![T::zero(); dim_x * (dim_y + 1)]),
            dim: (dim_x, dim_y),
        }
    }

    pub fn copy(&mut self, src: &MacGrid2d<T>) {
        self.x.copy(&src.x);
        self.y.copy(&src.y);
    }
}
