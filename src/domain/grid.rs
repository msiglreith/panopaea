
#[derive(Copy, Clone, Debug)]
pub struct Grid2d {
    dim: (usize, usize), // (y, x)
}

impl Grid2d {
    pub fn new(dim: (usize, usize)) -> Self {
        Grid2d { dim: dim }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.dim
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Grid3d {
    dim: (usize, usize, usize), // (z, y, x)
}
