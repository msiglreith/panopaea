
//! Bounded Unfiform Grid

use generic_array::typenum::U2;
use math::{Dim, Real, VectorN};
use std::usize;
use std::ops::Deref;

pub struct BoundedGrid<S: Real, N: Dim<usize> + Dim<(usize, usize)>> {
    num_cells: VectorN<usize, N>,
    cell_size: S,

    cell_ranges: Vec<(usize, usize)>,
}

impl<S> BoundedGrid<S, U2>
    where S: Real
{
    pub fn new(num_cells: VectorN<usize, U2>, cell_size: S) -> Self {
        let ranges = vec![(0, 0); num_cells[0] * num_cells[1]];
        BoundedGrid {
            num_cells: num_cells,
            cell_size: cell_size,
            cell_ranges: ranges,
        }
    }

    pub fn get_key(&self, position: &VectorN<S, U2>) -> usize {
        if let Some((x, y)) = self.get_cell(position) {
            x + y * self.num_cells[0]
        } else {
            usize::MAX
        }
    }

    pub fn get_cell(&self, position: &VectorN<S, U2>) -> Option<(usize, usize)> {
        let x: i64 = (position[0] / self.cell_size).floor().to_i64().unwrap();
        let y: i64 = (position[1] / self.cell_size).floor().to_i64().unwrap();

        if (0 <= x && x < self.num_cells[0] as i64) &&
           (0 <= y && y < self.num_cells[1] as i64) {
            Some((x as usize, y as usize))
        } else {
            None
        }
    }

    /// Reconstruct cell ranges from _sorted_ particle position.
    ///
    /// Ref: "Particle Simulation using CUDA", Green, Simon, 2013
    pub fn construct_ranges(&mut self, positions: &[VectorN<S, U2>]) {
        // reset ranges
        for cell in &mut self.cell_ranges {
            *cell = (0, 0);
        }

        let mut prev = self.get_key(&positions[0]);

        {
            if prev >= self.cell_ranges.len() { return; }
            self.cell_ranges[prev].0 = 0;
        }

        for particle in 1..positions.len() {
            let index = self.get_key(&positions[particle]);

            if index >= self.cell_ranges.len() {
                self.cell_ranges[prev].1 = particle;
                return;
            }

            if prev != index {
                // new cell
                self.cell_ranges[index].0 = particle;
                self.cell_ranges[prev].1 = particle;
            }

            prev = index;
        }

        self.cell_ranges[prev].1 = positions.len();
    }

    pub fn get_range(&self, cell: (usize, usize)) -> (usize, usize) {
        let index = cell.0 + cell.1 * self.num_cells[0];
        self.cell_ranges[index]
    }
}