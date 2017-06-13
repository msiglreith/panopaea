
//!

use math::{Real, Dim, VectorN};
use particle::Property;

/// Predicted position
pub struct PredPosition<T: Real, N: Dim<T>>(pub VectorN<T, N>);
impl<T: Real, N: Dim<T>> Property for PredPosition<T, N> {
    type Subtype = VectorN<T, N>;
    fn new() -> Self::Subtype {
        VectorN::from_elem(T::zero())
    }
}

/// Lambda (constraints)
pub struct Lambda<T: Real>(pub T);
impl<T: Real> Property for Lambda<T> {
	type Subtype = T;
	fn new() -> Self::Subtype {
		T::zero()
	}
}
