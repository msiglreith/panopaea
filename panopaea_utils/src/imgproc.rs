
pub fn transfer(x: &f64, lower: f64, upper: f64) -> u8 {
    let v = x.min(upper).max(lower);
    ((v - lower) / (upper - lower) * 255.0) as u8
}
