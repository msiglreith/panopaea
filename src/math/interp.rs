
use super::Real;

pub fn linear<S: Real>(a0: S, a1: S, s: S) -> S {
    a0 * (S::one() - s) + a1 * s
}

pub fn bilinear<S: Real>(a00: S, a01: S, a10: S, a11: S, s: S, t: S) -> S {
    linear(linear(a00, a01, s), linear(a10, a11, s), t)
}

pub fn trilinear<S: Real>(
    a000: S, a001: S, a010: S, a011: S,
    a100: S, a101: S, a110: S, a111: S,
    s: S, t: S, u: S
) -> S {
    linear(bilinear(a000, a001, a010, a011, s, t), bilinear(a100, a101, a110, a111, s, t), u)
}
