
#[macro_export]
/// Array zip macro (parallel)
/// Based on the ndarray::azip macro but with parallel apply!
macro_rules! azip_par {
    // Final Rule
    (@parse [$a:expr, $($aa:expr,)*] [$($p:pat,)+] in { $($t:tt)* }) => {
        use ::ndarray_parallel::prelude::*;
        ::ndarray::Zip::from($a)
            $(
                .and($aa)
            )*
            .par_apply(|$($p),+| {
                $($t)*
            })
    };
    // parsing stack: [expressions] [patterns] (one per operand)
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident ($e:expr) $($t:tt)*) => {
        azip_par!(@parse [$($exprs)* $e,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident $($t:tt)*) => {
        azip_par!(@parse [$($exprs)* &mut $x,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] , $($t:tt)*) => {
        azip_par!(@parse [$($exprs)*] [$($pats)*] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident ($e:expr) $($t:tt)*) => {
        azip_par!(@parse [$($exprs)* $e,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident $($t:tt)*) => {
        azip_par!(@parse [$($exprs)* &$x,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident ($e:expr) $($t:tt)*) => {
        azip_par!(@parse [$($exprs)* $e,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident $($t:tt)*) => {
        azip_par!(@parse [$($exprs)* &$x,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $($t:tt)*) => { };
    ($($t:tt)*) => {
        azip_par!(@parse [] [] $($t)*);
    }
}

/// Indexed Array zip macro (parallel)
/// Based on the ndarray::azip macro but with parallel apply!
macro_rules! azip_indexed_par {
    // Final Rule
    (@parse_index [$i:pat] [$a:expr, $($aa:expr,)*] [$($p:pat,)+] in { $($t:tt)* }) => {
        use ::ndarray_parallel::prelude::*;
        ::ndarray::Zip::indexed($a)
            $(
                .and($aa)
            )*
            .par_apply(|$i, $($p),+| {
                $($t)*
            })
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] index $($t:tt)*) => {
        azip_indexed_par!(@parse_index_pre [$($exprs)*] [$($pats)*] $($t)*);
    };
    (@parse_index_pre [$($exprs:tt)*] [$($pats:tt)*] $i:ident $($t:tt)*) => {
        azip_indexed_par!(@parse_index [$i] [$($exprs)*] [$($pats)*] $($t)*);
    };

    // parsing stack: [expressions] [patterns] (one per operand)
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident ($e:expr) $($t:tt)*) => {
        azip_indexed_par!(@parse [$($exprs)* $e,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident $($t:tt)*) => {
        azip_indexed_par!(@parse [$($exprs)* &mut $x,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] , $($t:tt)*) => {
        azip_indexed_par!(@parse [$($exprs)*] [$($pats)*] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident ($e:expr) $($t:tt)*) => {
        azip_indexed_par!(@parse [$($exprs)* $e,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident $($t:tt)*) => {
        azip_indexed_par!(@parse [$($exprs)* &$x,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident ($e:expr) $($t:tt)*) => {
        azip_indexed_par!(@parse [$($exprs)* $e,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident $($t:tt)*) => {
        azip_indexed_par!(@parse [$($exprs)* &$x,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $($t:tt)*) => { };
    ($($t:tt)*) => {
        azip_indexed_par!(@parse [] [] $($t)*);
    }
}