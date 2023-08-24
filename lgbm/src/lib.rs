//! Unofficial Rust bindings for [LightGBM](https://lightgbm.readthedocs.io/en/latest/)

mod booster;
mod dataset;
mod error;
pub mod mat;
pub mod parameters;

pub(crate) mod utils;

pub use booster::*;
pub use dataset::*;
pub use error::*;
pub use mat::MatBuf;
pub use parameters::Parameters;

#[cfg(doctest)]
mod tests {
    #[doc = include_str!("../../README.md")]
    mod readme {}
}
