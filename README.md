# LGBM-rs

[![Crates.io](https://img.shields.io/crates/v/lgbm.svg)](https://crates.io/crates/lgbm)
[![Docs.rs](https://docs.rs/lgbm/badge.svg)](https://docs.rs/lgbm/)
[![Actions Status](https://github.com/frozenlib/lgbm-rs/workflows/CI/badge.svg)](https://github.com/frozenlib/lgbm-rs/actions)

Unofficial Rust bindings for [LightGBM](https://lightgbm.readthedocs.io/en/latest/)

## Requirement

### Windows or Linux

1. Install LightGBM and build according to [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html).
2. Set the environment variable `LIGHTGBM_LIB_DIR` to the directory containing the build output (`.dll` and `.lib` on Windows, `.so` on Linux).

### MacOS

1. Run `brew install lightgbm` and install LightGBM on your system.
2. Set the environment variable `LIGHTGBM_LIB_DIR` to the directory containing `lib_lightgbm.dylib`.

```sh
brew install lightgbm
export LIGHTGBM_LIB_DIR=/opt/homebrew/Cellar/lightgbm/4.5.0/lib/
```

## Example

**Cargo.toml**

```toml
[dependencies]
lgbm = "0.0.3"
```

**main.rs**

```rust
use lgbm::{
    parameters::{Objective, Verbosity},
    Booster, Dataset, Field, MatBuf, Parameters, PredictType,
};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let mut p = Parameters::new();
    p.push("num_class", 3);
    p.push("objective", Objective::Multiclass);
    p.push("verbosity", Verbosity::Fatal);

    let mut train = Dataset::from_mat(&MatBuf::from_rows(train_features()), None, &p)?;
    train.set_field(Field::LABEL, &train_labels())?;

    let mut valid = Dataset::from_mat(&MatBuf::from_rows(valid_features()), Some(&train), &p)?;
    valid.set_field(Field::LABEL, &valid_labels())?;

    let mut b = Booster::new(Arc::new(train), &p)?;
    b.add_valid_data(Arc::new(valid))?;
    for _ in 0..100 {
        if b.update_one_iter()? {
            break;
        }
    }
    let p = Parameters::new();
    let rs = b.predict_for_mat(
        &MatBuf::from_rows(test_features()),
        PredictType::Normal,
        0,
        None,
        &p,
    )?;
    println!("\n{rs:.5}");
    Ok(())
}
fn train_features() -> Vec<[f64; 1]> {
    (0..128).map(|x| [(x % 3) as f64]).collect()
}
fn train_labels() -> Vec<f32> {
    (0..128).map(|x| (x % 3) as f32).collect()
}
fn valid_features() -> Vec<[f64; 1]> {
    (0..64).map(|x| [(x % 3) as f64]).collect()
}
fn valid_labels() -> Vec<f32> {
    (0..64).map(|x| (x % 3) as f32).collect()
}
fn test_features() -> Vec<[f64; 1]> {
    (0..4).map(|x| [(x % 3) as f64]).collect()
}
```

**output**

```txt
num_data  : 4
num_class : 3
num_2     : 1

   |    0    |    1    |    2    |
---|---------|---------|---------|
 0 | 0.99998 | 0.00001 | 0.00001 |
 1 | 0.00001 | 0.99998 | 0.00001 |
 2 | 0.00001 | 0.00001 | 0.99998 |
 3 | 0.99998 | 0.00001 | 0.00001 |
```

## Static linking or dynamic linking

The following types of linking are supported.

| os      | static | dynamic |
| ------- | ------ | ------- |
| Windows | ✔      | ✔       |
| Linux   | ✔      | ✔       |
| MacOS   |        | ✔       |

On Windows, if `lib_lightgbm.dll` exists in the directory specified by `LIGHTGBM_LIB_DIR`, it will be dynamically linked. Otherwise, it will be statically linked.

On Linux, if `lib_lightgbm.a` exists in the directory specified by `LIGHTGBM_LIB_DIR`, it is statically linked. Otherwise, it is dynamically linked.

## License

This project is licensed under MIT. See the LICENSE files for details.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you will be under the MIT license, without any additional terms or conditions.
