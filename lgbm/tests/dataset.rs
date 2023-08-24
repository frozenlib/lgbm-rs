use anyhow::Result;
use lgbm::{Dataset, Field, MatBuf, Parameters};
use std::env;

#[test]
fn from_csv() -> Result<()> {
    let csv_path = env::current_dir()?.join("tests/data/data.csv");
    let mut p = Parameters::new();
    p.push("header", true);
    let d = Dataset::from_file(&csv_path, None, &p)?;
    assert_eq!(d.get_num_feature()?, 3);
    assert_eq!(d.get_num_data()?, 2);
    Ok(())
}

#[test]
fn from_mat_f32() {
    let d = Dataset::from_mat(
        &MatBuf::from_rows([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        None,
        &Parameters::new(),
    )
    .unwrap();

    assert_eq!(d.get_num_feature().unwrap(), 3);
    assert_eq!(d.get_num_data().unwrap(), 2);
}

#[test]
fn from_mat_f64() {
    let d = Dataset::from_mat(
        &MatBuf::from_rows([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        None,
        &Parameters::new(),
    )
    .unwrap();

    assert_eq!(d.get_num_feature().unwrap(), 3);
    assert_eq!(d.get_num_data().unwrap(), 2);
}

#[test]
fn set_label() -> Result<()> {
    let mut d = Dataset::from_mat(
        &MatBuf::from_rows([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        None,
        &Parameters::new(),
    )?;

    d.set_field(Field::LABEL, &[0.0, 1.0])?;
    Ok(())
}
