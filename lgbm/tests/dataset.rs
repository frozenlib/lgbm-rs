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
fn from_mats() -> Result<()> {
    let d = Dataset::from_mats(
        [
            MatBuf::from_rows([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            MatBuf::from_rows([[1.0f64, 2.0, 3.0], [1.0f64, 2.0, 3.0], [1.0f64, 2.0, 3.0]]),
        ],
        None,
        &Parameters::new(),
    )?;
    assert_eq!(d.get_num_feature()?, 3);
    assert_eq!(d.get_num_data()?, 5);
    Ok(())
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

#[test]
fn get_feature_names() -> Result<()> {
    let mut d = Dataset::from_mat(
        &MatBuf::from_rows([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        None,
        &Parameters::new(),
    )?;

    let names0 = vec!["aaa", "bb", "c"];
    d.set_feature_names(&names0)?;
    let names1 = d.get_feature_names()?;
    assert_eq!(names0, names1);
    Ok(())
}

#[test]
fn set_feature_names_string() -> Result<()> {
    let mut d = Dataset::from_mat(
        &MatBuf::from_rows([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        None,
        &Parameters::new(),
    )?;

    let names0 = vec!["aaa".to_string(), "bb".to_string(), "c".to_string()];
    d.set_feature_names(&names0)?;
    let names1 = d.get_feature_names()?;
    assert_eq!(names0, names1);
    Ok(())
}
