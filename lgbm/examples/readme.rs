use lgbm::{
    Booster, Dataset, Field, MatBuf, Parameters, PredictType,
    parameters::{Objective, Verbosity},
};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let mut p = Parameters::new();
    p.push("num_class", 3);
    p.push("objective", Objective::Multiclass);
    p.push("verbosity", Verbosity::Fatal);

    let mut train = Dataset::from_mat(MatBuf::from_rows(train_features()), None, &p)?;
    train.set_field(Field::LABEL, &train_labels())?;

    let mut valid = Dataset::from_mat(MatBuf::from_rows(valid_features()), Some(&train), &p)?;
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
        MatBuf::from_rows(test_features()),
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
