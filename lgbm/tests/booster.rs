use anyhow::Result;
use lgbm::{
    mat::RowMajor,
    parameters::{Boosting, Metric, Objective, Verbosity},
    Booster, Dataset, FeatureImportanceType, Field, MatBuf, Parameters, PredictType,
};
use std::sync::Arc;

#[test]
fn booster_new() -> Result<()> {
    let train_data = Dataset::from_mat(
        &MatBuf::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        None,
        &parameters(),
    )?;
    let _b = Booster::new(Arc::new(train_data), &parameters())?;
    Ok(())
}

#[test]
fn binary_classification() -> Result<()> {
    let num_class = 2;

    let mut p = parameters();
    p.push("boosting_type", Boosting::Gbdt);
    p.push("objective", Objective::Binary);
    p.push("metric", [Metric::BinaryLogloss, Metric::Auc]);
    p.push("min_data_in_leaf", 20);

    println!("make train dataset");
    let train_feature = make_features(128, num_class);
    let train_label = make_labels(128, num_class);
    let mut train = Dataset::from_mat(&train_feature, None, &p)?;
    train.set_field(Field::LABEL, &train_label)?;

    println!("make test dataset");
    let test_feature = make_features(4, num_class);
    let test_label = make_labels(4, num_class);
    let mut test = Dataset::from_mat(&test_feature, Some(&train), &p)?;
    test.set_field(Field::LABEL, &test_label)?;

    println!("crate booster");
    let mut b = Booster::new(Arc::new(train), &p)?;
    b.add_valid_data(Arc::new(test))?;
    for n in 0..100 {
        println!("iter {n}");
        let is_finish = b.update_one_iter()?;
        let eval_names = b.get_eval_names()?;
        let evals = b.get_eval(0)?;
        for i in 0..eval_names.len() {
            println!("training {}: {}", eval_names[i], evals[i]);
        }
        let evals = b.get_eval(1)?;
        for i in 0..eval_names.len() {
            println!("valid    {}: {}", eval_names[i], evals[i]);
        }
        if is_finish {
            break;
        }
    }

    let p = Parameters::new();
    let rs = b.predict_for_mat(&test_feature, PredictType::Normal, 0, None, &p)?;
    println!("\n{rs}");
    for i in 0..test_label.len() {
        let r = rs[i];
        if test_label[i] == 0.0 {
            assert!(r < 0.1);
        } else {
            assert!(r > 0.9);
        }
    }
    Ok(())
}

#[test]
fn binary_classification_categorical() -> Result<()> {
    let num_category = 3;

    let mut p = Parameters::new();
    p.push("boosting_type", Boosting::Gbdt);
    p.push("objective", Objective::Binary);
    p.push("metric", [Metric::BinaryLogloss, Metric::Auc]);
    p.push("min_data_in_leaf", 20);
    p.push("verbosity", Verbosity::Fatal);
    p.push("categorical_feature", [0]);

    println!("make train dataset");
    let train_feature = make_features_categorycal(128, num_category);
    let train_label = make_labels_categorycal(128, num_category);
    let mut train = Dataset::from_mat(&train_feature, None, &p)?;
    train.set_field(Field::LABEL, &train_label)?;

    println!("make test dataset");
    let test_feature = make_features_categorycal(50, num_category);
    let test_label = make_labels_categorycal(50, num_category);
    let mut test = Dataset::from_mat(&test_feature, Some(&train), &p)?;
    test.set_field(Field::LABEL, &test_label)?;

    println!("crate booster");
    let mut b = Booster::new(Arc::new(train), &p)?;
    b.add_valid_data(Arc::new(test))?;
    for n in 0..100 {
        println!("iter {n}");
        let is_finish = b.update_one_iter()?;
        let eval_names = b.get_eval_names()?;
        let evals = b.get_eval(0)?;
        for i in 0..eval_names.len() {
            println!("training {}: {}", eval_names[i], evals[i]);
        }
        let evals = b.get_eval(1)?;
        for i in 0..eval_names.len() {
            println!("valid    {}: {}", eval_names[i], evals[i]);
        }
        if is_finish {
            break;
        }
    }

    let p = Parameters::new();
    let rs = b.predict_for_mat(&test_feature, PredictType::Normal, 0, None, &p)?;
    println!("\n{rs}");
    for i in 0..test_label.len() {
        let r = rs[i];
        if test_label[i] == 0.0 {
            assert!(r < 0.1);
        } else {
            assert!(r > 0.9);
        }
    }
    Ok(())
}

#[test]
fn multiclass_classification() -> Result<()> {
    let num_class = 3;

    let mut p = Parameters::new();
    p.push("num_class", num_class);
    p.push("objective", Objective::Multiclass);
    p.push("min_data_in_leaf", 20);
    p.push("verbosity", Verbosity::Fatal);

    println!("make train dataset");
    let train_feature = make_features(128, num_class);
    let train_label = make_labels(128, num_class);
    let mut train = Dataset::from_mat(&train_feature, None, &p)?;
    train.set_field(Field::LABEL, &train_label)?;

    println!("make test dataset");
    let test_feature = make_features(4, num_class);
    let test_label = make_labels(4, num_class);
    let mut test = Dataset::from_mat(&test_feature, Some(&train), &p)?;
    test.set_field(Field::LABEL, &test_label)?;

    println!("make booster");
    let mut b = Booster::new(Arc::new(train), &p)?;
    println!("add valid data");
    b.add_valid_data(Arc::new(test))?;
    for n in 0..100 {
        println!("update_one_iter {n}");
        let is_finish = b.update_one_iter()?;
        let eval_names = b.get_eval_names()?;
        let evals = b.get_eval(0)?;
        for i in 0..eval_names.len() {
            println!("training {}: {}", eval_names[i], evals[i]);
        }
        let evals = b.get_eval(1)?;
        for i in 0..eval_names.len() {
            println!("valid    {}: {}", eval_names[i], evals[i]);
        }
        if is_finish {
            break;
        }
    }

    let p = Parameters::new();
    let rs = b.predict_for_mat(&test_feature, PredictType::Normal, 0, None, &p)?;
    println!("\n{rs}");
    for n_data in 0..4 {
        for n_class in 0..3 {
            let r = rs[[n_data, n_class]];
            if n_data % 3 == n_class {
                assert!(r > 0.9);
            } else {
                assert!(r < 0.1);
            }
        }
    }
    Ok(())
}

#[test]
fn update_one_iter_custom() -> Result<()> {
    let num_class = 2;

    let mut p = Parameters::new();
    p.push("boosting_type", Boosting::Gbdt);
    p.push("objective", Objective::Custom);
    p.push("min_data_in_leaf", 20);
    p.push("verbosity", Verbosity::Fatal);

    println!("make train dataset");
    let train_feature = make_features(128, num_class);
    let train_label = make_labels(128, num_class);
    let mut train = Dataset::from_mat(&train_feature, None, &p)?;
    train.set_field(Field::LABEL, &train_label)?;

    println!("make test dataset");
    let test_feature = make_features(4, num_class);
    let test_label = make_labels(4, num_class);
    let mut test = Dataset::from_mat(&test_feature, Some(&train), &p)?;
    test.set_field(Field::LABEL, &test_label)?;

    println!("crate booster");
    let mut b = Booster::new(Arc::new(train), &p)?;
    b.add_valid_data(Arc::new(test))?;

    let mut grad = Vec::new();
    let mut hess = Vec::new();
    for n in 0..100 {
        println!("iter {n}");
        grad.clear();
        hess.clear();
        let p = b.get_predict(0)?;
        for i in 0..train_label.len() {
            let p = (p[i] as f32).max(0.0001).min(0.9999);
            let g = p as f32 - train_label[i];
            grad.push(g);
            hess.push(p * (1.0 - p));
        }
        let is_finish = b.update_one_iter_custom(&grad, &hess)?;
        let eval_names = b.get_eval_names()?;
        let evals = b.get_eval(0)?;
        for i in 0..eval_names.len() {
            println!("training {}: {}", eval_names[i], evals[i]);
        }
        let evals = b.get_eval(1)?;
        for i in 0..eval_names.len() {
            println!("valid    {}: {}", eval_names[i], evals[i]);
        }
        if is_finish {
            break;
        }
    }

    let p = Parameters::new();
    let rs = b.predict_for_mat(&test_feature, PredictType::Normal, 0, None, &p)?;
    println!("\n{rs}");
    for i in 0..test_label.len() {
        let r = rs[i];
        if test_label[i] == 0.0 {
            assert!(r < 0.1);
        } else {
            assert!(r > 0.9);
        }
    }
    Ok(())
}

#[test]
fn calc_num_predict_binary() -> Result<()> {
    let mut p = Parameters::new();
    p.push("objective", Objective::Binary);
    p.push("verbosity", Verbosity::Fatal);

    let d = make_dataset(125, 2, None, &p)?;
    let mut b = Booster::new(d, &p)?;
    for _ in 0..10 {
        if b.update_one_iter()? {
            break;
        }
    }
    let num_row = 10;
    let num_iteration = b.get_current_iteration()?;
    let num_feature = 2;
    let num_class = 1;
    assert!(num_iteration > 1);

    let value = b.calc_num_predict(num_row, PredictType::Normal, 0, None)?;
    assert_eq!(value, num_class * num_row);

    let value = b.calc_num_predict(num_row, PredictType::RawScore, 0, None)?;
    assert_eq!(value, num_class * num_row);

    let value = b.calc_num_predict(num_row, PredictType::LeafIndex, 0, None)?;
    assert_eq!(value, num_class * num_row * num_iteration);

    let value = b.calc_num_predict(num_row, PredictType::Contrib, 0, None)?;
    assert_eq!(value, num_class * num_row * (num_feature + 1));

    Ok(())
}

#[test]
fn calc_num_predict_multiclass() -> Result<()> {
    let num_class = 3;

    let mut p = Parameters::new();
    p.push("objective", Objective::Multiclass);
    p.push("num_class", num_class);
    p.push("verbosity", Verbosity::Fatal);

    let d = make_dataset(125, num_class, None, &p)?;
    let mut b = Booster::new(d, &p)?;
    for _ in 0..10 {
        if b.update_one_iter()? {
            break;
        }
    }
    let num_row = 10;
    let num_iteration = b.get_current_iteration()?;
    let num_feature = 2;
    assert_eq!(num_class, 3);
    assert!(num_iteration > 1);

    let value = b.calc_num_predict(num_row, PredictType::Normal, 0, None)?;
    assert_eq!(value, num_class * num_row);

    let value = b.calc_num_predict(num_row, PredictType::RawScore, 0, None)?;
    assert_eq!(value, num_class * num_row);

    let value = b.calc_num_predict(num_row, PredictType::LeafIndex, 0, None)?;
    assert_eq!(value, num_class * num_row * num_iteration);

    let value = b.calc_num_predict(num_row, PredictType::Contrib, 0, None)?;
    assert_eq!(value, num_class * num_row * (num_feature + 1));

    Ok(())
}

#[test]
fn get_num_predict_binary() -> Result<()> {
    let num_row = [128, 64];
    let num_class = 2;
    let mut p = Parameters::new();
    p.push("objective", Objective::Binary);
    p.push("verbosity", Verbosity::Fatal);

    let train_data = make_dataset(num_row[0], num_class, None, &p)?;
    let valid_data = make_dataset(num_row[1], num_class, Some(&train_data), &p)?;
    let mut b = Booster::new(train_data, &p)?;
    b.add_valid_data(valid_data)?;
    for _ in 0..10 {
        if b.update_one_iter()? {
            break;
        }
    }

    for (data_idx, &num_row) in num_row.iter().enumerate() {
        let value = b.get_num_predict(data_idx)?;
        assert_eq!(value, num_row);
    }
    Ok(())
}

#[test]
fn get_num_predict_multiclass() -> Result<()> {
    let num_row = [128, 64];
    let num_class = 3;

    let mut p = Parameters::new();
    p.push("objective", Objective::Multiclass);
    p.push("num_class", num_class);
    p.push("verbosity", Verbosity::Fatal);

    let train_data = make_dataset(num_row[0], num_class, None, &p)?;
    let valid_data = make_dataset(num_row[1], num_class, Some(&train_data), &p)?;
    let mut b = Booster::new(train_data, &p)?;
    b.add_valid_data(valid_data)?;
    for _ in 0..10 {
        if b.update_one_iter()? {
            break;
        }
    }

    for (data_idx, &num_row) in num_row.iter().enumerate() {
        let value = b.get_num_predict(data_idx)?;
        assert_eq!(value, num_row * num_class);
    }
    Ok(())
}

#[test]
fn get_current_iteration() -> Result<()> {
    let num_row = [128, 64];
    let num_class = 2;
    let mut p = Parameters::new();
    p.push("objective", Objective::Binary);
    p.push("verbosity", Verbosity::Fatal);

    let train_data = make_dataset(num_row[0], num_class, None, &p)?;
    let valid_data = make_dataset(num_row[1], num_class, Some(&train_data), &p)?;
    let mut b = Booster::new(train_data, &p)?;
    b.add_valid_data(valid_data)?;
    for _ in 0..10 {
        if b.update_one_iter()? {
            break;
        }
    }
    let iteration = b.get_current_iteration()?;
    assert_eq!(iteration, 10);
    Ok(())
}

#[test]
fn rollback_one_iter() -> Result<()> {
    let num_row = [128, 64];
    let num_class = 2;
    let mut p = Parameters::new();
    p.push("objective", Objective::Binary);
    p.push("verbosity", Verbosity::Fatal);

    let train_data = make_dataset(num_row[0], num_class, None, &p)?;
    let valid_data = make_dataset(num_row[1], num_class, Some(&train_data), &p)?;
    let mut b = Booster::new(train_data, &p)?;
    b.add_valid_data(valid_data)?;
    for _ in 0..10 {
        if b.update_one_iter()? {
            break;
        }
    }
    assert_eq!(b.get_current_iteration()?, 10);
    b.rollback_one_iter()?;
    assert_eq!(b.get_current_iteration()?, 9);
    b.rollback_one_iter()?;
    assert_eq!(b.get_current_iteration()?, 8);
    Ok(())
}

#[test]
fn save_to_string() -> Result<()> {
    let num_row = [128, 64];
    let num_class = 2;
    let mut p = Parameters::new();
    p.push("objective", Objective::Binary);
    p.push("verbosity", Verbosity::Fatal);

    let train_data = make_dataset(num_row[0], num_class, None, &p)?;
    let valid_data = make_dataset(num_row[1], num_class, Some(&train_data), &p)?;
    let mut b = Booster::new(train_data, &p)?;
    b.add_valid_data(valid_data)?;
    for _ in 0..10 {
        if b.update_one_iter()? {
            break;
        }
    }
    let i0 = b.get_current_iteration()?;

    let features = make_features(64, num_class);
    for start_iteration in [0, 3, 6] {
        let m = b.save_model_to_string(start_iteration, None, FeatureImportanceType::Gain)?;
        let (b1, i1) = Booster::from_string(&m)?;
        assert_eq!(i1, i0 - start_iteration);
        assert_eq!(b1.get_current_iteration()?, i0 - start_iteration);

        let r0 = b.predict_for_mat(&features, PredictType::Normal, start_iteration, None, &p)?;
        let r1 = b1.predict_for_mat(&features, PredictType::Normal, 0, None, &p)?;
        assert!(r0.approx_eq(&r1, 0.001));
    }

    Ok(())
}

fn make_features(num_row: usize, num_class: usize) -> MatBuf<f64, RowMajor> {
    MatBuf::from_rows((0..num_row).map(|x| [(x % num_class) as f64 + 1.0, x as f64]))
}
fn make_labels(num_row: usize, num_class: usize) -> Vec<f32> {
    (0..num_row).map(|x| (x % num_class) as f32).collect()
}

fn make_features_categorycal(num_row: usize, num_category: usize) -> MatBuf<f64, RowMajor> {
    MatBuf::from_rows((0..num_row).map(|x| {
        let category = x % num_category;
        let value = (x / num_category) % ((category + 1) * 2);
        [category as f64, value as f64]
    }))
}
fn make_labels_categorycal(num_row: usize, num_category: usize) -> Vec<f32> {
    (0..num_row)
        .map(|x| {
            let category = x % num_category;
            let value = (x / num_category) % ((category + 1) * 2);
            if value > category {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn make_dataset(
    num_row: usize,
    num_class: usize,
    reference: Option<&Dataset>,
    p: &Parameters,
) -> Result<Arc<Dataset>> {
    let mut d = Dataset::from_mat(&make_features(num_row, num_class), reference, p)?;
    d.set_field(Field::LABEL, &make_labels(num_row, num_class))?;
    Ok(Arc::new(d))
}

fn parameters() -> Parameters {
    let mut p = Parameters::new();
    p.push("verbosity", Verbosity::Fatal);
    p
}
