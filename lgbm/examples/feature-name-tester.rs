use anyhow::Result;
use lgbm::{Dataset, MatBuf, Parameters};

fn main() -> Result<()> {
    // Minimal single-feature dataset for exercising feature-name validation.
    let data = MatBuf::from_rows([[0.0_f64; 1]; 1]);
    let mut dataset = Dataset::from_mat(data, None, &Parameters::new())?;

    let mut json_invalid = Vec::new();
    let mut other_invalid = Vec::new();
    for code in 0u8..=127 {
        let ch = code as char;
        let feature_name = format!("f{ch}");
        if let Err(err) = dataset.set_feature_names([feature_name.as_str()]) {
            let message = err.to_string();
            if message.contains("Do not support special JSON characters in feature name.") {
                json_invalid.push((code, ch));
            } else {
                other_invalid.push((code, ch, message));
            }
        }
    }

    println!(
        "ASCII characters rejected with \"Do not support special JSON characters in feature name.\":"
    );
    if json_invalid.is_empty() {
        println!("(none)");
    } else {
        for (code, ch) in json_invalid {
            println!("{code:3} '{}'", ch.escape_default());
        }
    }

    if !other_invalid.is_empty() {
        println!("\nOther ASCII characters rejected:");
        for (code, ch, message) in other_invalid {
            println!("{code:3} '{}' -> {message}", ch.escape_default());
        }
    }

    Ok(())
}
