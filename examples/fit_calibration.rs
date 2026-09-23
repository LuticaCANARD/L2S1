//! Offline fitting from an inspected identity, a task and labeled raw-logit records.
use l2s1::*;
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    id: String,
    model: ModelIdentity,
    decision: Decision,
    calibration: Vec<CalibrationRecord>,
    held_out: Vec<CalibrationRecord>,
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: fit_calibration INPUT.json ARTIFACT.json")?;
    let output = args
        .next()
        .ok_or("usage: fit_calibration INPUT.json ARTIFACT.json")?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let input: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    let artifact =
        ScalarCalibration::fit(input.id, &input.model, &input.decision, &input.calibration)?;
    let before = calibration_metrics(&input.held_out, 1.0)?;
    let after = artifact.evaluate_held_out(&input.held_out)?;
    // Failed held-out validation never publishes an artifact.
    std::fs::write(output, serde_json::to_vec_pretty(&artifact)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"held_out_uncalibrated":before,"held_out_calibrated":after,"temperature":artifact.temperature,"scope":"held-out records supplied by caller; group independence is checked by exact source group ID"})
        )?
    );
    Ok(())
}
