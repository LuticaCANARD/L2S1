//! Offline family fitting from an inspected identity and labeled raw-logit
//! records of several tasks sharing one decision kind and option count.
use l2s1::*;
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    id: String,
    model: ModelIdentity,
    tasks: Vec<FamilyCalibrationTask>,
    /// Records from tasks and source groups absent from `tasks`.
    #[serde(default)]
    held_out: Vec<CalibrationRecord>,
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: fit_family_calibration INPUT.json ARTIFACT.json";
    let path = args.next().ok_or(usage)?;
    let output = args.next().ok_or(usage)?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let input: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    let artifact = FamilyCalibration::fit(input.id, &input.model, &input.tasks)?;
    let held_out = if input.held_out.is_empty() {
        None
    } else {
        Some((
            calibration_metrics(&input.held_out, 1.0)?,
            artifact.evaluate_held_out(&input.held_out)?,
        ))
    };
    // Failed held-out validation never publishes an artifact.
    std::fs::write(output, serde_json::to_vec_pretty(&artifact)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope": {"decision_kind": artifact.decision_kind, "option_count": artifact.option_count},
            "temperature": artifact.temperature,
            "fit_uncalibrated": artifact.uncalibrated_fit_metrics,
            "fit_calibrated": artifact.calibrated_fit_metrics,
            "leave_one_task_out": artifact.leave_one_task_out,
            "held_out_uncalibrated": held_out.as_ref().map(|h| &h.0),
            "held_out_calibrated": held_out.as_ref().map(|h| &h.1),
            "evidence": "leave-one-task-out refits without each task and its source groups; held_out is caller-supplied"
        }))?
    );
    Ok(())
}
