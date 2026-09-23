mod analyze_intent_rotation;
mod benchmark_models;
mod calibrate_ag_news;
mod common;
mod evaluate_decision_lora;
mod evaluate_intents;
mod evaluate_intents_wide;
mod jevbench_matrix;
mod jevbench_public;
mod kaggle_ag_news;
mod kaggle_airline;
mod laya_benchmark;
mod parquet_data;
mod prepare_accuracy_study;
mod prepare_decision_finetune;
mod prepare_output_head;
mod probe_airline_order;
mod python_random;
mod report_accuracy_study;
mod report_airline;
mod report_decision_finetune;
mod report_intents;
mod report_intents_wide;
mod report_jevbench_matrix;
mod report_output_head;
mod summarize_output_head_runtime;
mod tune_compute;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "L2S1 dataset, benchmark, and report workflows")]
struct Cli {
    /// Repository root containing Cargo.toml, tests/, and models/.
    #[arg(long, global = true, default_value = ".")]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the local GGUF decision rule benchmark.
    BenchmarkModels(benchmark_models::Args),
    /// Audit paired answer-code rotation diagnostics.
    AnalyzeIntentRotation(analyze_intent_rotation::Args),
    /// Prepare and fit AG News candidate temperature calibration.
    CalibrateAgNews(calibrate_ag_news::Args),
    /// Prepare, run, or report the AG News study.
    KaggleAgNews(kaggle_ag_news::Args),
    /// Prepare, run, or report the airline sentiment study.
    KaggleAirline(kaggle_airline::Args),
    /// Evaluate the pinned public JevBench set with local GGUF inference.
    JevbenchPublic(jevbench_public::Args),
    /// Download pinned checkpoints and run a serial JevBench matrix.
    JevbenchMatrix(jevbench_matrix::Args),
    /// Fetch, prepare, evaluate, and compare pinned public Laya tasks.
    LayaBenchmark(laya_benchmark::Args),
    /// Prepare, run, or score the BANKING77/MASSIVE intent tournament.
    EvaluateIntents(evaluate_intents::Args),
    /// Evaluate all intent labels in one fixed-width code decision.
    EvaluateIntentsWide(evaluate_intents_wide::Args),
    /// Run paired base/LoRA evaluation and timing checks.
    EvaluateDecisionLora(evaluate_decision_lora::Args),
    /// Freeze disjoint airline training, calibration, test, and probe splits.
    PrepareDecisionFinetune(prepare_decision_finetune::Args),
    /// Freeze and validate the paired synthetic accuracy study.
    PrepareAccuracyStudy(prepare_accuracy_study::Args),
    /// Freeze the output-head training and evaluation split.
    PrepareOutputHead(prepare_output_head::Args),
    /// Run the post hoc airline label order diagnostic.
    ProbeAirlineOrder(probe_airline_order::Args),
    /// Compare base and LoRA prediction artifacts.
    ReportDecisionFinetune(report_decision_finetune::Args),
    /// Audit and render the airline model comparison.
    ReportAirline(report_airline::Args),
    /// Audit frozen accuracy predictions and optionally select a dev configuration.
    ReportAccuracyStudy(report_accuracy_study::Args),
    /// Recount and render a public JevBench model matrix.
    ReportJevbenchMatrix(report_jevbench_matrix::Args),
    /// Independently audit intent tournament predictions.
    ReportIntents(report_intents::Args),
    /// Independently audit full-label intent scores and regression evidence.
    ReportIntentsWide(report_intents_wide::Args),
    /// Audit trained output-head scores and holdout results.
    ReportOutputHead(report_output_head::Args),
    /// Audit paired output-head timing and result equality.
    SummarizeOutputHeadRuntime(summarize_output_head_runtime::Args),
    /// Execute and compare a pinned compute configuration matrix.
    TuneCompute(tune_compute::Args),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = cli.root.canonicalize()?;
    match cli.command {
        Command::BenchmarkModels(args) => benchmark_models::run(&root, args),
        Command::AnalyzeIntentRotation(args) => analyze_intent_rotation::run(args),
        Command::CalibrateAgNews(args) => calibrate_ag_news::run(args),
        Command::KaggleAgNews(args) => kaggle_ag_news::run(&root, args),
        Command::KaggleAirline(args) => kaggle_airline::run(&root, args),
        Command::JevbenchPublic(args) => jevbench_public::run(args),
        Command::JevbenchMatrix(args) => jevbench_matrix::run(args),
        Command::LayaBenchmark(args) => laya_benchmark::run(args),
        Command::EvaluateIntents(args) => evaluate_intents::run(args),
        Command::EvaluateIntentsWide(args) => evaluate_intents_wide::run(args),
        Command::EvaluateDecisionLora(args) => evaluate_decision_lora::run(&root, args),
        Command::PrepareDecisionFinetune(args) => prepare_decision_finetune::run(args),
        Command::PrepareAccuracyStudy(args) => prepare_accuracy_study::run(args),
        Command::PrepareOutputHead(args) => prepare_output_head::run(args),
        Command::ProbeAirlineOrder(args) => probe_airline_order::run(&root, args),
        Command::ReportDecisionFinetune(args) => report_decision_finetune::run(args),
        Command::ReportAirline(args) => report_airline::run(&root, args),
        Command::ReportAccuracyStudy(args) => report_accuracy_study::run(args),
        Command::ReportJevbenchMatrix(args) => report_jevbench_matrix::run(args),
        Command::ReportIntents(args) => report_intents::run(args),
        Command::ReportIntentsWide(args) => report_intents_wide::run(args),
        Command::ReportOutputHead(args) => report_output_head::run(args),
        Command::SummarizeOutputHeadRuntime(args) => summarize_output_head_runtime::run(args),
        Command::TuneCompute(args) => tune_compute::run(args),
    }
}
