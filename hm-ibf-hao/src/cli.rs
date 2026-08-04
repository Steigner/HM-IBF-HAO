//! Command line interface of the `hm-ibf-hao` binary.
//!
//! The experiment pipeline is exposed as one binary with subcommands so that training,
//! evaluation and the combined run share the same flags, defaults and `--help` output.

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};

use crate::config::{DEFAULT_EVALUATION_PARAMS, DEFAULT_TRAINING_PARAMS};

/// Default directory holding the instances the search trains on.
pub const DEFAULT_INSTANCES_DIR: &str = "instances";
/// Default directory receiving the trained GRAHF graph and its parameters.
pub const DEFAULT_RUN_DIR: &str = "hao_run";
/// Default directory receiving per-run evaluation results.
pub const DEFAULT_RESULTS_DIR: &str = "results";
/// Default base name of the trained elitist artefacts inside the run directory.
pub const DEFAULT_ELITIST: &str = "elitist_0";
/// Default aggregated evaluation summary file.
pub const DEFAULT_SUMMARY_CSV: &str = "results_hao.csv";

/// Preprocess -> train -> evaluate pipeline for the HM-IBF-HAO benchmark.
#[derive(Debug, Parser)]
#[command(name = "hm-ibf-hao", version, about, long_about = None)]
pub struct Cli {
    /// Number of worker threads; defaults to the available parallelism.
    #[arg(long, global = true, value_name = "N")]
    pub jobs: Option<usize>,

    /// Directory holding the instances the search trains on.
    #[arg(long, global = true, default_value = DEFAULT_INSTANCES_DIR, value_name = "DIR")]
    pub instances_dir: PathBuf,

    /// Directory holding the instances the trained algorithm is evaluated on.
    ///
    /// Defaults to `--instances-dir`, which is what the checked-in instance set expects; point
    /// it at a held-out directory to evaluate on unseen routes.
    #[arg(long, global = true, value_name = "DIR")]
    pub eval_instances_dir: Option<PathBuf>,

    /// Directory holding the trained GRAHF graph and its tuned parameters.
    #[arg(long, global = true, default_value = DEFAULT_RUN_DIR, value_name = "DIR")]
    pub run_dir: PathBuf,

    /// TOML file holding the GRAHF structure search's tuning parameters.
    ///
    /// Also supplies `max_evaluations`, the evaluation budget of a single metaheuristic
    /// run: the budget defines the benchmark, so training and evaluation share it, and
    /// `dimensions_allowed`, the working dimensions the islands may run at.
    #[arg(long, global = true, default_value = DEFAULT_TRAINING_PARAMS, value_name = "FILE")]
    pub training_params: PathBuf,

    /// TOML file holding the evaluate stage's tuning parameters.
    #[arg(long, global = true, default_value = DEFAULT_EVALUATION_PARAMS, value_name = "FILE")]
    pub evaluation_params: PathBuf,

    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// Returns the effective number of worker threads.
    ///
    /// # Returns
    ///
    /// The value of `--jobs`, or the machine's available parallelism if it was not given.
    /// Never returns zero.
    pub fn effective_jobs(&self) -> usize {
        self.jobs.unwrap_or_else(num_cpus::get).max(1)
    }

    /// Returns the directory the trained algorithm is evaluated on.
    ///
    /// # Returns
    ///
    /// `--eval-instances-dir` if it was given, otherwise `--instances-dir`.
    pub fn effective_eval_instances_dir(&self) -> &Path {
        self.eval_instances_dir
            .as_deref()
            .unwrap_or(&self.instances_dir)
    }
}

/// The pipeline stages exposed by the binary.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Search for a GRAHF island graph and store the elitist.
    Train(TrainArgs),
    /// Evaluate a trained island graph over a range of seeds.
    Evaluate(EvaluateArgs),
    /// Run `train` and then `evaluate` in one go.
    Pipeline {
        /// Training options.
        #[command(flatten)]
        train: TrainArgs,
        /// Evaluation options.
        #[command(flatten)]
        evaluate: EvaluateArgs,
    },
}

/// Options of the `train` subcommand.
#[derive(Debug, Clone, Args)]
pub struct TrainArgs {
    /// Seed of the outer GRAHF structure search.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Directory receiving the randomly generated initial population.
    #[arg(long, default_value = "experiments_hao", value_name = "DIR")]
    pub experiments_dir: PathBuf,
}

/// Options of the `evaluate` subcommand.
#[derive(Debug, Clone, Args)]
pub struct EvaluateArgs {
    /// First evaluation seed.
    #[arg(long, default_value_t = 42)]
    pub first_seed: u64,

    /// Number of consecutive evaluation seeds.
    #[arg(long, default_value_t = 15)]
    pub num_seeds: u64,

    /// Base name of the trained artefacts inside the run directory.
    #[arg(long, default_value = DEFAULT_ELITIST, value_name = "NAME")]
    pub elitist: String,

    /// Directory receiving the per-run result folders.
    #[arg(long, default_value = DEFAULT_RESULTS_DIR, value_name = "DIR")]
    pub results_dir: PathBuf,

    /// Aggregated summary written after all seeds finished.
    #[arg(long, default_value = DEFAULT_SUMMARY_CSV, value_name = "FILE")]
    pub summary_csv: PathBuf,
}

impl EvaluateArgs {
    /// Returns the seeds to evaluate.
    ///
    /// # Returns
    ///
    /// `num_seeds` consecutive seeds starting at `first_seed`; empty if `num_seeds` is zero.
    pub fn seeds(&self) -> Vec<u64> {
        (0..self.num_seeds)
            .map(|offset| self.first_seed + offset)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn the_command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn train_is_parsed_with_documented_defaults() {
        let cli = Cli::parse_from(["hm-ibf-hao", "train"]);

        assert_eq!(cli.instances_dir, PathBuf::from(DEFAULT_INSTANCES_DIR));
        assert_eq!(cli.run_dir, PathBuf::from(DEFAULT_RUN_DIR));
        assert_eq!(cli.training_params, PathBuf::from(DEFAULT_TRAINING_PARAMS));
        assert_eq!(
            cli.evaluation_params,
            PathBuf::from(DEFAULT_EVALUATION_PARAMS)
        );
        match cli.command {
            Command::Train(args) => {
                assert_eq!(args.seed, 42);
                assert_eq!(args.experiments_dir, PathBuf::from("experiments_hao"));
            }
            other => panic!("expected train, got {other:?}"),
        }
    }

    #[test]
    fn the_training_and_evaluation_param_files_are_shared_by_both_stages() {
        let cli = Cli::parse_from([
            "hm-ibf-hao",
            "pipeline",
            "--training-params",
            "/tmp/train.conf",
            "--evaluation-params",
            "/tmp/eval.conf",
        ]);

        assert_eq!(cli.training_params, PathBuf::from("/tmp/train.conf"));
        assert_eq!(cli.evaluation_params, PathBuf::from("/tmp/eval.conf"));
    }

    #[test]
    fn global_flags_are_accepted_after_the_subcommand() {
        let cli = Cli::parse_from(["hm-ibf-hao", "evaluate", "--instances-dir", "/tmp/inst"]);

        assert_eq!(cli.instances_dir, PathBuf::from("/tmp/inst"));
    }

    #[test]
    fn the_evaluation_instances_default_to_the_training_instances() {
        let cli = Cli::parse_from(["hm-ibf-hao", "evaluate", "--instances-dir", "/tmp/inst"]);

        assert_eq!(cli.effective_eval_instances_dir(), Path::new("/tmp/inst"));
    }

    #[test]
    fn a_held_out_evaluation_directory_overrides_the_default() {
        let cli = Cli::parse_from([
            "hm-ibf-hao",
            "pipeline",
            "--instances-dir",
            "/tmp/train",
            "--eval-instances-dir",
            "/tmp/eval",
        ]);

        assert_eq!(cli.instances_dir, PathBuf::from("/tmp/train"));
        assert_eq!(cli.effective_eval_instances_dir(), Path::new("/tmp/eval"));
    }

    #[test]
    fn jobs_defaults_to_the_available_parallelism() {
        let cli = Cli::parse_from(["hm-ibf-hao", "train"]);
        assert_eq!(cli.effective_jobs(), num_cpus::get().max(1));

        let cli = Cli::parse_from(["hm-ibf-hao", "--jobs", "3", "train"]);
        assert_eq!(cli.effective_jobs(), 3);
    }

    #[test]
    fn zero_jobs_is_clamped_to_one() {
        let cli = Cli::parse_from(["hm-ibf-hao", "--jobs", "0", "train"]);

        assert_eq!(cli.effective_jobs(), 1);
    }

    #[test]
    fn evaluation_seeds_are_consecutive() {
        let cli = Cli::parse_from([
            "hm-ibf-hao",
            "evaluate",
            "--first-seed",
            "7",
            "--num-seeds",
            "3",
        ]);

        match cli.command {
            Command::Evaluate(args) => assert_eq!(args.seeds(), vec![7, 8, 9]),
            other => panic!("expected evaluate, got {other:?}"),
        }
    }

    #[test]
    fn zero_seeds_evaluate_nothing() {
        let cli = Cli::parse_from(["hm-ibf-hao", "evaluate", "--num-seeds", "0"]);

        match cli.command {
            Command::Evaluate(args) => assert!(args.seeds().is_empty()),
            other => panic!("expected evaluate, got {other:?}"),
        }
    }

    #[test]
    fn the_pipeline_subcommand_carries_both_option_groups() {
        let cli = Cli::parse_from(["hm-ibf-hao", "pipeline", "--seed", "1", "--first-seed", "2"]);

        match cli.command {
            Command::Pipeline { train, evaluate } => {
                assert_eq!(train.seed, 1);
                assert_eq!(evaluate.first_seed, 2);
            }
            other => panic!("expected pipeline, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_subcommand_is_rejected() {
        assert!(Cli::try_parse_from(["hm-ibf-hao", "frobnicate"]).is_err());
    }
}
