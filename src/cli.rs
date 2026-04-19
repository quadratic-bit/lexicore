use crate::commands;
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lexicore")]
#[command(about = "Pack, inspect, dump, build, and infer trigram-model bundles")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum InferModeArg {
    Tokens,
    Sentences,
}

#[derive(Subcommand)]
enum Commands {
    /// Read a bundle and print its section table
    Parse {
        #[arg(long)]
        input: PathBuf,
    },
    /// Pack a split filesystem layout into one bundle file
    Pack {
        #[arg(long)]
        input_root: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long = "model", required = true)]
        models: Vec<String>,
    },
    /// Dump a bundle back into a split filesystem layout
    Dump {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output_root: PathBuf,
    },
    /// Tokenize text files and build one bundle file
    Tokenize {
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long = "model", required = true)]
        models: Vec<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Load a bundle and generate output using two named models
    Infer {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        primary: String,
        #[arg(long)]
        secondary: String,
        #[arg(long, default_value_t = 0.03)]
        alpha: f64,
        #[arg(long, default_value_t = 0.25)]
        beta: f64,
        #[arg(long, value_enum, default_value_t = InferModeArg::Tokens)]
        mode: InferModeArg,
        #[arg(long)]
        size: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { input } => commands::parse::run(&input),
        Commands::Pack {
            input_root,
            output,
            models,
        } => commands::pack::run(&input_root, &output, &models),
        Commands::Dump { input, output_root } => commands::dump::run(&input, &output_root),
        Commands::Tokenize {
            inputs,
            models,
            output,
        } => commands::tokenize::run(&inputs, &models, &output),
        Commands::Infer {
            input,
            primary,
            secondary,
            alpha,
            beta,
            mode,
            size,
            seed,
        } => {
            let mode = match mode {
                InferModeArg::Tokens => commands::infer::OutputMode::Tokens,
                InferModeArg::Sentences => commands::infer::OutputMode::Sentences,
            };

            commands::infer::run(&input, &primary, &secondary, alpha, beta, mode, size, seed)
        }
    }
}
