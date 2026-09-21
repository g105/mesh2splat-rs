use anyhow::Result;
use clap::Parser;
use mesh2splat::cli::{self, Cli, Command};

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,mesh2splat=info"),
    )
    .init();
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Convert(args)) => cli::run_convert(args),
        Some(Command::Render(args)) => cli::run_render(args),
        Some(Command::Gui { file }) => run_gui(file),
        None => run_gui(None),
    }
}

#[cfg(feature = "gui")]
fn run_gui(file: Option<std::path::PathBuf>) -> Result<()> {
    mesh2splat::app::run(file)
}

#[cfg(not(feature = "gui"))]
fn run_gui(_file: Option<std::path::PathBuf>) -> Result<()> {
    anyhow::bail!("built without the `gui` feature; use the `convert` or `render` subcommands")
}
