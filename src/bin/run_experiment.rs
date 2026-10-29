//! Main binary for running experiments
//! 
//! Usage: cargo run --release --bin run_experiment -- [config_path]

use std::env;
use std::path::PathBuf;

use microgpt_rust::experiments::run_experiment::ExperimentRunner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    let config_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("experiments/config/experiment_config.json")
    };
    
    println!("=== MicroGPT Evolution Experiment Runner ===");
    println!("Config: {}", config_path.display());
    
    if !config_path.exists() {
        eprintln!("Error: Config file not found: {}", config_path.display());
        eprintln!("Create the config file or specify a valid path");
        std::process::exit(1);
    }
    
    let mut runner = ExperimentRunner::new(config_path)?;
    runner.run()?;
    
    println!("=== Experiment Completed ===");
    Ok(())
}
