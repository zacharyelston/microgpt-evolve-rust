//! Batch Experiment Runner
//! 
//! Run multiple 3x3 experiments with different configurations and compare results.

use std::collections::HashMap;
use std::fs;
use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentSummary {
    name: String,
    config_file: String,
    final_accuracy: f64,
    best_accuracy: f64,
    training_time: f64,
    plateau_detected: bool,
    per_direction_performance: HashMap<String, f64>,
    status: String, // "completed", "failed", "running"
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchResults {
    timestamp: String,
    experiments: Vec<ExperimentSummary>,
    comparison: ComparisonMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
struct ComparisonMetrics {
    best_accuracy: f64,
    worst_accuracy: f64,
    average_accuracy: f64,
    most_stable: String,
    fastest_training: String,
    best_per_direction: HashMap<String, String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    // Default experiments to run
    let default_configs = vec![
        "configs/basic_3x3.json",
        "configs/spatial_3x3.json", 
        "configs/enhanced_3x3.json",
        "configs/deep_3x3.json",
    ];
    
    let configs = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        default_configs
    };
    
    println!("=== Batch 3x3 Experiment Runner ===");
    println!("Running {} experiments:", configs.len());
    for (i, config) in configs.iter().enumerate() {
        println!("  {}. {}", i + 1, config);
    }
    println!();
    
    // Create results directory
    fs::create_dir_all("batch_results")?;
    
    // Run experiments
    let mut results = Vec::new();
    
    for config_file in &configs {
        println!("--- Running experiment: {} ---", config_file);
        
        let experiment_name = extract_experiment_name(config_file);
        let output_dir = format!("batch_results/{}", experiment_name);
        
        // Run the configurable trainer
        let output = Command::new("cargo")
            .args(&[
                "run", "--release", "--bin", "train_3x3_configurable",
                "config", config_file
            ])
            .output()?;
        
        let status = if output.status.success() {
            println!("✅ Completed: {}", config_file);
            
            // Try to load results
            let results_file = format!("{}/results.json", output_dir);
            if let Ok(results_content) = fs::read_to_string(&results_file) {
                if let Ok(training_results) = serde_json::from_str::<serde_json::Value>(&results_content) {
                    let summary = ExperimentSummary {
                        name: experiment_name.clone(),
                        config_file: config_file.clone(),
                        final_accuracy: training_results["final_accuracy"].as_f64().unwrap_or(0.0),
                        best_accuracy: training_results["best_accuracy"].as_f64().unwrap_or(0.0),
                        training_time: training_results["training_time"].as_f64().unwrap_or(0.0),
                        plateau_detected: training_results["plateau_detected"].as_bool().unwrap_or(false),
                        per_direction_performance: parse_per_direction(&training_results["per_direction_accuracy"]),
                        status: "completed".to_string(),
                    };
                    results.push(summary);
                } else {
                    println!("⚠️  Could not parse results for {}", config_file);
                    results.push(create_failed_summary(config_file, &experiment_name));
                }
            } else {
                println!("⚠️  Could not find results file for {}", config_file);
                results.push(create_failed_summary(config_file, &experiment_name));
            }
        } else {
            println!("❌ Failed: {}", config_file);
            eprintln!("Error output: {}", String::from_utf8_lossy(&output.stderr));
            results.push(create_failed_summary(config_file, &experiment_name));
        }
        
        println!();
    }
    
    // Generate comparison
    let comparison = generate_comparison(&results);
    
    // Save batch results
    let batch_results = BatchResults {
        timestamp: chrono::Utc::now().to_rfc3339(),
        experiments: results.clone(),
        comparison,
    };
    
    let results_file = "batch_results/batch_summary.json";
    fs::write(results_file, serde_json::to_string_pretty(&batch_results)?)?;
    
    // Print summary
    print_batch_summary(&batch_results);
    
    Ok(())
}

fn extract_experiment_name(config_file: &str) -> String {
    config_file
        .split('/')
        .last()
        .unwrap_or("unknown")
        .trim_end_matches(".json")
        .to_string()
}

fn parse_per_direction(value: &serde_json::Value) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    if let Some(obj) = value.as_object() {
        for (key, val) in obj {
            if let Some(accuracy) = val.as_f64() {
                map.insert(key.clone(), accuracy);
            }
        }
    }
    map
}

fn create_failed_summary(config_file: &str, experiment_name: &str) -> ExperimentSummary {
    ExperimentSummary {
        name: experiment_name.to_string(),
        config_file: config_file.to_string(),
        final_accuracy: 0.0,
        best_accuracy: 0.0,
        training_time: 0.0,
        plateau_detected: false,
        per_direction_performance: HashMap::new(),
        status: "failed".to_string(),
    }
}

fn generate_comparison(results: &[ExperimentSummary]) -> ComparisonMetrics {
    let completed: Vec<_> = results.iter()
        .filter(|r| r.status == "completed")
        .collect();
    
    if completed.is_empty() {
        return ComparisonMetrics {
            best_accuracy: 0.0,
            worst_accuracy: 0.0,
            average_accuracy: 0.0,
            most_stable: "None".to_string(),
            fastest_training: "None".to_string(),
            best_per_direction: HashMap::new(),
        };
    }
    
    let accuracies: Vec<f64> = completed.iter().map(|r| r.final_accuracy).collect();
    let best_accuracy = accuracies.iter().fold(0.0, |a, &b| a.max(b));
    let worst_accuracy = accuracies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let average_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
    
    let most_stable = completed.iter()
        .min_by(|a, b| a.plateau_detected.cmp(&b.plateau_detected))
        .map(|r| r.name.clone())
        .unwrap_or("None".to_string());
    
    let fastest_training = completed.iter()
        .min_by(|a, b| a.training_time.partial_cmp(&b.training_time).unwrap())
        .map(|r| r.name.clone())
        .unwrap_or("None".to_string());
    
    // Find best performing experiment for each direction
    let mut best_per_direction = HashMap::new();
    let directions = ["L", "R", "T", "B", "TL", "TR", "BL", "BR", "U"];
    
    for &direction in &directions {
        let best = completed.iter()
            .filter_map(|r| {
                r.per_direction_performance.get(direction)
                    .map(|&acc| (acc, &r.name))
            })
            .max_by(|a, b| a.0.partial_cmp(b.0).unwrap());
        
        if let Some((_, name)) = best {
            best_per_direction.insert(direction.to_string(), name.clone());
        }
    }
    
    ComparisonMetrics {
        best_accuracy,
        worst_accuracy,
        average_accuracy,
        most_stable,
        fastest_training,
        best_per_direction,
    }
}

fn print_batch_summary(results: &BatchResults) {
    println!("=== Batch Experiment Summary ===");
    println!("Timestamp: {}", results.timestamp);
    println!("Total experiments: {}", results.experiments.len());
    
    let completed = results.experiments.iter().filter(|r| r.status == "completed").count();
    let failed = results.experiments.iter().filter(|r| r.status == "failed").count();
    
    println!("Completed: {}", completed);
    println!("Failed: {}", failed);
    
    if completed > 0 {
        println!("\n--- Performance Comparison ---");
        println!("Best accuracy: {:.2}%", results.comparison.best_accuracy);
        println!("Worst accuracy: {:.2}%", results.comparison.worst_accuracy);
        println!("Average accuracy: {:.2}%", results.comparison.average_accuracy);
        println!("Most stable: {}", results.comparison.most_stable);
        println!("Fastest training: {}", results.comparison.fastest_training);
        
        println!("\n--- Individual Results ---");
        for experiment in &results.experiments {
            let status_icon = match experiment.status.as_str() {
                "completed" => "✅",
                "failed" => "❌",
                _ => "⏳",
            };
            
            println!("{} {}: {:.2}% ({:.1}s)", 
                status_icon, 
                experiment.name, 
                experiment.final_accuracy,
                experiment.training_time
            );
        }
        
        println!("\n--- Best Per-Direction Performance ---");
        for (direction, best_experiment) in &results.comparison.best_per_direction {
            println!("{}: {}", direction, best_experiment);
        }
    }
    
    println!("\n--- Files Generated ---");
    println!("📁 batch_results/batch_summary.json - Complete results");
    for experiment in &results.experiments {
        if experiment.status == "completed" {
            println!("📁 batch_results/{}/ - Individual experiment", experiment.name);
        }
    }
    
    println!("\n=== Batch Testing Complete ===");
}
