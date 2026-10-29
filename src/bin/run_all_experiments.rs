//! Batch experiment runner
//! 
//! Runs all 10 experiments in sequence and generates a comprehensive report

use std::process::Command;
use std::time::Instant;
use std::path::Path;

const EXPERIMENTS: &[&str] = &[
    "exp01_basic_neuron",
    "exp02_larger_neuron", 
    "exp03_bundle_foundation",
    "exp04_evolution_test",
    "exp05_optimization_test",
    "exp06_performance_test",
    "exp07_production_scale",
    "exp08_ensemble_test",
    "exp09_adaptive_test",
    "exp10_ultimate_test",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MicroGPT Evolution Batch Runner ===");
    println!("Running {} experiments sequentially...", EXPERIMENTS.len());
    
    let mut results = Vec::new();
    let total_start = Instant::now();
    
    for (i, exp_name) in EXPERIMENTS.iter().enumerate() {
        println!("\n[{}] 🚀 Starting: {}", i + 1, exp_name);
        let exp_start = Instant::now();
        
        // Run the experiment
        let config_path = format!("experiments/config/{}.json", exp_name);
        let output = Command::new("cargo")
            .args(&["run", "--release", "--bin", "run_experiment", "--", &config_path])
            .output()?;
        
        let exp_duration = exp_start.elapsed();
        let success = output.status.success();
        
        // Store results
        results.push(ExperimentResult {
            name: exp_name.to_string(),
            success,
            duration: exp_duration,
            exit_code: output.status.code().unwrap_or(-1),
        });
        
        // Print result
        if success {
            println!("✅ {}: SUCCESS ({:.2}s)", exp_name, exp_duration.as_secs_f64());
        } else {
            println!("❌ {}: FAILED (exit code: {})", exp_name, output.status.code().unwrap_or(-1));
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                println!("Error: {}", stderr);
            }
        }
        
        // Save intermediate results
        save_results(&results)?;
    }
    
    let total_duration = total_start.elapsed();
    
    // Generate final report
    generate_report(&results, total_duration)?;
    
    println!("\n=== Batch Execution Complete ===");
    println!("Total time: {:.2}s", total_duration.as_secs_f64());
    println!("Successful: {}/{}", results.iter().filter(|r| r.success).count(), results.len());
    
    Ok(())
}

#[derive(serde::Serialize)]
struct ExperimentResult {
    name: String,
    success: bool,
    duration: std::time::Duration,
    exit_code: i32,
}

fn save_results(results: &[ExperimentResult]) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write("results/batch_results.json", json)?;
    Ok(())
}

fn generate_report(results: &[ExperimentResult], total_duration: std::time::Duration) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    
    report.push_str("# MicroGPT Evolution Experiment Report\n\n");
    report.push_str(&format!("**Total Experiments**: {}\n", results.len()));
    report.push_str(&format!("**Total Duration**: {:.2}s\n\n", total_duration.as_secs_f64()));
    
    // Summary
    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.len() - successful;
    report.push_str("## Summary\n\n");
    report.push_str(&format!("- ✅ **Successful**: {} ({:.1}%)\n", successful, successful as f64 / results.len() as f64 * 100.0));
    report.push_str(&format!("- ❌ **Failed**: {} ({:.1}%)\n\n", failed, failed as f64 / results.len() as f64 * 100.0));
    
    // Detailed results
    report.push_str("## Detailed Results\n\n");
    report.push_str("| # | Experiment | Status | Duration | Exit Code |\n");
    report.push_str("|---|------------|--------|----------|----------|\n");
    
    for (i, result) in results.iter().enumerate() {
        let status = if result.success { "✅" } else { "❌" };
        let duration = format!("{:.2}s", result.duration.as_secs_f64());
        report.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            i + 1,
            result.name,
            status,
            duration,
            result.exit_code
        ));
    }
    
    // Performance analysis
    report.push_str("\n## Performance Analysis\n\n");
    if successful > 0 {
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        let avg_duration: f64 = successful_results.iter()
            .map(|r| r.duration.as_secs_f64())
            .sum::<f64>() / successful_results.len() as f64;
        let min_duration = successful_results.iter()
            .map(|r| r.duration.as_secs_f64())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_duration = successful_results.iter()
            .map(|r| r.duration.as_secs_f64())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        report.push_str(&format!("- **Average Duration**: {:.2}s\n", avg_duration));
        report.push_str(&format!("- **Fastest**: {:.2}s\n", min_duration));
        report.push_str(&format!("- **Slowest**: {:.2}s\n", max_duration));
    }
    
    // Recommendations
    report.push_str("\n## Recommendations\n\n");
    if failed == 0 {
        report.push_str("🎉 **All experiments completed successfully!**\n");
        report.push_str("- System is ready for production deployment\n");
        report.push_str("- Consider running with larger datasets\n");
    } else if failed <= 2 {
        report.push_str("⚠️ **Mostly successful**\n");
        report.push_str("- Review failed experiments for configuration issues\n");
        report.push_str("- System is nearly production-ready\n");
    } else {
        report.push_str("❌ **Multiple failures detected**\n");
        report.push_str("- Review system configuration and resources\n");
        report.push_str("- Consider reducing experiment complexity\n");
    }
    
    std::fs::write("results/experiment_report.md", report)?;
    
    println!("📊 Report generated: results/experiment_report.md");
    
    Ok(())
}
