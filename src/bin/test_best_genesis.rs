/*
    Test the best genesis organism to see what names it generates
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};

const INPUT_FILE: &str = "input.txt";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Best Genesis Organism ===");
    
    // Load training data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(INPUT_FILE);

    // Use the best config from genesis evolution
    let config = TrainingConfig {
        n_emb: 2,
        n_ctx: 3,
        n_layer: 3,
        n_head: 1,
        n_ff_exp: 2,
        lr: 0.0256,
        steps: 1000,
        input_file: INPUT_FILE.to_string(),
        gen_samples: 20,  // Generate 20 samples
        ..Default::default()
    };

    println!("Config: Emb:{} Ctx:{} Lay:{} Head:{} FF:{} LR:{:.4} Steps:{}", 
        config.n_emb, config.n_ctx, config.n_layer, 
        config.n_head, config.n_ff_exp, config.lr, config.steps);
    println!("Generating 20 name samples...\n");

    // Train and generate
    let result = train_and_generate(&config, false);
    
    println!("\n=== Results ===");
    println!("Final loss: {:.6}", result.final_loss);
    println!("Generated samples:");
    for (i, sample) in result.names.iter().enumerate() {
        println!("  {}: {}", i + 1, sample);
    }
    
    Ok(())
}
