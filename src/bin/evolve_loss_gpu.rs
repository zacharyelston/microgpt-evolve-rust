/*
    MicroGPT Evolution Engine v3 - GPU Version
    
    Evolution engine with GPU-accelerated matrix operations.
    This will be compared against the CPU-only version.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

const POPULATION_SIZE: usize = 12;
const NUM_GENERATIONS: usize = 20;
const MUTATION_RATE: f64 = 0.3;
const TARGET_LOSS: f64 = 1.2;
const INPUT_FILE: &str = "input.txt";

#[derive(Clone, Debug)]
struct Organism {
    config: TrainingConfig,
    loss: f64,
    evaluated: bool,
    origin: String,
}

impl Organism {
    fn from_config(config: &TrainingConfig, origin: &str) -> Self {
        Self {
            config: config.clone(),
            loss: f64::MAX,
            evaluated: false,
            origin: origin.to_string(),
        }
    }

    fn from_config_with_loss(config: &TrainingConfig, loss: f64, origin: &str) -> Self {
        Self {
            config: config.clone(),
            loss,
            evaluated: true,
            origin: origin.to_string(),
        }
    }

    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let config = TrainingConfig {
            n_emb: *[8, 12, 16, 20, 24, 32].choose(&mut rng).unwrap(),
            n_ctx: *[8, 12, 16, 24, 32].choose(&mut rng).unwrap(),
            n_layer: rng.gen_range(1..=4),
            n_head: *[1, 2, 4].choose(&mut rng).unwrap(),
            n_ff_exp: rng.gen_range(1..=4),
            lr: 10f64.powf(rng.gen_range(-3.0..-1.0)),
            steps: *[200, 400, 600, 800, 1000, 1500].choose(&mut rng).unwrap(),
            input_file: INPUT_FILE.to_string(),
            gen_samples: 1,
            ..Default::default()
        };
        Self::from_config(&config, "random")
    }

    fn mutate(&self, base_config: &TrainingConfig) -> Self {
        let mut rng = rand::thread_rng();
        let mut config = base_config.clone();
        
        // Small mutations around the base config
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_emb = *[8, 12, 16, 20, 24, 32].choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_ctx = *[8, 12, 16, 24, 32].choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_layer = rng.gen_range(1..=4);
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_head = *[1, 2, 4].choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_ff_exp = rng.gen_range(1..=4);
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.lr = 10f64.powf(rng.gen_range(-3.0..-1.0));
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.steps = *[200, 400, 600, 800, 1000, 1500].choose(&mut rng).unwrap();
        }
        
        Self::from_config(&config, "mutant")
    }

    fn crossover(parent1: &Organism, parent2: &Organism) -> Self {
        let mut rng = rand::thread_rng();
        let mut config = parent1.config.clone();
        
        // Mix parameters from both parents
        if rng.gen() {
            config.n_emb = parent2.config.n_emb;
        }
        if rng.gen() {
            config.n_ctx = parent2.config.n_ctx;
        }
        if rng.gen() {
            config.n_layer = parent2.config.n_layer;
        }
        if rng.gen() {
            config.n_head = parent2.config.n_head;
        }
        if rng.gen() {
            config.n_ff_exp = parent2.config.n_ff_exp;
        }
        if rng.gen() {
            config.lr = parent2.config.lr;
        }
        if rng.gen() {
            config.steps = parent2.config.steps;
        }
        
        Self::from_config(&config, "cross")
    }

    fn evaluate(&mut self) {
        if self.evaluated {
            return;
        }
        
        let start = Instant::now();
        let result = std::panic::catch_unwind(|| {
            train_and_generate(&self.config, true)
        });
        
        match result {
            Ok(r) => {
                self.loss = r.final_loss;
                self.evaluated = true;
                println!("[GPU] {} | loss={:.4} ({:.1}s)", 
                    self.desc(), self.loss, start.elapsed().as_secs_f64());
            }
            Err(_) => {
                self.loss = f64::MAX;
                self.evaluated = true;
                println!("[GPU] {} | PANICKED", self.desc());
            }
        }
    }

    fn desc(&self) -> String {
        format!("Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.4} Steps:{} | {}",
            self.config.n_emb, self.config.n_head, self.config.n_layer,
            self.config.n_ctx, self.config.n_ff_exp, self.config.lr, self.config.steps,
            self.origin)
    }
}

fn main() {
    std::fs::create_dir_all("experiments").ok();
    
    println!("=== MicroGPT Evolution Engine v3 - GPU ACCELERATED ===");
    println!("Target: loss < {:.1}", TARGET_LOSS);
    println!("Population: {}, Generations: {}", POPULATION_SIZE, NUM_GENERATIONS);
    
    // Load training data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(INPUT_FILE);

    // Load existing genome or start fresh
    let (base_config, base_loss, base_gen) = TrainingConfig::load_genome()
        .unwrap_or_else(|| (TrainingConfig::default(), f64::MAX, 0));
    
    println!("Starting from genome: loss {:.6} (generation {})", base_loss, base_gen);

    // Create initial population
    let mut population = Vec::with_capacity(POPULATION_SIZE);
    
    // Always include the base genome if it exists
    if base_loss < f64::MAX {
        population.push(Organism::from_config_with_loss(&base_config, base_loss, "genome"));
        println!("Added base genome: loss {:.6}", base_loss);
    }
    
    // Fill with mutations around base config
    while population.len() < POPULATION_SIZE {
        if base_loss < f64::MAX && population.len() < POPULATION_SIZE / 2 {
            population.push(Organism::mutate(&population[0], &base_config));
        } else {
            population.push(Organism::random());
        }
    }

    let mut best_ever = population[0].clone();
    let total_start = Instant::now();

    // Evolution loop
    for gen in 0..NUM_GENERATIONS {
        println!("\n--- Generation {}/{} ---", gen + 1, NUM_GENERATIONS);
        
        // Evaluate population (skip already evaluated)
        population.par_iter_mut().for_each(|org| {
            if !org.evaluated {
                org.evaluate();
            }
        });

        // Sort by loss
        population.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());

        // Print results
        println!("Results:");
        for (i, org) in population.iter().take(5).enumerate() {
            println!("  #{}: {:.4} | {}", i + 1, org.loss, org.desc());
        }

        let best_this_gen = population[0].clone();
        println!("Best this gen: {:.4} | {}", best_this_gen.loss, best_this_gen.desc());

        // Update best ever
        if best_this_gen.loss < best_ever.loss {
            println!("🎉 NEW BEST! {:.6} -> {:.6}", best_ever.loss, best_this_gen.loss);
            best_ever = best_this_gen.clone();
            
            // Save if genuinely better than existing genome
            if best_ever.loss < base_loss {
                println!("💾 Saving improved genome...");
                if let Err(e) = best_ever.config.save_genome(best_ever.loss, gen + 1) {
                    println!("Failed to save genome: {}", e);
                }
            }
        }

        // Check target
        if best_ever.loss < TARGET_LOSS {
            println!("🎯 TARGET REACHED! Loss: {:.6}", best_ever.loss);
            break;
        }

        // Create next generation
        let mut next_pop = Vec::with_capacity(POPULATION_SIZE);
        
        // Keep elite
        next_pop.push(population[0].clone());
        
        // Add mutations of elite
        for _ in 0..3 {
            next_pop.push(Organism::mutate(&population[0], &base_config));
        }
        
        // Add crossovers
        for i in 0..2 {
            let parent2_idx = (i + 1) % population.len().min(3);
            next_pop.push(Organism::crossover(&population[0], &population[parent2_idx]));
        }
        
        // Add some random diversity
        while next_pop.len() < POPULATION_SIZE {
            next_pop.push(Organism::random());
        }
        
        population = next_pop;
    }

    println!("\n=== Evolution Complete (GPU) ===");
    println!("Time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!("Best loss: {:.6}", best_ever.loss);
    println!("Best config: {}", best_ever.desc());
    
    if best_ever.loss < base_loss {
        println!("✅ Improvement found: {:.6} -> {:.6}", base_loss, best_ever.loss);
    } else {
        println!("ℹ️  No improvement over base genome: {:.6}", base_loss);
    }
}
