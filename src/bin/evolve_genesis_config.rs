/*
    MicroGPT Evolution Engine - Genesis with Config File
    
    This version reads all parameter ranges from evolution_config.json
    and uses them to control the evolution bounds precisely.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig, gpu_accel::{create_gpu_accelerator, cpu_matrix_vector_multiply}, evolution_config::EvolutionConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Instant;
use std::sync::Arc;

const INPUT_FILE: &str = "input.txt";
const CONFIG_FILE: &str = "evolution_config.json";

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

    fn random(evo_config: &EvolutionConfig) -> Self {
        let mut rng = rand::thread_rng();
        let mut config = TrainingConfig {
            n_emb: *evo_config.get_n_emb_choices().choose(&mut rng).unwrap(),
            n_ctx: *evo_config.get_n_ctx_choices().choose(&mut rng).unwrap(),
            n_layer: evo_config.random_n_layer(),
            n_head: *evo_config.get_n_head_choices().choose(&mut rng).unwrap(),
            n_ff_exp: evo_config.random_n_ff_exp(),
            lr: evo_config.random_lr(),
            steps: *evo_config.get_steps_choices().choose(&mut rng).unwrap(),
            input_file: INPUT_FILE.to_string(),
            gen_samples: 1,
            ..Default::default()
        };
        
        // Clamp to valid ranges
        config.clamp();
        
        Self::from_config(&config, "random")
    }

    fn mutate(&self, evo_config: &EvolutionConfig) -> Self {
        let mut rng = rand::thread_rng();
        let mut config = self.config.clone();
        
        // Mutations using config ranges
        if rng.gen::<f64>() < evo_config.evolution_settings.mutation_rate {
            config.n_emb = *evo_config.get_n_emb_choices().choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < evo_config.evolution_settings.mutation_rate {
            config.n_ctx = *evo_config.get_n_ctx_choices().choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < evo_config.evolution_settings.mutation_rate {
            config.n_layer = evo_config.random_n_layer();
        }
        if rng.gen::<f64>() < evo_config.evolution_settings.mutation_rate {
            config.n_head = *evo_config.get_n_head_choices().choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < evo_config.evolution_settings.mutation_rate {
            config.n_ff_exp = evo_config.random_n_ff_exp();
        }
        if rng.gen::<f64>() < evo_config.evolution_settings.mutation_rate {
            config.lr = evo_config.random_lr();
        }
        if rng.gen::<f64>() < evo_config.evolution_settings.mutation_rate {
            config.steps = *evo_config.get_steps_choices().choose(&mut rng).unwrap();
        }
        
        // Clamp to valid ranges and fix consistency
        config.clamp();
        
        Self::from_config(&config, "mutant")
    }

    fn crossover(parent1: &Organism, parent2: &Organism, evo_config: &EvolutionConfig) -> Self {
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
        
        // Clamp to valid ranges and fix consistency
        config.clamp();
        
        Self::from_config(&config, "cross")
    }

    fn evaluate(&mut self, gpu_accel: &Arc<microgpt_rust::gpu_accel::GpuAccelerator>) {
        if self.evaluated {
            return;
        }
        
        let start = Instant::now();
        let result = std::panic::catch_unwind(|| {
            train_with_gpu(&self.config, true, gpu_accel)
        });
        
        match result {
            Ok(r) => {
                self.loss = r.final_loss;
                self.evaluated = true;
                let gpu_status = if gpu_accel.is_available() { "GPU" } else { "CPU" };
                println!("[{}] {} | loss={:.4} ({:.1}s)", 
                    gpu_status, self.desc(), self.loss, start.elapsed().as_secs_f64());
            }
            Err(_) => {
                self.loss = f64::MAX;
                self.evaluated = true;
                println!("[{}] {} | PANICKED", 
                    if gpu_accel.is_available() { "GPU" } else { "CPU" }, self.desc());
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

fn train_with_gpu(config: &TrainingConfig, silent: bool, gpu_accel: &Arc<microgpt_rust::gpu_accel::GpuAccelerator>) -> microgpt_rust::TrainingResult {
    // Use regular training with GPU available
    microgpt_rust::train_and_generate(config, silent)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("experiments").ok();
    
    // Load evolution configuration
    let evo_config = EvolutionConfig::load_or_default(CONFIG_FILE);
    
    // Validate the config
    if let Err(e) = evo_config.validate_config() {
        println!("❌ Invalid evolution config: {}", e);
        return Err(e.into());
    }
    
    println!("=== MicroGPT Evolution Engine - GENESIS WITH CONFIG ===");
    println!("📋 Loaded config from: {}", CONFIG_FILE);
    println!("🎯 Target: loss < {:.1}", evo_config.evolution_settings.target_loss);
    println!("👥 Population: {}, Generations: {}", 
        evo_config.evolution_settings.population_size, 
        evo_config.evolution_settings.generations);
    println!("📊 Parameter ranges:");
    println!("  - n_emb: {:?} to {:?}", evo_config.parameter_ranges.n_emb.min, evo_config.parameter_ranges.n_emb.max);
    println!("  - n_ctx: {:?} to {:?}", evo_config.parameter_ranges.n_ctx.min, evo_config.parameter_ranges.n_ctx.max);
    println!("  - n_layer: {:?} to {:?}", evo_config.parameter_ranges.n_layer.min, evo_config.parameter_ranges.n_layer.max);
    println!("  - n_head: {:?} to {:?}", evo_config.parameter_ranges.n_head.min, evo_config.parameter_ranges.n_head.max);
    println!("  - lr: {:.3} to {:.3} (log scale: {})", 
        evo_config.parameter_ranges.lr.min, 
        evo_config.parameter_ranges.lr.max, 
        evo_config.parameter_ranges.lr.log_scale);
    
    // Initialize GPU accelerator
    let gpu_accel = Arc::new(create_gpu_accelerator()?);
    println!("🚀 GPU Available: {}", gpu_accel.is_available());
    
    // Load training data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(INPUT_FILE);

    // Create genesis config from config file
    let mut genesis_config = TrainingConfig {
        n_emb: evo_config.genesis_config.n_emb,
        n_ctx: evo_config.genesis_config.n_ctx,
        n_layer: evo_config.genesis_config.n_layer,
        n_head: evo_config.genesis_config.n_head,
        n_ff_exp: evo_config.genesis_config.n_ff_exp,
        lr: evo_config.genesis_config.lr,
        steps: evo_config.genesis_config.steps,
        input_file: INPUT_FILE.to_string(),
        gen_samples: 1,
        ..Default::default()
    };
    
    // Clamp genesis config to valid ranges
    genesis_config.clamp();
    
    println!("🌱 GENESIS: Starting from config file values");
    println!("Genesis config: Emb:{} Ctx:{} Lay:{} Head:{} FF:{} LR:{:.4} Steps:{}", 
        genesis_config.n_emb, genesis_config.n_ctx, genesis_config.n_layer, 
        genesis_config.n_head, genesis_config.n_ff_exp, genesis_config.lr, genesis_config.steps);
    println!("Validation: {}", if genesis_config.is_reasonable() { "✅ PASS" } else { "❌ FAIL" });

    // Create initial population - start from genesis and mutations
    let mut population = Vec::with_capacity(evo_config.evolution_settings.population_size);
    
    // Add the genesis organism as first member
    population.push(Organism::from_config(&genesis_config, "genesis"));
    
    // Fill the rest with mutations of genesis (not random!)
    while population.len() < evo_config.evolution_settings.population_size {
        population.push(population[0].mutate(&evo_config));
    }

    let mut best_ever = population[0].clone();
    let total_start = Instant::now();

    println!("\n--- Starting Evolution from Genesis (Config-Driven) ---");

    // Evolution loop
    for gen in 0..evo_config.evolution_settings.generations {
        println!("\n--- Generation {}/{} ---", gen + 1, evo_config.evolution_settings.generations);
        
        // Evaluate population
        let gpu_accel_clone = gpu_accel.clone();
        population.par_iter_mut().for_each(|org| {
            if !org.evaluated {
                org.evaluate(&gpu_accel_clone);
            }
        });

        // Sort by loss (handle NaN values)
        population.sort_by(|a, b| {
            if a.loss.is_nan() && b.loss.is_nan() {
                std::cmp::Ordering::Equal
            } else if a.loss.is_nan() {
                std::cmp::Ordering::Greater  // NaN goes to end
            } else if b.loss.is_nan() {
                std::cmp::Ordering::Less   // NaN goes to end
            } else {
                a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

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
            
            // Save the best genome
            println!("💾 Saving genesis best genome...");
            if let Err(e) = best_ever.config.save_genome(best_ever.loss, gen + 1) {
                println!("Failed to save genome: {}", e);
            }
        }

        // Check target
        if best_ever.loss < evo_config.evolution_settings.target_loss {
            println!("🎯 TARGET REACHED! Loss: {:.6}", best_ever.loss);
            break;
        }

        // Create next generation
        let mut next_pop = Vec::with_capacity(evo_config.evolution_settings.population_size);
        
        // Keep elite
        next_pop.push(population[0].clone());
        
        // Add mutations of elite
        for _ in 0..3 {
            next_pop.push(population[0].mutate(&evo_config));
        }
        
        // Add crossovers
        for i in 0..2 {
            let parent2_idx = (i + 1) % population.len().min(3);
            next_pop.push(Organism::crossover(&population[0], &population[parent2_idx], &evo_config));
        }
        
        // Add random diversity from config ranges
        while next_pop.len() < evo_config.evolution_settings.population_size {
            next_pop.push(Organism::random(&evo_config));
        }
        
        population = next_pop;
    }

    println!("\n=== Genesis Evolution Complete (Config-Driven) ===");
    println!("Total Time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!("GPU Used: {}", gpu_accel.is_available());
    println!("Best loss: {:.6}", best_ever.loss);
    println!("Best config: {}", best_ever.desc());
    
    // Show improvement from genesis
    println!("\n🌱 Genesis Progress:");
    println!("  Started from: Emb:{} Ctx:{} Lay:{} Head:{} FF:{} LR:{:.4} Steps:{}", 
        genesis_config.n_emb, genesis_config.n_ctx, genesis_config.n_layer, 
        genesis_config.n_head, genesis_config.n_ff_exp, genesis_config.lr, genesis_config.steps);
    println!("  Evolved to: {}", best_ever.desc());
    println!("  Improvement: {:.6} → {:.6}", 
        if best_ever.loss < f64::MAX { best_ever.loss } else { 0.0 }, 
        best_ever.loss);
    
    Ok(())
}
