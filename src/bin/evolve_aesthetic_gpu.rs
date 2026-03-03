/*
    MicroGPT Aesthetic Evolution Engine - GPU + Config Driven
    
    Optimizes for the *beauty* of generated names rather than raw loss.
    Uses GPU acceleration and config file to control parameter ranges.
    
    Fitness dimensions:
      - Flow: pronounceability (vowel/consonant alternation)
      - Symmetry: palindromes, repeating sub-patterns, pleasant endings
      - Creativity: penalty for memorizing training data; reward novelty
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig, gpu_accel::create_gpu_accelerator, evolution_config::EvolutionConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;
use std::sync::Arc;

const INPUT_FILE: &str = "input.txt";
const CONFIG_FILE: &str = "evolution_config.json";

// --- Genome: hyperparameters as DNA ---

#[derive(Clone, Debug)]
struct Genome {
    config: TrainingConfig,
    fitness: f64,
    names: Vec<String>,
    origin: String,
}

impl Genome {
    fn from_config(config: &TrainingConfig, origin: &str) -> Self {
        Self {
            config: config.clone(),
            fitness: 0.0,
            names: Vec::new(),
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
            gen_samples: 20,  // Generate more samples for aesthetic evaluation
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

    fn crossover(parent1: &Genome, parent2: &Genome, evo_config: &EvolutionConfig) -> Self {
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

    // Train a MicroGPT and evaluate the aesthetic quality of its output
    fn evaluate(&mut self, training_data: &HashSet<String>, gpu_accel: &Arc<microgpt_rust::gpu_accel::GpuAccelerator>) {
        if self.fitness != 0.0 && !self.names.is_empty() {
            return;
        }

        let start = Instant::now();
        let result = std::panic::catch_unwind(|| {
            train_and_generate(&self.config, true)
        });
        
        match result {
            Ok(r) => {
                let score = calculate_fitness(&r.names, training_data);
                self.names = r.names;
                self.fitness = score;
                let gpu_status = if gpu_accel.is_available() { "GPU" } else { "CPU" };
                println!("[{}] {} | fitness={:.4} ({:.1}s)", 
                    gpu_status, self.desc(), self.fitness, start.elapsed().as_secs_f64());
            }
            Err(_) => {
                self.fitness = -100.0;
                self.names.clear();
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

// --- Fitness: The Judge ---
// Evaluates generated names on three aesthetic dimensions.

fn calculate_fitness(names: &[String], training_data: &HashSet<String>) -> f64 {
    if names.is_empty() { return -100.0; }

    let mut total_score = 0.0;
    let mut valid_count = 0;

    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) { continue; }

        let s_flow = score_flow(&name);
        let s_sym = score_symmetry(&name);
        let s_creat = score_creativity(&name, training_data);

        // Creativity weighted 2x — novelty matters most
        total_score += s_flow * 1.0 + s_sym * 1.2 + s_creat * 2.0;
        valid_count += 1;
    }

    if valid_count == 0 { return -100.0; }
    total_score / valid_count as f64
}

// Flow: penalize unpronounceable clusters (3+ consecutive vowels or consonants)
fn score_flow(name: &str) -> f64 {
    let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u', 'y'].iter().cloned().collect();
    let mut score = 0.0;
    let mut cons_v = 0;
    let mut cons_c = 0;

    for c in name.chars() {
        if vowels.contains(&c) {
            cons_v += 1;
            cons_c = 0;
        } else {
            cons_c += 1;
            cons_v = 0;
        }
        if cons_v > 2 || cons_c > 2 {
            score -= 1.0;
        }
    }

    // Bonus for ideal name length (4-8 characters)
    if name.len() >= 4 && name.len() <= 8 {
        score += 0.5;
    }
    score
}

// Symmetry: reward palindromes, repeating halves, pleasant endings
fn score_symmetry(name: &str) -> f64 {
    let mut score = 0.0;
    let chars: Vec<char> = name.chars().collect();

    // Perfect palindrome
    if name.len() > 3 && chars.iter().eq(chars.iter().rev()) {
        score += 2.0;
    }

    // Repeating first half (e.g., "mama")
    if name.len() >= 4 {
        let mid = name.len() / 2;
        if name[..mid] == name[mid..mid*2] {
            score += 1.5;
        }
    }

    // Pleasant ending sounds
    if name.ends_with('a') || name.ends_with('n') || name.ends_with('y') {
        score += 0.2;
    }
    score
}

// Creativity: heavy penalty for memorizing training data
fn score_creativity(name: &str, training_data: &HashSet<String>) -> f64 {
    if training_data.contains(name) {
        -5.0
    } else {
        1.0
    }
}

// --- Main Evolution Loop ---

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MicroGPT Aesthetic Evolution Engine - GPU + Config ===");
    
    // Load evolution configuration
    let evo_config = EvolutionConfig::load_or_default(CONFIG_FILE);
    
    // Validate the config
    if let Err(e) = evo_config.validate_config() {
        println!("❌ Invalid evolution config: {}", e);
        return Err(e.into());
    }
    
    println!("📋 Loaded config from: {}", CONFIG_FILE);
    println!("🎯 Target: fitness maximization (beauty score)");
    println!("👥 Population: {}, Generations: {}", 
        evo_config.evolution_settings.population_size, 
        evo_config.evolution_settings.generations);
    println!("📊 Parameter ranges:");
    println!("  - n_emb: {:?} to {:?}", evo_config.parameter_ranges.n_emb.min, evo_config.parameter_ranges.n_emb.max);
    println!("  - n_ctx: {:?} to {:?}", evo_config.parameter_ranges.n_ctx.min, evo_config.parameter_ranges.n_ctx.max);
    println!("  - n_layer: {:?} to {:?}", evo_config.parameter_ranges.n_layer.min, evo_config.parameter_ranges.n_layer.max);
    println!("  - n_head: {:?} to {:?}", evo_config.parameter_ranges.n_head.min, evo_config.parameter_ranges.n_head.max);

    // Initialize GPU accelerator
    let gpu_accel = Arc::new(create_gpu_accelerator()?);
    println!("🚀 GPU Available: {}", gpu_accel.is_available());

    // Load training data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    // Create initial population - start from genesis config and mutations
    let mut population = Vec::with_capacity(evo_config.evolution_settings.population_size);
    
    // Add genesis organism
    let mut genesis_config = TrainingConfig {
        n_emb: evo_config.genesis_config.n_emb,
        n_ctx: evo_config.genesis_config.n_ctx,
        n_layer: evo_config.genesis_config.n_layer,
        n_head: evo_config.genesis_config.n_head,
        n_ff_exp: evo_config.genesis_config.n_ff_exp,
        lr: evo_config.genesis_config.lr,
        steps: evo_config.genesis_config.steps,
        input_file: INPUT_FILE.to_string(),
        gen_samples: 20,
        ..Default::default()
    };
    genesis_config.clamp();
    
    population.push(Genome::from_config(&genesis_config, "genesis"));
    
    // Fill the rest with mutations of genesis
    while population.len() < evo_config.evolution_settings.population_size {
        population.push(population[0].mutate(&evo_config));
    }

    let total_start = Instant::now();
    let mut best_ever = population[0].clone();

    println!("\n--- Starting Aesthetic Evolution (Config-Driven) ---");

    // Evolution loop
    for gen in 0..evo_config.evolution_settings.generations {
        println!("\n=== Generation {}/{} ===", gen + 1, evo_config.evolution_settings.generations);
        
        // Evaluate all organisms in parallel
        let gpu_accel_clone = gpu_accel.clone();
        population.par_iter_mut().for_each(|genome| {
            genome.evaluate(&training_data, &gpu_accel_clone);
        });

        // Sort by fitness (higher = more beautiful)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Print results
        println!("\nResults:");
        for (i, g) in population.iter().take(5).enumerate() {
            println!("  #{}: {:.4} | {}", i + 1, g.fitness, g.desc());
            if !g.names.is_empty() {
                println!("      Samples: {}", g.names.iter().take(3).cloned().collect::<Vec<_>>().join(", "));
            }
        }

        let best_this_gen = population[0].clone();
        println!("\n>> Gen {} Champion: {}", gen + 1, best_this_gen.desc());
        println!(">> Score: {:.4}", best_this_gen.fitness);

        // Update best ever
        if best_this_gen.fitness > best_ever.fitness {
            println!("🎉 NEW BEAUTY CHAMPION! {:.4} -> {:.4}", best_ever.fitness, best_this_gen.fitness);
            best_ever = best_this_gen.clone();
            
            // Save the best genome
            println!("💾 Saving aesthetic champion...");
            if let Err(e) = best_ever.config.save_genome(best_ever.fitness, gen + 1) {
                println!("Failed to save genome: {}", e);
            }
        }

        // Breed next generation
        if gen < evo_config.evolution_settings.generations - 1 {
            let mut new_pop = Vec::with_capacity(evo_config.evolution_settings.population_size);
            
            // Keep elite
            new_pop.push(population[0].clone());
            
            // Add mutations of elite
            for _ in 0..3 {
                new_pop.push(population[0].mutate(&evo_config));
            }
            
            // Add crossovers
            for i in 0..2 {
                let parent2_idx = (i + 1) % population.len().min(3);
                new_pop.push(Genome::crossover(&population[0], &population[parent2_idx], &evo_config));
            }
            
            // Add random diversity
            while new_pop.len() < evo_config.evolution_settings.population_size {
                new_pop.push(Genome::random(&evo_config));
            }
            
            population = new_pop;
        }
    }

    println!("\n=== Aesthetic Evolution Complete ===");
    println!("Total Time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!("GPU Used: {}", gpu_accel.is_available());
    println!("Best fitness: {:.4}", best_ever.fitness);
    println!("Best config: {}", best_ever.desc());
    
    if !best_ever.names.is_empty() {
        println!("\n🎨 Champion Generated Names:");
        for (i, name) in best_ever.names.iter().take(10).enumerate() {
            println!("  {}: {}", i + 1, name);
        }
    }
    
    Ok(())
}
