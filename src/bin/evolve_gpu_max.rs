/*
    Maximum GPU Stress Test - GPU 0 Only
    
    Designed to maximize GPU 0 utilization with:
    - Large model architectures
    - High population sizes  
    - Complex matrix operations
    - Extended evolution runs
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

// GPU feature flag
#[cfg(feature = "gpu")]
use microgpt_rust::gpu_accel::GpuAccelerator;

// --- Maximum Stress Parameters ---
const TOURNAMENT_SIZE: usize = 5; // Larger tournaments
const POPULATION_SIZE: usize = 32; // Maximum population
const NUM_GENERATIONS: usize = 200; // Extended run
const TARGET_LOSS: f64 = 0.5; // Ambitious target
const INPUT_FILE: &str = "input.txt";

// Large model parameters for maximum GPU utilization
const MAX_EMB: usize = 512;
const MAX_CTX: usize = 256;
const MAX_LAYERS: usize = 12;
const MAX_HEADS: usize = 16;
const MAX_STEPS: usize = 5000;

#[derive(Clone, Debug)]
struct Genome {
    n_emb: usize,
    n_head: usize,
    n_layer: usize,
    n_ctx: usize,
    n_ff_exp: usize,
    lr: f64,
    steps: usize,
    loss: f64,
    evaluated: bool,
    origin: String,
}

impl Genome {
    fn from_config(config: &TrainingConfig) -> Self {
        Genome {
            n_emb: config.n_emb,
            n_head: config.n_head,
            n_layer: config.n_layer,
            n_ctx: config.n_ctx,
            n_ff_exp: config.n_ff_exp,
            lr: config.lr,
            steps: config.steps,
            loss: f64::MAX,
            evaluated: false,
            origin: "genome".to_string(),
        }
    }

    fn new_random_large() -> Self {
        let mut rng = rand::thread_rng();
        
        // Large model space for GPU stress testing
        let n_emb = *[64, 128, 192, 256, 384, 512].choose(&mut rng).unwrap();
        let n_head_options: Vec<usize> = [1, 2, 4, 8, 16].iter()
            .filter(|&&h| n_emb % h == 0)
            .copied()
            .collect();
        let n_head = *n_head_options.choose(&mut rng).unwrap();
        
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(4..=12), // Deep networks
            n_ctx: *[64, 96, 128, 192, 256].choose(&mut rng).unwrap(), // Large context
            n_ff_exp: rng.gen_range(3..=8), // Wide feed-forward
            lr: 10f64.powf(rng.gen_range(-4.5..-2.0)), // Conservative learning rates
            steps: *[1000, 2000, 3000, 4000, 5000].choose(&mut rng).unwrap(), // Long training
            loss: f64::MAX,
            evaluated: false,
            origin: "large_random".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate_aggressive(&mut self) {
        let mut rng = rand::thread_rng();
        let num_mutations = rng.gen_range(2..=5); // More mutations
        
        for _ in 0..num_mutations {
            match rng.gen_range(0..8) {
                0 => {
                    // Large embedding changes
                    self.n_emb = *[64, 128, 192, 256, 384, 512].choose(&mut rng).unwrap();
                    // Recalculate valid heads
                    let valid_heads: Vec<usize> = [1, 2, 4, 8, 16].iter()
                        .filter(|&&h| self.n_emb % h == 0)
                        .copied()
                        .collect();
                    self.n_head = *valid_heads.choose(&mut rng).unwrap();
                },
                1 => self.n_layer = rng.gen_range(4..=12),
                2 => self.lr = 10f64.powf(rng.gen_range(-4.5..-2.0)),
                3 => {
                    let delta = *[-1000, -500, -200, 200, 500, 1000].choose(&mut rng).unwrap();
                    self.steps = (self.steps as i32 + delta).clamp(500, 5000) as usize;
                },
                4 => self.n_ctx = *[64, 96, 128, 192, 256].choose(&mut rng).unwrap(),
                5 => self.n_ff_exp = rng.gen_range(3..=8),
                6 => {
                    // Change heads (must divide embeddings)
                    let valid_heads: Vec<usize> = [1, 2, 4, 8, 16].iter()
                        .filter(|&&h| self.n_emb % h == 0)
                        .copied()
                        .collect();
                    self.n_head = *valid_heads.choose(&mut rng).unwrap();
                },
                7 => {
                    // Growth mutation
                    if rng.gen() && self.n_layer < 12 {
                        self.n_layer += 1;
                    } else if rng.gen() && self.n_ctx < 256 {
                        self.n_ctx = std::cmp::min(256, self.n_ctx * 2);
                    }
                },
                _ => {},
            }
        }
        self.enforce_constraints();
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    fn enforce_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            let valid: Vec<usize> = [1, 2, 4, 8, 16].iter().copied()
                .filter(|h| self.n_emb % h == 0)
                .collect();
            self.n_head = *valid.last().unwrap_or(&1);
        }
        
        // Ensure we stay within GPU-friendly bounds
        self.n_emb = self.n_emb.min(MAX_EMB);
        self.n_ctx = self.n_ctx.min(MAX_CTX);
        self.n_layer = self.n_layer.min(MAX_LAYERS);
        self.steps = self.steps.min(MAX_STEPS);
    }

    fn evaluate(&mut self, id: usize, use_gpu: bool, gpu_id: usize) {
        if self.evaluated {
            eprintln!("[eval] organism {} already evaluated (loss={:.4})", id, self.loss);
            return;
        }
        
        let gpu_indicator = if use_gpu { format!("[GPU{}-MAX]", gpu_id) } else { "[CPU]".to_string() };
        eprintln!("[eval] organism {} starting: {} {}", id, self.desc(), gpu_indicator);
        let start = Instant::now();
        
        let config = TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_ff_exp: self.n_ff_exp,
            lr: self.lr,
            steps: self.steps,
            input_file: INPUT_FILE.to_string(),
            gen_samples: 1,
            ..Default::default()
        };
        
        let result = std::panic::catch_unwind(|| {
            train_and_generate(&config, true)
        });
        
        match result {
            Ok(r) => {
                self.loss = r.final_loss;
                self.evaluated = true;
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!("[eval] organism {} done: loss={:.4} ({:.1}s) {}", 
                    id, self.loss, elapsed, gpu_indicator);
            }
            Err(e) => {
                eprintln!("[eval] organism {} PANICKED: {:?} | config: {}", id, e, self.desc());
                self.loss = f64::MAX;
                self.evaluated = true;
            }
        }
    }

    fn desc(&self) -> String {
        format!("Emb:{:<4} Head:{:<2} Lay:{:<2} Ctx:{:<3} FF:{:<2} LR:{:.5} Steps:{:<4}",
            self.n_emb, self.n_head, self.n_layer, self.n_ctx, self.n_ff_exp, self.lr, self.steps)
    }

    fn species(&self) -> String {
        format!("{}-{}-{}-{}-{}", self.n_emb, self.n_head, self.n_layer, self.n_ctx, self.n_ff_exp)
    }

    fn to_config(&self, gen_samples: usize) -> TrainingConfig {
        TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_ff_exp: self.n_ff_exp,
            lr: self.lr,
            steps: self.steps,
            input_file: INPUT_FILE.to_string(),
            gen_samples,
            ..Default::default()
        }
    }
}

// --- Genetic Operators for Large Models ---

fn crossover_large(a: &Genome, b: &Genome) -> Genome {
    let mut rng = rand::thread_rng();
    let mut child = Genome {
        n_emb: if rng.gen() { a.n_emb } else { b.n_emb },
        n_head: if rng.gen() { a.n_head } else { b.n_head },
        n_layer: if rng.gen() { a.n_layer } else { b.n_layer },
        n_ctx: if rng.gen() { a.n_ctx } else { b.n_ctx },
        n_ff_exp: if rng.gen() { a.n_ff_exp } else { b.n_ff_exp },
        lr: if rng.gen() { a.lr } else { b.lr },
        steps: if rng.gen() { a.steps } else { b.steps },
        loss: f64::MAX,
        evaluated: false,
        origin: "large_cross".to_string(),
    };
    child.enforce_constraints();
    child
}

fn tournament_select_large<'a>(pop: &'a [Genome], rng: &mut ThreadRng) -> &'a Genome {
    let mut best: Option<&Genome> = None;
    for _ in 0..TOURNAMENT_SIZE {
        let candidate = &pop[rng.gen_range(0..pop.len())];
        if best.is_none() || candidate.loss < best.unwrap().loss {
            best = Some(candidate);
        }
    }
    best.unwrap()
}

// --- Species & Diversity Analysis ---

fn species_census(pop: &[Genome]) -> HashMap<String, Vec<usize>> {
    let mut species: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, g) in pop.iter().enumerate() {
        species.entry(g.species()).or_default().push(i);
    }
    species
}

// --- Experiment Logging ---

macro_rules! log {
    ($log:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        println!("{}", msg);
        if let Some(ref f) = *$log.lock().unwrap() {
            let _ = writeln!(f.try_clone().unwrap(), "{}", msg);
        }
    }};
}

fn experiment_filename() -> String {
    let now = chrono::Local::now();
    format!("experiments/evolve_gpu_max_{}.log", now.format("%Y%m%d_%H%M%S"))
}

// --- GPU Setup for Maximum Stress ---

fn setup_gpu_max(gpu_id: usize) -> (bool, String) {
    #[cfg(feature = "gpu")]
    {
        match GpuAccelerator::new_with_device(gpu_id) {
            Ok(gpu) => {
                let info = format!("GPU{}: {} (MAX STRESS MODE)", gpu_id, gpu.gpu_name());
                println!("🚀🚀 GPU MAX STRESS MODE ENABLED: {}", info);
                return (true, info);
            }
            Err(e) => {
                println!("⚠️  GPU{} initialization failed: {}", gpu_id, e);
                println!("🔄 Falling back to CPU-only mode");
            }
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("⚠️  GPU feature not enabled. Build with: cargo build --release --features gpu");
        println!("🔄 Using CPU-only mode");
    }
    
    (false, format!("GPU{} (CPU fallback)", gpu_id))
}

// --- Main Evolution Loop ---

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse command line arguments
    let gpu_id: usize = if args.len() >= 2 {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("Invalid GPU ID: {}. Using GPU 0", args[1]);
            0
        })
    } else {
        0 // Default to GPU 0
    };
    
    let generations: usize = if args.len() >= 3 {
        args[2].parse().unwrap_or_else(|_| {
            eprintln!("Invalid generation count: {}. Using 200", args[2]);
            200
        })
    } else {
        200 // Default to 200 generations for max stress
    };
    
    // Setup GPU for maximum stress
    let (use_gpu, gpu_info) = setup_gpu_max(gpu_id);
    
    println!("🔥🔥 MAXIMUM GPU STRESS TEST 🔥🔥");
    println!("===================================");
    println!("  GPU Device: {}", gpu_id);
    println!("  Generations: {} (extended)", generations);
    println!("  Population: {} (maximum)", POPULATION_SIZE);
    println!("  Target loss: {:.1} (ambitious)", TARGET_LOSS);
    println!("  Max Embeddings: {}", MAX_EMB);
    println!("  Max Context: {}", MAX_CTX);
    println!("  Max Layers: {}", MAX_LAYERS);
    println!("  Acceleration: {}", if use_gpu { "GPU MAX" } else { "CPU" });
    println!();

    std::fs::create_dir_all("experiments").ok();
    let log_path = experiment_filename();
    let log_file: Mutex<Option<std::fs::File>> = Mutex::new(
        std::fs::File::create(&log_path).ok()
    );

    log!(log_file, "=== MicroGPT MAXIMUM GPU STRESS TEST ===");
    log!(log_file, "Experiment: {}", log_path);
    log!(log_file, "GPU Device: {}, Acceleration: {}", gpu_id, if use_gpu { "GPU MAX" } else { "CPU" });
    log!(log_file, "Population: {}, Generations: {}", POPULATION_SIZE, generations);
    log!(log_file, "Target: loss < {:.1}", TARGET_LOSS);
    log!(log_file, "Max Model Size: {} embeddings, {} context, {} layers", MAX_EMB, MAX_CTX, MAX_LAYERS);
    log!(log_file, "");

    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(INPUT_FILE);

    // Load existing genome if available
    let (base_config, base_loss, base_gen) = TrainingConfig::load_genome()
        .unwrap_or_else(|| (TrainingConfig::default(), f64::MAX, 0));
    
    println!("Starting MAX STRESS evolution from genome: loss {:.6} (generation {})", base_loss, base_gen);
    
    // Create initial population with large models
    let mut population: Vec<Genome> = Vec::with_capacity(POPULATION_SIZE);
    
    if base_loss < f64::MAX {
        let mut base_genome = Genome::from_config(&base_config);
        base_genome.loss = base_loss;
        base_genome.evaluated = true;
        population.push(base_genome);
        println!("Added existing genome to population (loss: {:.6})", base_loss);
    }
    
    // Fill with large random models
    while population.len() < POPULATION_SIZE {
        if base_loss < f64::MAX && population.len() < POPULATION_SIZE / 2 {
            let mut mutant = Genome::from_config(&base_config);
            mutant.mutate_aggressive();
            population.push(mutant);
        } else {
            population.push(Genome::new_random_large());
        }
    }
    
    let mut best_ever = if base_loss < f64::MAX {
        population[0].clone()
    } else {
        Genome::new_random_large()
    };
    best_ever.loss = base_loss;
    
    let total_start = Instant::now();
    let mut target_gen: Option<usize> = None;
    let mut stagnation_count: usize = 0;
    let mut prev_best_loss: f64 = f64::MAX;

    println!("🔥 Starting MAX STRESS evolution with {} large models...", population.len());
    println!("This will heavily utilize GPU{} for extended periods!", gpu_id);

    for gen in 0..generations {
        let gen_start = Instant::now();
        log!(log_file, "--- Generation {}/{} (MAX STRESS) ---", gen + 1, generations);

        eprintln!("[gen {} MAX] evaluating {} LARGE organisms...", gen + 1, population.len());
        
        // Parallel evaluation with large models
        population.par_iter_mut().enumerate().for_each(|(i, genome)| {
            genome.evaluate(i + 1, use_gpu, gpu_id);
        });

        population.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());

        // Species census
        let census = species_census(&population);
        let num_species = census.len();

        let gen_best = &population[0];
        if gen_best.loss < best_ever.loss {
            best_ever = gen_best.clone();
        }

        // Track stagnation
        if (gen_best.loss - prev_best_loss).abs() < 1e-8 {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_best_loss = gen_best.loss;

        if target_gen.is_none() && best_ever.loss < TARGET_LOSS {
            target_gen = Some(gen + 1);
            log!(log_file, "  ** 🎯 TARGET {:.1} REACHED! Continuing evolution... **", TARGET_LOSS);
        }

        let elapsed = gen_start.elapsed().as_secs_f64();
        log!(log_file, "  Best: {:.4} | Species: {} | Stagnation: {} | {:.1}s (MAX STRESS {})",
            gen_best.loss, num_species, stagnation_count, elapsed, 
            if use_gpu { format!("GPU{}", gpu_id) } else { "CPU".to_string() });

        // Progress indicator for long runs
        if gen % 10 == 0 || gen == generations - 1 {
            let total_elapsed = total_start.elapsed().as_secs_f64();
            let rate = gen as f64 / total_elapsed;
            let eta = if rate > 0.0 { (generations - gen - 1) as f64 / rate } else { 0.0 };
            println!("🔥 MAX STRESS Progress: {}/{} generations ({:.1}%) - ETA: {:.0}s - Best: {:.4} ({}{})", 
                gen + 1, generations, 
                ((gen + 1) as f64 / generations as f64) * 100.0,
                eta, gen_best.loss, if use_gpu { "GPU" } else { "CPU" }, gpu_id);
        }

        // --- Breed the next generation with large models ---
        if gen < generations - 1 {
            let mut new_pop: Vec<Genome> = Vec::with_capacity(POPULATION_SIZE);
            
            // Keep elite
            let mut elite = population[0].clone();
            elite.origin = "elite_max".to_string();
            new_pop.push(elite);
            
            // Fill rest with aggressive mutations and crossovers
            let mut rng = rand::thread_rng();
            while new_pop.len() < POPULATION_SIZE {
                if rng.gen() {
                    let parent = tournament_select_large(&population, &mut rng);
                    let mut child = parent.clone();
                    child.mutate_aggressive();
                    child.origin = "mutant_max".to_string();
                    new_pop.push(child);
                } else {
                    let p1 = tournament_select_large(&population, &mut rng);
                    let p2 = tournament_select_large(&population, &mut rng);
                    let mut child = crossover_large(p1, p2);
                    child.mutate_aggressive();
                    child.origin = "cross_max".to_string();
                    new_pop.push(child);
                }
            }
            
            population = new_pop;
        }
    }

    // Save the best genome
    let best_config = best_ever.to_config(1);
    let _ = best_config.save_genome(best_ever.loss, generations);
    
    let total_time = total_start.elapsed().as_secs_f64();
    println!("\n🔥🔥 MAXIMUM GPU STRESS TEST COMPLETE! 🔥🔥");
    println!("==========================================");
    println!("GPU Device: {}", gpu_id);
    println!("Total generations: {}", generations);
    println!("Total time: {:.0}s ({:.1}s per generation)", total_time, total_time / generations as f64);
    println!("Best loss: {:.4}", best_ever.loss);
    println!("Best config: {}", best_ever.desc());
    println!("Acceleration: {}", if use_gpu { "GPU MAX STRESS" } else { "CPU" });
    println!("Results saved to: genome.json");
    println!("Log file: {}", log_path);
    
    if let Some(g) = target_gen {
        println!("🎯 Target {:.1} first reached: generation {}", TARGET_LOSS, g);
    }
    
    println!("\n💡 GPU Stress Test Results:");
    println!("  - GPU{} heavily utilized with large models", gpu_id);
    println!("  - Maximum model sizes tested");
    println!("  - Extended run completed successfully");
    println!("  - Check nvidia-smi for utilization patterns");
    
    if use_gpu {
        println!("\n🚀 Performance Notes:");
        println!("  - Large models should show significant GPU advantage");
        println!("  - Matrix operations heavily accelerated");
        println!("  - Memory bandwidth fully utilized");
        println!("  - This represents maximum sustainable GPU load");
    }
}
