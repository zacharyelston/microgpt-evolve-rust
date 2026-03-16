/*
    True GPU Matrix Stress Test
    
    Focuses on GPU-accelerated matrix operations by using
    models that maximize matrix computation vs scalar operations.
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

// --- GPU Matrix-Optimized Parameters ---
const TOURNAMENT_SIZE: usize = 3;
const POPULATION_SIZE: usize = 24; // High but manageable
const NUM_GENERATIONS: usize = 100; // Focus on quality over quantity
const TARGET_LOSS: f64 = 0.8;
const INPUT_FILE: &str = "input.txt";

// GPU-optimized model parameters
const OPTIMAL_EMB: usize = 256; // Good for GPU matrix ops
const OPTIMAL_CTX: usize = 128; // Good for GPU memory
const OPTIMAL_LAYERS: usize = 8; // Deep but not too deep
const OPTIMAL_HEADS: usize = 8; // Good parallelism

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

    fn new_gpu_optimized() -> Self {
        let mut rng = rand::thread_rng();
        
        // Focus on GPU-friendly architectures
        let n_emb = *[128, 192, 256, 384].choose(&mut rng).unwrap();
        let n_head_options: Vec<usize> = [2, 4, 8].iter()
            .filter(|&&h| n_emb % h == 0)
            .copied()
            .collect();
        let n_head = *n_head_options.choose(&mut rng).unwrap();
        
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(4..=10),
            n_ctx: *[64, 96, 128, 192].choose(&mut rng).unwrap(),
            n_ff_exp: rng.gen_range(3..=6),
            lr: 10f64.powf(rng.gen_range(-4.0..-2.5)),
            steps: *[1500, 2000, 2500, 3000].choose(&mut rng).unwrap(),
            loss: f64::MAX,
            evaluated: false,
            origin: "gpu_optimized".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate_gpu_focused(&mut self) {
        let mut rng = rand::thread_rng();
        let num_mutations = rng.gen_range(1..=3);
        
        for _ in 0..num_mutations {
            match rng.gen_range(0..7) {
                0 => {
                    // GPU-friendly embedding changes
                    self.n_emb = *[128, 192, 256, 384].choose(&mut rng).unwrap();
                    let valid_heads: Vec<usize> = [2, 4, 8].iter()
                        .filter(|&&h| self.n_emb % h == 0)
                        .copied()
                        .collect();
                    self.n_head = *valid_heads.choose(&mut rng).unwrap();
                },
                1 => self.n_layer = rng.gen_range(4..=10),
                2 => self.lr = 10f64.powf(rng.gen_range(-4.0..-2.5)),
                3 => {
                    let delta = *[-500, -250, -100, 100, 250, 500].choose(&mut rng).unwrap();
                    self.steps = (self.steps as i32 + delta).clamp(1000, 3000) as usize;
                },
                4 => self.n_ctx = *[64, 96, 128, 192].choose(&mut rng).unwrap(),
                5 => self.n_ff_exp = rng.gen_range(3..=6),
                6 => {
                    let valid_heads: Vec<usize> = [2, 4, 8].iter()
                        .filter(|&&h| self.n_emb % h == 0)
                        .copied()
                        .collect();
                    self.n_head = *valid_heads.choose(&mut rng).unwrap();
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
            let valid: Vec<usize> = [2, 4, 8].iter().copied()
                .filter(|h| self.n_emb % h == 0)
                .collect();
            self.n_head = *valid.last().unwrap_or(&2);
        }
        
        // Keep in GPU-optimal range
        self.n_emb = self.n_emb.clamp(128, 384);
        self.n_ctx = self.n_ctx.clamp(64, 192);
        self.n_layer = self.n_layer.clamp(4, 10);
        self.steps = self.steps.clamp(1000, 3000);
    }

    fn evaluate(&mut self, id: usize, use_gpu: bool, gpu_id: usize) {
        if self.evaluated {
            eprintln!("[eval] organism {} already evaluated (loss={:.4})", id, self.loss);
            return;
        }
        
        let gpu_indicator = if use_gpu { format!("[GPU{}-MATRIX]", gpu_id) } else { "[CPU]".to_string() };
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
        format!("Emb:{:<3} Head:{:<1} Lay:{:<1} Ctx:{:<3} FF:{:<1} LR:{:.4} Steps:{:<4}",
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

// --- Genetic Operators ---

fn crossover_gpu(a: &Genome, b: &Genome) -> Genome {
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
        origin: "gpu_cross".to_string(),
    };
    child.enforce_constraints();
    child
}

fn tournament_select_gpu<'a>(pop: &'a [Genome], rng: &mut ThreadRng) -> &'a Genome {
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
    format!("experiments/evolve_gpu_matrix_{}.log", now.format("%Y%m%d_%H%M%S"))
}

// --- GPU Setup for Matrix Operations ---

fn setup_gpu_matrix(gpu_id: usize) -> (bool, String) {
    #[cfg(feature = "gpu")]
    {
        match GpuAccelerator::new_with_device(gpu_id) {
            Ok(gpu) => {
                let info = format!("GPU{}: {} (MATRIX MODE)", gpu_id, gpu.gpu_name());
                println!("🚀 GPU MATRIX MODE ENABLED: {}", info);
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
            eprintln!("Invalid generation count: {}. Using 100", args[2]);
            100
        })
    } else {
        100 // Default to 100 generations
    };
    
    // Setup GPU for matrix operations
    let (use_gpu, gpu_info) = setup_gpu_matrix(gpu_id);
    
    println!("🧬 GPU Matrix-Optimized Evolution");
    println!("=================================");
    println!("  GPU Device: {}", gpu_id);
    println!("  Generations: {}", generations);
    println!("  Population: {}", POPULATION_SIZE);
    println!("  Target loss: {:.1}", TARGET_LOSS);
    println!("  Model Range: {}-{} embeddings", 128, 384);
    println!("  Context Range: {}-{} tokens", 64, 192);
    println!("  Acceleration: {}", if use_gpu { "GPU MATRIX" } else { "CPU" });
    println!();

    std::fs::create_dir_all("experiments").ok();
    let log_path = experiment_filename();
    let log_file: Mutex<Option<std::fs::File>> = Mutex::new(
        std::fs::File::create(&log_path).ok()
    );

    log!(log_file, "=== MicroGPT GPU Matrix-Optimized Evolution ===");
    log!(log_file, "Experiment: {}", log_path);
    log!(log_file, "GPU Device: {}, Acceleration: {}", gpu_id, if use_gpu { "GPU MATRIX" } else { "CPU" });
    log!(log_file, "Population: {}, Generations: {}", POPULATION_SIZE, generations);
    log!(log_file, "Target: loss < {:.1}", TARGET_LOSS);
    log!(log_file, "Model Range: {}-{} embeddings, {}-{} context", 128, 384, 64, 192);
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
    
    println!("Starting GPU-optimized evolution from genome: loss {:.6} (generation {})", base_loss, base_gen);
    
    // Create initial population with GPU-optimized models
    let mut population: Vec<Genome> = Vec::with_capacity(POPULATION_SIZE);
    
    if base_loss < f64::MAX {
        let mut base_genome = Genome::from_config(&base_config);
        base_genome.loss = base_loss;
        base_genome.evaluated = true;
        population.push(base_genome);
        println!("Added existing genome to population (loss: {:.6})", base_loss);
    }
    
    // Fill with GPU-optimized models
    while population.len() < POPULATION_SIZE {
        if base_loss < f64::MAX && population.len() < POPULATION_SIZE / 2 {
            let mut mutant = Genome::from_config(&base_config);
            mutant.mutate_gpu_focused();
            population.push(mutant);
        } else {
            population.push(Genome::new_gpu_optimized());
        }
    }
    
    let mut best_ever = if base_loss < f64::MAX {
        population[0].clone()
    } else {
        Genome::new_gpu_optimized()
    };
    best_ever.loss = base_loss;
    
    let total_start = Instant::now();
    let mut target_gen: Option<usize> = None;
    let mut stagnation_count: usize = 0;
    let mut prev_best_loss: f64 = f64::MAX;

    println!("🚀 Starting GPU matrix-optimized evolution with {} organisms...", population.len());
    println!("This focuses on GPU-accelerated matrix operations!");

    for gen in 0..generations {
        let gen_start = Instant::now();
        log!(log_file, "--- Generation {}/{} (GPU MATRIX) ---", gen + 1, generations);

        eprintln!("[gen {} MATRIX] evaluating {} organisms...", gen + 1, population.len());
        
        // Parallel evaluation with GPU-optimized models
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
            log!(log_file, "  ** 🎯 TARGET {:.1} REACHED! **", TARGET_LOSS);
        }

        let elapsed = gen_start.elapsed().as_secs_f64();
        log!(log_file, "  Best: {:.4} | Species: {} | Stagnation: {} | {:.1}s (GPU MATRIX {})",
            gen_best.loss, num_species, stagnation_count, elapsed, gpu_id);

        // Progress indicator
        if gen % 10 == 0 || gen == generations - 1 {
            let total_elapsed = total_start.elapsed().as_secs_f64();
            let rate = gen as f64 / total_elapsed;
            let eta = if rate > 0.0 { (generations - gen - 1) as f64 / rate } else { 0.0 };
            println!("🚀 GPU Matrix Progress: {}/{} generations ({:.1}%) - ETA: {:.0}s - Best: {:.4} ({}{})", 
                gen + 1, generations, 
                ((gen + 1) as f64 / generations as f64) * 100.0,
                eta, gen_best.loss, if use_gpu { "GPU" } else { "CPU" }, gpu_id);
        }

        // --- Breed the next generation ---
        if gen < generations - 1 {
            let mut new_pop: Vec<Genome> = Vec::with_capacity(POPULATION_SIZE);
            
            // Keep elite
            let mut elite = population[0].clone();
            elite.origin = "elite_matrix".to_string();
            new_pop.push(elite);
            
            // Fill rest with GPU-focused mutations and crossovers
            let mut rng = rand::thread_rng();
            while new_pop.len() < POPULATION_SIZE {
                if rng.gen() {
                    let parent = tournament_select_gpu(&population, &mut rng);
                    let mut child = parent.clone();
                    child.mutate_gpu_focused();
                    child.origin = "mutant_matrix".to_string();
                    new_pop.push(child);
                } else {
                    let p1 = tournament_select_gpu(&population, &mut rng);
                    let p2 = tournament_select_gpu(&population, &mut rng);
                    let mut child = crossover_gpu(p1, p2);
                    child.mutate_gpu_focused();
                    child.origin = "cross_matrix".to_string();
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
    println!("\n🚀 GPU Matrix-Optimized Evolution Complete!");
    println!("==========================================");
    println!("GPU Device: {}", gpu_id);
    println!("Total generations: {}", generations);
    println!("Total time: {:.0}s ({:.1}s per generation)", total_time, total_time / generations as f64);
    println!("Best loss: {:.4}", best_ever.loss);
    println!("Best config: {}", best_ever.desc());
    println!("Acceleration: {}", if use_gpu { "GPU MATRIX" } else { "CPU" });
    println!("Results saved to: genome.json");
    println!("Log file: {}", log_path);
    
    if let Some(g) = target_gen {
        println!("🎯 Target {:.1} reached: generation {}", TARGET_LOSS, g);
    }
    
    if use_gpu {
        println!("\n💡 GPU Matrix Performance:");
        println!("  - Focused on GPU-accelerated matrix operations");
        println!("  - Optimized model sizes for GPU efficiency");
        println!("  - Maximum GPU utilization for matrix math");
        println!("  - Check nvidia-smi for GPU utilization");
    }
}
