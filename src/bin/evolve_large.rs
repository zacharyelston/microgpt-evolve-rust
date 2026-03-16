/*
    MicroGPT Loss Evolution Engine for Large-Scale Tests
    
    Modified version of evolve_loss.rs that supports configurable
    generation counts for 100 and 1000 generation tests.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

// --- Evolution Parameters (configurable via command line) ---
const TOURNAMENT_SIZE: usize = 3;
const NUM_IMMIGRANTS: usize = 2;
const STAGNATION_CHAMPIONSHIP: usize = 2;
const STAGNATION_CATACLYSM: usize = 4;
const LOSER_THRESHOLD: f64 = 2.3;
const LOSER_MIN_SAMPLES: usize = 2;
const INPUT_FILE: &str = "input.txt";

// Configurable parameters
struct ConfigurableParams {
    population_size: usize,
    num_generations: usize,
    target_loss: f64,
}

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

    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let n_emb = *[8, 12, 16, 20, 24, 32].choose(&mut rng).unwrap();
        let n_head = *[1, 2, 4].choose(&mut rng).unwrap();
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(1..=3),
            n_ctx: *[8, 12, 16, 24].choose(&mut rng).unwrap(),
            n_ff_exp: rng.gen_range(1..=4),
            lr: 10f64.powf(rng.gen_range(-3.0..-1.3)),
            steps: *[200, 300, 500, 750, 1000, 1500].choose(&mut rng).unwrap(),
            loss: f64::MAX,
            evaluated: false,
            origin: "random".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn new_random_wide() -> Self {
        let mut rng = rand::thread_rng();
        let n_emb = *[8, 16, 24, 32, 48, 64].choose(&mut rng).unwrap();
        let n_head = *[1, 2, 4, 8].choose(&mut rng).unwrap();
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(1..=6),
            n_ctx: *[8, 12, 16, 24, 32].choose(&mut rng).unwrap(),
            n_ff_exp: rng.gen_range(1..=6),
            lr: 10f64.powf(rng.gen_range(-4.0..-1.0)),
            steps: *[300, 500, 1000, 1500, 2000, 3000].choose(&mut rng).unwrap(),
            loss: f64::MAX,
            evaluated: false,
            origin: "cataclysm".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let num_mutations = rng.gen_range(1..=3);
        for _ in 0..num_mutations {
            match rng.gen_range(0..7) {
                0 => self.n_emb = *[8, 12, 16, 20, 24, 32].choose(&mut rng).unwrap(),
                1 => self.n_head = *[1, 2, 4].choose(&mut rng).unwrap(),
                2 => self.n_layer = rng.gen_range(1..=4),
                3 => self.lr = 10f64.powf(rng.gen_range(-3.0..-1.3)),
                4 => {
                    let delta = *[-500, -250, -100, 100, 250, 500].choose(&mut rng).unwrap();
                    self.steps = (self.steps as i32 + delta).clamp(100, 2000) as usize;
                },
                5 => self.n_ctx = *[8, 12, 16, 24].choose(&mut rng).unwrap(),
                6 => self.n_ff_exp = rng.gen_range(1..=4),
                _ => {},
            }
        }
        self.enforce_constraints();
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    fn fine_tune(&mut self) {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..3) {
            0 => {
                let factor = rng.gen_range(0.7..1.3);
                self.lr = (self.lr * factor).clamp(0.0001, 0.05);
            }
            1 => {
                let delta = rng.gen_range(-200..=200);
                self.steps = (self.steps as i32 + delta).clamp(100, 3000) as usize;
            }
            2 => {
                let factor = rng.gen_range(0.8..1.2);
                self.lr = (self.lr * factor).clamp(0.0001, 0.05);
                let delta = rng.gen_range(-100..=100);
                self.steps = (self.steps as i32 + delta).clamp(100, 3000) as usize;
            }
            _ => {}
        }
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    fn grow(&mut self) {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..4) {
            0 => {
                self.n_layer += 1;
            }
            1 => {
                let new_heads = self.n_head * 2;
                if self.n_emb % new_heads == 0 {
                    self.n_head = new_heads;
                } else {
                    self.n_layer += 1;
                }
            }
            2 => {
                self.n_ctx = match self.n_ctx {
                    c if c < 12 => 12,
                    c if c < 16 => 16,
                    c if c < 24 => 24,
                    c if c < 32 => 32,
                    _ => self.n_ctx,
                };
            }
            3 => {
                self.n_ff_exp += 1;
            }
            _ => {}
        }
        self.enforce_constraints();
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    fn enforce_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            let valid: Vec<usize> = [1, 2, 4, 8].iter().copied()
                .filter(|h| self.n_emb % h == 0)
                .collect();
            self.n_head = *valid.last().unwrap_or(&1);
        }
    }

    fn evaluate(&mut self, id: usize) {
        if self.evaluated {
            eprintln!("[eval] organism {} already evaluated (loss={:.4})", id, self.loss);
            return;
        }
        eprintln!("[eval] organism {} starting: {}", id, self.desc());
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
                eprintln!("[eval] organism {} done: loss={:.4} ({:.1}s)", id, self.loss, start.elapsed().as_secs_f64());
            }
            Err(e) => {
                eprintln!("[eval] organism {} PANICKED: {:?} | config: {}", id, e, self.desc());
                self.loss = f64::MAX;
                self.evaluated = true;
            }
        }
    }

    fn desc(&self) -> String {
        format!("Emb:{:<3} Head:{} Lay:{} Ctx:{:<2} FF:{} LR:{:.4} Steps:{:<4}",
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

fn crossover(a: &Genome, b: &Genome) -> Genome {
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
        origin: "cross".to_string(),
    };
    child.enforce_constraints();
    child
}

fn tournament_select<'a>(pop: &'a [Genome], rng: &mut ThreadRng) -> &'a Genome {
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

// --- Loser Blacklist ---

struct Blacklist {
    failures: HashMap<String, Vec<f64>>,
}

impl Blacklist {
    fn new() -> Self { Blacklist { failures: HashMap::new() } }

    fn record(&mut self, genome: &Genome) {
        if genome.loss > LOSER_THRESHOLD && genome.loss < f64::MAX {
            self.failures.entry(genome.species()).or_default().push(genome.loss);
        }
    }

    fn is_blacklisted(&self, species: &str) -> bool {
        if let Some(losses) = self.failures.get(species) {
            losses.len() >= LOSER_MIN_SAMPLES
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.failures.values().filter(|v| v.len() >= LOSER_MIN_SAMPLES).count()
    }

    fn random_avoiding(&self) -> Genome {
        for _ in 0..20 {
            let g = Genome::new_random();
            if !self.is_blacklisted(&g.species()) {
                return g;
            }
        }
        Genome::new_random()
    }

    fn random_wide_avoiding(&self) -> Genome {
        for _ in 0..20 {
            let g = Genome::new_random_wide();
            if !self.is_blacklisted(&g.species()) {
                return g;
            }
        }
        Genome::new_random_wide()
    }
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
    format!("experiments/evolve_{}.log", now.format("%Y%m%d_%H%M%S"))
}

// --- Main Evolution Loop ---

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse command line arguments
    let params = if args.len() >= 2 {
        let generations: usize = args[1].parse().unwrap_or_else(|_| {
            eprintln!("Invalid generation count: {}", args[1]);
            std::process::exit(1);
        });
        
        // Configure based on generation count
        if generations == 100 {
            ConfigurableParams {
                population_size: 12,
                num_generations: generations,
                target_loss: 1.0,
            }
        } else if generations == 1000 {
            ConfigurableParams {
                population_size: 16,
                num_generations: generations,
                target_loss: 0.8,
            }
        } else {
            // Custom configuration
            ConfigurableParams {
                population_size: std::cmp::max(8, std::cmp::min(20, generations / 10)),
                num_generations: generations,
                target_loss: if generations >= 500 { 0.8 } else { 1.0 },
            }
        }
    } else {
        // Default to 100 generations
        ConfigurableParams {
            population_size: 12,
            num_generations: 100,
            target_loss: 1.0,
        }
    };

    println!("Starting large-scale evolution:");
    println!("  Generations: {}", params.num_generations);
    println!("  Population size: {}", params.population_size);
    println!("  Target loss: {}", params.target_loss);

    std::fs::create_dir_all("experiments").ok();
    let log_path = experiment_filename();
    let log_file: Mutex<Option<std::fs::File>> = Mutex::new(
        std::fs::File::create(&log_path).ok()
    );

    log!(log_file, "=== MicroGPT Loss Evolution Engine (Large-Scale) ===");
    log!(log_file, "Experiment: {}", log_path);
    log!(log_file, "Target: loss < {:.1}", params.target_loss);
    log!(log_file, "Population: {}, Generations: {}", params.population_size, params.num_generations);
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
    
    println!("Starting evolution from genome: loss {:.6} (generation {})", base_loss, base_gen);
    
    // Create initial population
    let mut population: Vec<Genome> = Vec::with_capacity(params.population_size);
    
    if base_loss < f64::MAX {
        let mut base_genome = Genome::from_config(&base_config);
        base_genome.loss = base_loss;
        base_genome.evaluated = true;
        population.push(base_genome);
        println!("Added existing genome to population (loss: {:.6})", base_loss);
    }
    
    while population.len() < params.population_size {
        if base_loss < f64::MAX && population.len() < params.population_size / 2 {
            let mut mutant = Genome::from_config(&base_config);
            mutant.mutate();
            population.push(mutant);
        } else {
            population.push(Genome::new_random());
        }
    }
    
    let mut best_ever = if base_loss < f64::MAX {
        population[0].clone()
    } else {
        Genome::new_random()
    };
    best_ever.loss = base_loss;
    let mut blacklist = Blacklist::new();
    let total_start = Instant::now();
    let mut target_gen: Option<usize> = None;
    let mut stagnation_count: usize = 0;
    let mut prev_best_loss: f64 = f64::MAX;

    for gen in 0..params.num_generations {
        let gen_start = Instant::now();
        log!(log_file, "--- Generation {}/{} ---", gen + 1, params.num_generations);

        eprintln!("[gen {}] evaluating {} organisms...", gen + 1, population.len());
        population.par_iter_mut().enumerate().for_each(|(i, genome)| {
            genome.evaluate(i + 1);
        });

        population.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());

        // Record losers in the blacklist
        for g in &population {
            blacklist.record(g);
        }

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

        if target_gen.is_none() && best_ever.loss < params.target_loss {
            target_gen = Some(gen + 1);
            log!(log_file, "  ** Target {:.1} reached! Continuing to evolve... **", params.target_loss);
        }

        let elapsed = gen_start.elapsed().as_secs_f64();
        log!(log_file, "  Best: {:.4} | Species: {} | Stagnation: {} | Blacklisted: {} | {:.0}s\n",
            gen_best.loss, num_species, stagnation_count, blacklist.len(), elapsed);

        // Progress indicator for long runs
        if gen % 10 == 0 || gen == params.num_generations - 1 {
            let total_elapsed = total_start.elapsed().as_secs_f64();
            let rate = gen as f64 / total_elapsed;
            let eta = if rate > 0.0 { (params.num_generations - gen - 1) as f64 / rate } else { 0.0 };
            println!("Progress: {}/{} generations ({:.1}%) - ETA: {:.0}s - Best: {:.4}", 
                gen + 1, params.num_generations, 
                ((gen + 1) as f64 / params.num_generations as f64) * 100.0,
                eta, gen_best.loss);
        }

        // --- Breed the next generation ---
        if gen < params.num_generations - 1 {
            // Simplified breeding for large runs
            let mut new_pop: Vec<Genome> = Vec::with_capacity(params.population_size);
            
            // Keep elite
            let mut elite = population[0].clone();
            elite.origin = "elite".to_string();
            new_pop.push(elite);
            
            // Fill rest with mutations and crossovers
            let mut rng = rand::thread_rng();
            while new_pop.len() < params.population_size {
                if rng.gen() {
                    let parent = tournament_select(&population, &mut rng);
                    let mut child = parent.clone();
                    child.mutate();
                    child.origin = "mutant".to_string();
                    new_pop.push(child);
                } else {
                    let p1 = tournament_select(&population, &mut rng);
                    let p2 = tournament_select(&population, &mut rng);
                    let mut child = crossover(p1, p2);
                    child.mutate();
                    child.origin = "cross".to_string();
                    new_pop.push(child);
                }
            }
            
            population = new_pop;
        }
    }

    // Save the best genome
    let best_config = best_ever.to_config(1);
    best_config.save_genome(best_ever.loss, params.num_generations);
    
    let total_time = total_start.elapsed().as_secs_f64();
    println!("\n🎉 Evolution Complete!");
    println!("======================");
    println!("Total generations: {}", params.num_generations);
    println!("Total time: {:.0}s ({:.1}s per generation)", total_time, total_time / params.num_generations as f64);
    println!("Best loss: {:.4}", best_ever.loss);
    println!("Best config: {}", best_ever.desc());
    println!("Results saved to: genome.json");
    println!("Log file: {}", log_path);
    
    if let Some(g) = target_gen {
        println!("Target {:.1} first reached: generation {}", params.target_loss, g);
    }
}
