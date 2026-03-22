/*
    MicroGPT Loss Evolution Engine with Config Support
    
    Extended version that reads evolution parameters from JSON config files
    to support large-scale evolution experiments (100, 1000+ generations).
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use serde_json;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

// Config structures
#[derive(serde::Deserialize, Debug)]
struct EvolutionConfig {
    parameter_ranges: ParameterRanges,
    evolution_settings: EvolutionSettings,
    genesis_config: TrainingConfig,
}

#[derive(serde::Deserialize, Debug)]
struct ParameterRanges {
    n_emb: ValueRange,
    n_ctx: ValueRange,
    n_layer: ValueRange,
    n_head: ValueRange,
    n_ff_exp: ValueRange,
    lr: ValueRange,
    steps: ValueRange,
}

#[derive(serde::Deserialize, Debug)]
struct ValueRange {
    min: usize,
    max: usize,
    #[serde(default)]
    values: Option<Vec<usize>>,
    #[serde(default)]
    log_scale: Option<bool>,
}

#[derive(serde::Deserialize, Debug)]
struct EvolutionSettings {
    population_size: usize,
    generations: usize,
    #[serde(default = "default_mutation_rate")]
    mutation_rate: f64,
    #[serde(default = "default_target_loss")]
    target_loss: f64,
}

fn default_mutation_rate() -> f64 { 0.3 }
fn default_target_loss() -> f64 { 1.2 }

// Evolution parameters (will be loaded from config)
struct EvolutionParams {
    population_size: usize,
    num_generations: usize,
    tournament_size: usize,
    num_immigrants: usize,
    target_loss: f64,
    stagnation_championship: usize,
    stagnation_cataclysm: usize,
    loser_threshold: f64,
    loser_min_samples: usize,
    input_file: String,
}

// Genome structure (same as original)
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

    fn new_random_from_config(config: &EvolutionConfig) -> Self {
        let mut rng = rand::thread_rng();
        
        let n_emb = config.parameter_ranges.n_emb.random_value(&mut rng);
        let n_head = config.parameter_ranges.n_head.random_value(&mut rng);
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(config.parameter_ranges.n_layer.min..=config.parameter_ranges.n_layer.max),
            n_ctx: config.parameter_ranges.n_ctx.random_value(&mut rng),
            n_ff_exp: rng.gen_range(config.parameter_ranges.n_ff_exp.min..=config.parameter_ranges.n_ff_exp.max),
            lr: config.parameter_ranges.lr.random_f64(&mut rng),
            steps: config.parameter_ranges.steps.random_value(&mut rng),
            loss: f64::MAX,
            evaluated: false,
            origin: "random".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn new_random_wide_from_config(config: &EvolutionConfig) -> Self {
        let mut rng = rand::thread_rng();
        
        // For wide search, use expanded ranges
        let n_emb = rng.gen_range(config.parameter_ranges.n_emb.min..=config.parameter_ranges.n_emb.max * 2);
        let n_head = *[1, 2, 4, 6, 8].choose(&mut rng).unwrap();
        let mut g = Genome {
            n_emb,
            n_head,
            n_layer: rng.gen_range(config.parameter_ranges.n_layer.min..=config.parameter_ranges.n_layer.max * 2),
            n_ctx: rng.gen_range(config.parameter_ranges.n_ctx.min..=config.parameter_ranges.n_ctx.max * 2),
            n_ff_exp: rng.gen_range(config.parameter_ranges.n_ff_exp.min..=config.parameter_ranges.n_ff_exp.max * 2),
            lr: 10f64.powf(rng.gen_range(-4.0..-0.8)),
            steps: rng.gen_range(config.parameter_ranges.steps.min..=config.parameter_ranges.steps.max * 2),
            loss: f64::MAX,
            evaluated: false,
            origin: "cataclysm".to_string(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate_from_config(&mut self, config: &EvolutionConfig) {
        let mut rng = rand::thread_rng();
        let num_mutations = rng.gen_range(1..=3);
        for _ in 0..num_mutations {
            match rng.gen_range(0..7) {
                0 => self.n_emb = config.parameter_ranges.n_emb.random_value(&mut rng),
                1 => self.n_head = config.parameter_ranges.n_head.random_value(&mut rng),
                2 => self.n_layer = rng.gen_range(config.parameter_ranges.n_layer.min..=config.parameter_ranges.n_layer.max),
                3 => self.lr = config.parameter_ranges.lr.random_f64(&mut rng),
                4 => {
                    let delta = *[-500, -250, -100, 100, 250, 500].choose(&mut rng).unwrap();
                    self.steps = (self.steps as i32 + delta).clamp(config.parameter_ranges.steps.min as i32, config.parameter_ranges.steps.max as i32 * 2) as usize;
                },
                5 => self.n_ctx = config.parameter_ranges.n_ctx.random_value(&mut rng),
                6 => self.n_ff_exp = rng.gen_range(config.parameter_ranges.n_ff_exp.min..=config.parameter_ranges.n_ff_exp.max),
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
            0 => self.n_layer += 1,
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
                    c if c < 48 => 48,
                    _ => self.n_ctx,
                };
            }
            3 => self.n_ff_exp += 1,
            _ => {}
        }
        self.enforce_constraints();
        self.loss = f64::MAX;
        self.evaluated = false;
    }

    fn enforce_constraints(&mut self) {
        if self.n_emb % self.n_head != 0 {
            let valid: Vec<usize> = [1, 2, 4, 6, 8].iter().copied()
                .filter(|h| self.n_emb % h == 0)
                .collect();
            self.n_head = *valid.last().unwrap_or(&1);
        }
    }

    fn evaluate(&mut self, id: usize, input_file: &str) {
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
            input_file: input_file.to_string(),
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

    fn to_config(&self, gen_samples: usize, input_file: &str) -> TrainingConfig {
        TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_ff_exp: self.n_ff_exp,
            lr: self.lr,
            steps: self.steps,
            input_file: input_file.to_string(),
            gen_samples,
            ..Default::default()
        }
    }
}

// Helper traits for ValueRange
trait RandomValue {
    fn random_value(&self, rng: &mut ThreadRng) -> usize;
    fn random_f64(&self, rng: &mut ThreadRng) -> f64;
}

impl RandomValue for ValueRange {
    fn random_value(&self, rng: &mut ThreadRng) -> usize {
        if let Some(ref values) = self.values {
            *values.choose(rng).unwrap()
        } else {
            rng.gen_range(self.min..=self.max)
        }
    }

    fn random_f64(&self, rng: &mut ThreadRng) -> f64 {
        if self.log_scale.unwrap_or(false) {
            10f64.powf(rng.gen_range((self.min as f64).log10()..(self.max as f64).log10()))
        } else {
            rng.gen_range(self.min as f64..=self.max as f64)
        }
    }
}

// Blacklist structure (same as original)
struct Blacklist {
    failures: HashMap<String, Vec<f64>>,
}

impl Blacklist {
    fn new() -> Self { Blacklist { failures: HashMap::new() } }

    fn record(&mut self, genome: &Genome, loser_threshold: f64) {
        if genome.loss > loser_threshold && genome.loss < f64::MAX {
            self.failures.entry(genome.species()).or_default().push(genome.loss);
        }
    }

    fn is_blacklisted(&self, species: &str, loser_min_samples: usize) -> bool {
        if let Some(losses) = self.failures.get(species) {
            losses.len() >= loser_min_samples
        } else {
            false
        }
    }

    fn len(&self, loser_min_samples: usize) -> usize {
        self.failures.values().filter(|v| v.len() >= loser_min_samples).count()
    }

    fn random_avoiding(&self, config: &EvolutionConfig) -> Genome {
        for _ in 0..20 {
            let g = Genome::new_random_from_config(config);
            if !self.is_blacklisted(&g.species(), 2) {
                return g;
            }
        }
        Genome::new_random_from_config(config)
    }

    fn random_wide_avoiding(&self, config: &EvolutionConfig) -> Genome {
        for _ in 0..20 {
            let g = Genome::new_random_wide_from_config(config);
            if !self.is_blacklisted(&g.species(), 2) {
                return g;
            }
        }
        Genome::new_random_wide_from_config(config)
    }
}

// Other functions (crossover, tournament_select, etc.) would be similar to original
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

fn tournament_select<'a>(pop: &'a [Genome], rng: &mut ThreadRng, tournament_size: usize) -> &'a Genome {
    let mut best: Option<&Genome> = None;
    for _ in 0..tournament_size {
        let candidate = &pop[rng.gen_range(0..pop.len())];
        if best.is_none() || candidate.loss < best.unwrap().loss {
            best = Some(candidate);
        }
    }
    best.unwrap()
}

fn species_census(pop: &[Genome]) -> HashMap<String, Vec<usize>> {
    let mut species: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, g) in pop.iter().enumerate() {
        species.entry(g.species()).or_default().push(i);
    }
    species
}

fn load_config(config_path: &str) -> EvolutionConfig {
    let config_content = std::fs::read_to_string(config_path)
        .unwrap_or_else(|_| panic!("Failed to read config file: {}", config_path));
    
    serde_json::from_str(&config_content)
        .unwrap_or_else(|_| panic!("Failed to parse config file: {}", config_path))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    // Default to 100 generations if no config specified
    let config_path = if args.len() > 1 {
        &args[1]
    } else {
        "evolution_config_100.json"
    };
    
    println!("Loading evolution config from: {}", config_path);
    let config = load_config(config_path);
    
    // Setup evolution parameters from config
    let params = EvolutionParams {
        population_size: config.evolution_settings.population_size,
        num_generations: config.evolution_settings.generations,
        tournament_size: 3,
        num_immigrants: 2,
        target_loss: config.evolution_settings.target_loss,
        stagnation_championship: 2,
        stagnation_cataclysm: 4,
        loser_threshold: 2.3,
        loser_min_samples: 2,
        input_file: "input.txt".to_string(),
    };
    
    println!("Evolution parameters:");
    println!("  Population size: {}", params.population_size);
    println!("  Generations: {}", params.num_generations);
    println!("  Target loss: {}", params.target_loss);
    
    std::fs::create_dir_all("experiments").ok();
    let log_path = format!("experiments/evolve_{}_{}.log", 
        config_path.strip_suffix(".json").unwrap_or_else(|| config_path),
        chrono::Local::now().format("%Y%m%d_%H%M%S"));
    
    let log_file: Mutex<Option<std::fs::File>> = Mutex::new(
        std::fs::File::create(&log_path).ok()
    );

    log!(log_file, "=== MicroGPT Loss Evolution Engine (Configurable) ===");
    log!(log_file, "Config: {}", config_path);
    log!(log_file, "Experiment: {}", log_path);
    log!(log_file, "Target: loss < {:.1}", params.target_loss);
    log!(log_file, "Population: {}, Generations: {}", params.population_size, params.num_generations);
    log!(log_file, "");

    if std::fs::metadata(&params.input_file).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", &params.input_file, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(&params.input_file);

    // Load existing genome if available
    let (base_config, base_loss, base_gen) = TrainingConfig::load_genome()
        .unwrap_or_else(|| (config.genesis_config.clone(), f64::MAX, 0));
    
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
            mutant.mutate_from_config(&config);
            population.push(mutant);
        } else {
            population.push(Genome::new_random_from_config(&config));
        }
    }
    
    let mut best_ever = if base_loss < f64::MAX {
        population[0].clone()
    } else {
        Genome::new_random_from_config(&config)
    };
    best_ever.loss = base_loss;
    
    let mut blacklist = Blacklist::new();
    let total_start = Instant::now();
    
    // Main evolution loop (similar to original but using params from config)
    for gen in 0..params.num_generations {
        let gen_start = Instant::now();
        log!(log_file, "--- Generation {}/{} ---", gen + 1, params.num_generations);

        eprintln!("[gen {}] evaluating {} organisms...", gen + 1, population.len());
        population.par_iter_mut().enumerate().for_each(|(i, genome)| {
            genome.evaluate(i + 1, &params.input_file);
        });

        population.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());

        for g in &population {
            blacklist.record(g, params.loser_threshold);
        }

        let census = species_census(&population);
        let num_species = census.len();
        
        log!(log_file, "  Best: {:.4} | Species: {} | {:.0}s",
            population[0].loss, num_species, gen_start.elapsed().as_secs_f64());

        // Check if we've reached the target
        if population[0].loss < params.target_loss {
            log!(log_file, "  ** Target {:.1} reached! **", params.target_loss);
            if gen >= 10 { // Allow some refinement after reaching target
                break;
            }
        }

        // Breed next generation (simplified for this example)
        if gen < params.num_generations - 1 {
            // Simple breeding logic - would include championship/cataclysm logic from original
            let mut new_pop: Vec<Genome> = Vec::with_capacity(params.population_size);
            
            // Keep elite
            let mut elite = population[0].clone();
            elite.origin = "elite".to_string();
            new_pop.push(elite);
            
            // Fill rest with mutations and crossovers
            let mut rng = rand::thread_rng();
            while new_pop.len() < params.population_size {
                if rng.gen() {
                    let parent = tournament_select(&population, &mut rng, params.tournament_size);
                    let mut child = parent.clone();
                    child.mutate_from_config(&config);
                    child.origin = "mutant".to_string();
                    new_pop.push(child);
                } else {
                    let p1 = tournament_select(&population, &mut rng, params.tournament_size);
                    let p2 = tournament_select(&population, &mut rng, params.tournament_size);
                    let mut child = crossover(p1, p2);
                    child.mutate_from_config(&config);
                    child.origin = "cross".to_string();
                    new_pop.push(child);
                }
            }
            
            population = new_pop;
        }
        
        // Progress indicator for long runs
        if gen % 10 == 0 || gen == params.num_generations - 1 {
            let elapsed = total_start.elapsed().as_secs_f64();
            let rate = gen as f64 / elapsed;
            let eta = if rate > 0.0 { (params.num_generations - gen - 1) as f64 / rate } else { 0.0 };
            println!("Progress: {}/{} generations ({:.1}%) - ETA: {:.0}s", 
                gen + 1, params.num_generations, 
                ((gen + 1) as f64 / params.num_generations as f64) * 100.0,
                eta);
        }
    }

    // Save the best genome
    let best_config = best_ever.to_config(1, &params.input_file);
    best_config.save_genome(params.num_generations, best_ever.loss);
    
    let total_time = total_start.elapsed().as_secs_f64();
    println!("Evolution complete!");
    println!("Total time: {:.0}s ({:.1}s per generation)", total_time, total_time / params.num_generations as f64);
    println!("Best loss: {:.4}", best_ever.loss);
    println!("Best config: {}", best_ever.desc());
    println!("Results saved to: genome.json");
    println!("Log file: {}", log_path);
}

// Macro for logging (same as original)
macro_rules! log {
    ($log:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        println!("{}", msg);
        if let Some(ref f) = *$log.lock().unwrap() {
            let _ = writeln!(f.try_clone().unwrap(), "{}", msg);
        }
    }};
}
