/*
    MicroGPT Hydra: Evolution Report (0, 10, 50 Generations)
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const POPULATION_PER_HEAD: usize = 8;
const HEAD_GENERATIONS: usize = 1;  // 1 gen per cycle for fine-grained reporting
const CYCLES: usize = 50;           // Total 50 generations
const INPUT_FILE: &str = "input.txt";
const TRAIN_STEPS: usize = 100;

// --- Genome: hyperparameters as DNA ---

#[derive(Clone, Debug)]
struct Genome {
    n_emb: usize,
    n_head: usize,
    n_layer: usize,
    n_ctx: usize,
    n_ff_exp: usize,
    lr: f64,
    fitness: f64,
    names: Vec<String>,
}

impl Genome {
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let mut g = Genome {
            n_emb: rng.gen_range(1..=8),
            n_head: rng.gen_range(1..=8),
            n_layer: rng.gen_range(1..=8),
            n_ctx: rng.gen_range(1..=8),
            n_ff_exp: rng.gen_range(1..=8),
            lr: rng.gen_range(0.001..0.02),
            fitness: 0.0,
            names: Vec::new(),
        };
        g.enforce_constraints();
        g
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let choice = rng.gen_range(0..6);
        match choice {
            0 => self.n_emb = rng.gen_range(1..=8),
            1 => self.n_head = rng.gen_range(1..=8),
            2 => self.n_layer = rng.gen_range(1..=8),
            3 => self.n_ctx = rng.gen_range(1..=8),
            4 => self.n_ff_exp = rng.gen_range(1..=8),
            5 => self.lr = (self.lr * rng.gen_range(0.7..1.3)).max(0.0001).min(0.1),
            _ => {},
        }
        self.enforce_constraints();
        self.fitness = 0.0;
        self.names.clear();
    }

    fn enforce_constraints(&mut self) {
        self.n_emb = self.n_emb.max(1).min(8);
        self.n_layer = self.n_layer.max(1).min(8);
        self.n_ctx = self.n_ctx.max(1).min(8);
        self.n_ff_exp = self.n_ff_exp.max(1).min(8);
        self.n_head = self.n_head.max(1).min(self.n_emb);
        while self.n_emb % self.n_head != 0 {
            self.n_head -= 1;
        }
        self.n_ctx = self.n_ctx.max(2);
    }

    fn evaluate(&mut self, training_data: &HashSet<String>, objective: &Objective) {
        if self.fitness != 0.0 && !self.names.is_empty() {
            return;
        }

        let config = TrainingConfig {
            n_emb: self.n_emb,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_ff_exp: self.n_ff_exp,
            lr: self.lr,
            steps: TRAIN_STEPS,
            input_file: INPUT_FILE.to_string(),
            ..Default::default()
        };

        let result = train_and_generate(&config, true); // Silent mode for speed
        
        let score = match objective {
            Objective::Weaver => calculate_flow(&result.names),
            Objective::Mirror => calculate_symmetry(&result.names),
            Objective::Spark => calculate_creativity(&result.names, training_data),
            Objective::Origin => 1.0 / result.final_loss.max(0.0001),
        };

        self.names = result.names;
        self.fitness = score;
    }
}

// --- Hydra Heads ---

#[derive(Clone, Copy, Debug)]
enum Objective {
    Weaver, Mirror, Spark, Origin
}

struct HydraHead {
    objective: Objective,
    population: Vec<Genome>,
    name: String,
}

impl HydraHead {
    fn new(objective: Objective, name: &str) -> Self {
        HydraHead {
            objective,
            population: (0..POPULATION_PER_HEAD).map(|_| Genome::new_random()).collect(),
            name: name.to_string(),
        }
    }

    fn evolve(&mut self, training_data: &HashSet<String>) {
        let obj = self.objective;
        self.population.par_iter_mut().for_each(|genome| {
            genome.evaluate(training_data, &obj);
        });
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    }

    fn breed(&mut self) {
        let elitism = 2;
        let mut new_pop = Vec::with_capacity(POPULATION_PER_HEAD);
        for i in 0..elitism { new_pop.push(self.population[i].clone()); }
        let mut rng = rand::thread_rng();
        while new_pop.len() < POPULATION_PER_HEAD {
            let parent = &self.population[rng.gen_range(0..elitism)];
            let mut child = parent.clone();
            child.mutate();
            new_pop.push(child);
        }
        self.population = new_pop;
    }

    fn inject_immigrants(&mut self, immigrants: Vec<Genome>) {
        let count = immigrants.len();
        let start = self.population.len() - count;
        for (i, immigrant) in immigrants.into_iter().enumerate() {
            if start + i < self.population.len() {
                let mut new_genome = immigrant;
                new_genome.fitness = 0.0; 
                new_genome.names.clear();
                self.population[start + i] = new_genome;
            }
        }
    }
}

// --- Fitness Functions ---

fn calculate_flow(names: &[String]) -> f64 {
    if names.is_empty() { return -100.0; }
    let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u', 'y'].iter().cloned().collect();
    let mut total = 0.0;
    let mut valid = 0;
    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) { continue; }
        let mut score = 0.0;
        let mut cons_v = 0;
        let mut cons_c = 0;
        for c in name.chars() {
            if vowels.contains(&c) { cons_v += 1; cons_c = 0; } else { cons_c += 1; cons_v = 0; }
            if cons_v > 2 || cons_c > 2 { score -= 1.0; }
        }
        if name.len() >= 4 && name.len() <= 8 { score += 1.0; }
        total += score;
        valid += 1;
    }
    if valid == 0 { -100.0 } else { total / valid as f64 }
}

fn calculate_symmetry(names: &[String]) -> f64 {
    if names.is_empty() { return -100.0; }
    let mut total = 0.0;
    let mut valid = 0;
    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 { continue; }
        let mut score = 0.0;
        let chars: Vec<char> = name.chars().collect();
        if chars.iter().eq(chars.iter().rev()) { score += 3.0; }
        if name.len() >= 4 {
            let mid = name.len() / 2;
            if name[..mid] == name[mid..mid*2] { score += 2.0; }
        }
        if name.ends_with('a') || name.ends_with('n') || name.ends_with('y') { score += 0.5; }
        total += score;
        valid += 1;
    }
    if valid == 0 { -100.0 } else { total / valid as f64 }
}

fn calculate_creativity(names: &[String], training_data: &HashSet<String>) -> f64 {
    if names.is_empty() { return -100.0; }
    let mut total = 0.0;
    let mut valid = 0;
    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 { continue; }
        if training_data.contains(&name) { total -= 5.0; } else { total += 2.0; }
        if name.len() > 6 { total += 0.5; }
        valid += 1;
    }
    if valid == 0 { -100.0 } else { total / valid as f64 }
}

// --- Main Hydra Loop ---

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       Hydra Evolution Report: 50 Generations     ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("Heads: Weaver, Mirror, Spark, Origin");
    println!("Constraints: 1-8 range for all hyperparameters");

    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    let mut heads = vec![
        HydraHead::new(Objective::Weaver, "Weaver"),
        HydraHead::new(Objective::Mirror, "Mirror"),
        HydraHead::new(Objective::Spark,  "Spark"),
        HydraHead::new(Objective::Origin, "Origin"),
    ];

    let start_time = Instant::now();

    for cycle in 0..CYCLES {
        let gen = cycle + 1;
        let is_report_step = gen == 1 || gen == 10 || gen == 50;

        // 1. Evolve
        for head in &mut heads {
            head.evolve(&training_data);
            if gen < CYCLES {
                head.breed();
            }
        }

        // Report
        if is_report_step {
            println!("\n=== Generation {} Report ===", gen);
            for head in &heads {
                let best = &head.population[0];
                println!("[{}] Score: {:.4} | DNA: Emb:{} H:{} L:{} C:{} FF:{} LR:{:.5}", 
                    head.name, best.fitness, 
                    best.n_emb, best.n_head, best.n_layer, best.n_ctx, best.n_ff_exp, best.lr);
                println!("   Samples: {}", best.names.iter().take(5).cloned().collect::<Vec<_>>().join(", "));
            }
        } else {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        // 2. Exchange
        let mut exchange_pool = Vec::new();
        for head in &heads {
            exchange_pool.push(head.population[0].clone());
            exchange_pool.push(head.population[1].clone());
        }
        for i in 0..heads.len() {
            let mut incoming = Vec::new();
            for genome in &exchange_pool {
                incoming.push(genome.clone());
            }
            heads[i].inject_immigrants(incoming);
        }
    }

    println!("\n\n>> Total Time: {:.2?}", start_time.elapsed());
}
