/*
    MicroGPT Hydra: Configurable Evolution Report
*/

use clap::Parser;
use microgpt_rust::{
    load_training_data, 
    Genome,
    heads::{Head, Weaver, Mirror, Spark, Origin}
};
use rand::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of generations (cycles) to run
    #[arg(short, long, default_value_t = 50)]
    gens: usize,

    /// Training steps per organism per generation
    #[arg(short, long, default_value_t = 100)]
    steps: usize,

    /// Output directory for experiments
    #[arg(short, long, default_value = "experiments")]
    output: String,
    
    /// Population size per head
    #[arg(short, long, default_value_t = 8)]
    pop: usize,
}

const INPUT_FILE: &str = "input.txt";

struct HydraHead {
    head_impl: Box<dyn Head>,
    population: Vec<Genome>,
}

impl HydraHead {
    fn new(head_impl: Box<dyn Head>, pop_size: usize) -> Self {
        HydraHead {
            head_impl,
            population: (0..pop_size).map(|_| Genome::new_random()).collect(),
        }
    }

    fn name(&self) -> &str {
        self.head_impl.name()
    }

    fn evolve(&mut self, training_data: &HashSet<String>, steps: usize) {
        let head_ref = self.head_impl.as_ref();
        self.population.par_iter_mut().for_each(|genome| {
            genome.evaluate(training_data, head_ref, INPUT_FILE, steps);
        });
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    }

    fn breed(&mut self, pop_size: usize) {
        let elitism = 2;
        let mut new_pop = Vec::with_capacity(pop_size);
        // Keep elites
        for i in 0..elitism.min(self.population.len()) {
            new_pop.push(self.population[i].clone());
        }
        
        let mut rng = rand::thread_rng();
        while new_pop.len() < pop_size {
            let parent = &self.population[rng.gen_range(0..elitism.min(self.population.len()))];
            let mut child = parent.clone();
            child.mutate();
            new_pop.push(child);
        }
        self.population = new_pop;
    }

    fn inject_immigrants(&mut self, immigrants: Vec<Genome>) {
        let count = immigrants.len();
        let start = self.population.len().saturating_sub(count);
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

// Serializable wrapper for reporting
#[derive(Serialize)]
struct HydraHeadReport<'a> {
    name: &'a str,
    population: &'a Vec<Genome>,
}

fn main() {
    let args = Args::parse();
    // Use ISO-like format but safe for filenames: YYYYMMDD_HHMMSS
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let exp_dir = format!("{}/run_{}", args.output, timestamp);

    println!("╔══════════════════════════════════════════════════╗");
    println!("║       Hydra Evolution Report: Configurable       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("Generations: {}, Steps: {}, Pop: {}", args.gens, args.steps, args.pop);
    println!("Output Directory: {}", exp_dir);

    // Create directories
    fs::create_dir_all(&exp_dir).expect("Failed to create experiment directory");
    let csv_path = format!("{}/metrics.csv", exp_dir);
    let mut csv_file = File::create(&csv_path).expect("Failed to create CSV file");
    writeln!(csv_file, "generation,head,score,n_emb,n_head,n_layer,n_ctx,n_ff,lr,samples").unwrap();

    // Setup Data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    let mut heads = vec![
        HydraHead::new(Box::new(Weaver), args.pop),
        HydraHead::new(Box::new(Mirror), args.pop),
        HydraHead::new(Box::new(Spark), args.pop),
        HydraHead::new(Box::new(Origin), args.pop),
    ];

    let start_time = Instant::now();

    for cycle in 0..args.gens {
        let gen = cycle + 1;
        
        // 1. Evolve
        for head in &mut heads {
            head.evolve(&training_data, args.steps);
            if gen < args.gens {
                head.breed(args.pop);
            }
        }

        // 2. Report & Log
        // Print header for first generation or periodically
        if gen == 1 || gen % 10 == 0 || gen == args.gens {
             println!("\n=== Generation {} ===", gen);
        }

        for head in &heads {
            let best = &head.population[0];
            let samples = best.names.iter().take(3).cloned().collect::<Vec<_>>().join("|");
            
            // CSV
            writeln!(csv_file, "{},{},{:.4},{},{},{},{},{},{:.5},{}", 
                gen, head.name(), best.fitness, 
                best.n_emb, best.n_head, best.n_layer, best.n_ctx, best.n_ff_exp, best.lr, samples
            ).unwrap();

            // Console (periodic)
            if gen == 1 || gen % 10 == 0 || gen == args.gens {
                println!("[{}] Score: {:.4} | DNA: E:{} H:{} L:{} C:{} F:{} | {}", 
                    head.name(), best.fitness, 
                    best.n_emb, best.n_head, best.n_layer, best.n_ctx, best.n_ff_exp, samples);
            }
        }

        // 3. Snapshot (Every 10 gens or last gen)
        if gen % 10 == 0 || gen == args.gens {
             let json_path = format!("{}/heads_gen_{:04}.json", exp_dir, gen);
             let report_heads: Vec<HydraHeadReport> = heads.iter().map(|h| HydraHeadReport {
                 name: h.name(),
                 population: &h.population
             }).collect();
             
             let json = serde_json::to_string_pretty(&report_heads).unwrap();
             fs::write(json_path, json).expect("Failed to write snapshot");
        }

        if !(gen == 1 || gen % 10 == 0 || gen == args.gens) {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        } 

        // 4. Exchange
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

    println!("\n\n>> Experiment Complete. Data saved to {}", exp_dir);
    println!(">> Total Time: {:.2?}", start_time.elapsed());
}
