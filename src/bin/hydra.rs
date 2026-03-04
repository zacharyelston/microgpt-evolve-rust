/*
    MicroGPT Hydra: Multi-Head Evolutionary Engine

    A multi-objective evolutionary system where distinct sub-populations ("Heads")
    optimize for different aesthetic criteria in parallel, then exchange genetic
    material through a central "Body".

    Heads:
    1. The Weaver (Flow): Optimizes for pronounceability and linguistic rhythm.
    2. The Mirror (Symmetry): Optimizes for structural patterns and palindromes.
    3. The Spark (Creativity): Optimizes for novelty and deviation from training data.
    4. The Origin (Loss): Optimizes for low loss (original objective).

    Cycle:
    1. Independent Evolution: Each head runs N generations isolated.
    2. The Gathering: Heads submit their champions to the Body.
    3. Cross-Pollination: The Body breeds champions and redistributes offspring back to heads.
*/

use microgpt_rust::{
    load_training_data, 
    Genome, 
    heads::{Head, Weaver, Mirror, Spark, Origin}
};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const POPULATION_PER_HEAD: usize = 8;
const HEAD_GENERATIONS: usize = 3;  // Gens per cycle before exchange
const CYCLES: usize = 5;            // Total synchronization cycles
const INPUT_FILE: &str = "input.txt";
const TRAIN_STEPS: usize = 300;

struct HydraHead {
    head_impl: Box<dyn Head>,
    population: Vec<Genome>,
}

impl HydraHead {
    fn new(head_impl: Box<dyn Head>) -> Self {
        HydraHead {
            head_impl,
            population: (0..POPULATION_PER_HEAD).map(|_| Genome::new_random()).collect(),
        }
    }

    fn name(&self) -> &str {
        self.head_impl.name()
    }

    fn evolve(&mut self, training_data: &HashSet<String>) {
        // Parallel evaluation of the population
        let head_ref = self.head_impl.as_ref();
        
        self.population.par_iter_mut().for_each(|genome| {
            genome.evaluate(training_data, head_ref, INPUT_FILE, TRAIN_STEPS);
        });

        // Sort by fitness
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        println!("  [{}] Top Score: {:.4} (Genome: Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.5})", 
            self.name(), 
            self.population[0].fitness, 
            self.population[0].n_emb, 
            self.population[0].n_head, 
            self.population[0].n_layer,
            self.population[0].n_ctx,
            self.population[0].n_ff_exp,
            self.population[0].lr
        );
    }

    fn breed(&mut self) {
        // Simple elitism + mutation
        let elitism = 2;
        let mut new_pop = Vec::with_capacity(POPULATION_PER_HEAD);
        
        // Keep elites
        for i in 0..elitism {
            new_pop.push(self.population[i].clone());
        }

        // Breed rest
        let mut rng = rand::thread_rng();
        while new_pop.len() < POPULATION_PER_HEAD {
            let parent = &self.population[rng.gen_range(0..elitism)];
            let mut child = parent.clone();
            child.mutate();
            new_pop.push(child);
        }
        self.population = new_pop;
    }

    // Accept immigrants from other heads
    fn inject_immigrants(&mut self, immigrants: Vec<Genome>) {
        let count = immigrants.len();
        // Replace the worst performing organisms with immigrants
        let start = self.population.len() - count;
        for (i, immigrant) in immigrants.into_iter().enumerate() {
            if start + i < self.population.len() {
                // Reset fitness so it gets re-evaluated under THIS head's objective
                let mut new_genome = immigrant;
                new_genome.fitness = 0.0; 
                new_genome.names.clear();
                self.population[start + i] = new_genome;
            }
        }
    }
}

// --- Main Hydra Loop ---

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       MicroGPT Hydra: Multi-Head Evolution       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("Heads: Weaver (Flow), Mirror (Symmetry), Spark (Creativity), Origin (Loss)");
    println!("Pop per Head: {}, Cycles: {}, Gens/Cycle: {}", POPULATION_PER_HEAD, CYCLES, HEAD_GENERATIONS);

    // Load Data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    // Initialize Heads
    let mut heads = vec![
        HydraHead::new(Box::new(Weaver)),
        HydraHead::new(Box::new(Mirror)),
        HydraHead::new(Box::new(Spark)),
        HydraHead::new(Box::new(Origin)),
    ];

    for cycle in 0..CYCLES {
        println!("\n=== Cycle {}/{} ===", cycle + 1, CYCLES);
        let cycle_start = Instant::now();

        // 1. Independent Evolution
        // Sequential iteration over heads to avoid oversubscribing threads
        for head in &mut heads {
            println!(">> {} is thinking...", head.name());
            for gen in 0..HEAD_GENERATIONS {
                head.evolve(&training_data);
                if gen < HEAD_GENERATIONS - 1 {
                    head.breed();
                }
            }
        }

        // 2. The Gathering (Cross-Pollination)
        println!("\n>> The Gathering: Heads exchange secrets...");
        
        // Collect elites from each head
        let mut exchange_pool = Vec::new();
        for head in &heads {
            // Take top 2 from each head
            exchange_pool.push(head.population[0].clone());
            exchange_pool.push(head.population[1].clone());
        }

        // Distribute pool to all heads (as immigrants)
        for i in 0..heads.len() {
            let mut incoming = Vec::new();
            for genome in &exchange_pool {
                incoming.push(genome.clone());
            }
            heads[i].inject_immigrants(incoming);
        }

        println!(">> Cycle Complete in {:.2?}", cycle_start.elapsed());
    }

    // Final Results
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║                 Hydra Ascended                   ║");
    println!("╚══════════════════════════════════════════════════╝");

    for head in &heads {
        let best = &head.population[0];
        println!("\n[{}] Champion:", head.name());
        println!("  Genome: Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.5}", 
            best.n_emb, best.n_head, best.n_layer, best.n_ctx, best.n_ff_exp, best.lr);
        println!("  Score:  {:.4}", best.fitness);
        println!("  Names:  {}", best.names.iter().take(5).cloned().collect::<Vec<_>>().join(", "));
    }
}
