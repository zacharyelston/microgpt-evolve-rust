use crate::{train_and_generate, TrainingConfig};
use crate::heads::Head;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub n_emb: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ctx: usize,
    pub n_ff_exp: usize,
    pub lr: f64,
    pub fitness: f64,
    pub names: Vec<String>,
}

impl Genome {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let mut g = Genome {
            n_emb: rng.gen_range(8..=64),
            n_head: rng.gen_range(1..=8),
            n_layer: rng.gen_range(1..=12),
            n_ctx: rng.gen_range(8..=32),
            n_ff_exp: rng.gen_range(1..=8),
            lr: rng.gen_range(0.0001..0.01),
            fitness: 0.0,
            names: Vec::new(),
        };
        g.enforce_constraints();
        g
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let choice = rng.gen_range(0..6);
        match choice {
            0 => self.n_emb = rng.gen_range(8..=64),
            1 => self.n_head = rng.gen_range(1..=8),
            2 => self.n_layer = rng.gen_range(1..=12),
            3 => self.n_ctx = rng.gen_range(8..=32),
            4 => self.n_ff_exp = rng.gen_range(1..=8),
            5 => self.lr = (self.lr * rng.gen_range(0.7..1.3)).max(0.0001).min(0.02),
            _ => {},
        }
        self.enforce_constraints();
        self.fitness = 0.0;
        self.names.clear();
    }

    // Ensure embedding dimension is divisible by number of heads
    pub fn enforce_constraints(&mut self) {
        // Clamp all to reasonable ranges
        self.n_emb = self.n_emb.max(8).min(64);
        self.n_layer = self.n_layer.max(1).min(12);
        self.n_ctx = self.n_ctx.max(8).min(32);
        self.n_ff_exp = self.n_ff_exp.max(1).min(8);
        
        // n_head must be <= n_emb
        self.n_head = self.n_head.max(1).min(8);

        // Ensure divisibility: n_emb must be divisible by n_head
        // We adjust n_emb to be divisible by n_head if possible, or adjust n_head
        while self.n_emb % self.n_head != 0 {
             // Try to find a divisor
             if self.n_head > 1 {
                 self.n_head -= 1;
             } else {
                 break; 
             }
        }

        // Context must be at least 2 to have input->target pairs
        self.n_ctx = self.n_ctx.max(2);
    }

    // Train and evaluate based on a specific head (objective)
    pub fn evaluate(&mut self, training_data: &HashSet<String>, head: &dyn Head, input_file: &str, steps: usize) {
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
            steps,
            input_file: input_file.to_string(),
            ..Default::default()
        };

        let result = train_and_generate(&config, true);
        
        // Calculate fitness using the Head trait
        let score = head.calculate_fitness(&result, training_data);

        self.names = result.names;
        self.fitness = score;
    }
}
