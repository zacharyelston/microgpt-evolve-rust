/*
    MicroGPT Library - CPU Only Version
    
    This version uses pure CPU matrix operations for comparison.
*/

use rand::Rng;
use std::{cell::RefCell, collections::HashSet, io::Write, ops::{Add, Mul, Neg, Sub}, rc::Rc};

pub type Vec1 = Vec<Val>;
pub type Mat2 = Vec<Vec1>;

#[derive(Clone)]
pub struct Val(pub Rc<RefCell<Node>>);

pub struct Node {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<(Val, f64)>,
}

impl Val {
    pub fn new(data: f64) -> Self { Val(Rc::new(RefCell::new(Node { data, grad: 0., prev: vec![] }))) }
    pub fn data(&self) -> f64 { self.0.borrow().data }
    pub fn grad(&self) -> f64 { self.0.borrow().grad }
    pub fn zero(&self) { self.0.borrow_mut().grad = 0.; }

    pub fn backward(&self) {
        let mut order = vec![];
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut order);
        
        for v in order.iter().rev() {
            v.0.borrow_mut().grad = 0.;
        }
        
        self.0.borrow_mut().grad = 1.;
        
        for v in order.iter().rev() {
            let (data, grad, prev) = {
                let node = v.0.borrow();
                (node.data, node.grad, node.prev.clone())
            };
            for (child, local_grad) in prev {
                child.0.borrow_mut().grad += grad * local_grad;
            }
        }
    }
    
    fn build_topo(&self, visited: &mut HashSet<*const RefCell<Node>>, order: &mut Vec<Val>) {
        let ptr = self.0.as_ptr();
        if visited.contains(&ptr) { return; }
        visited.insert(ptr);
        for (child, _) in &self.0.borrow().prev {
            child.build_topo(visited, order);
        }
        order.push(self.clone());
    }
}

// CPU-only matrix operations
pub fn mat(r: usize, c: usize, scale: f64) -> Mat2 {
    let mut rng = rand::thread_rng();
    (0..r).map(|_| (0..c).map(|_| Val::new(rng.gen_range(-1.0..1.0) * scale)).collect()).collect()
}

pub fn linear(x: &[Val], w: &Mat2) -> Vec1 {
    // Pure CPU matrix multiplication
    w.iter().map(|row| row.iter().zip(x).map(|(w, x)| w * x).fold(Val::new(0.), |a, b| a + b)).collect()
}

pub fn softmax(x: &[Val]) -> Vec1 {
    let max = x.iter().map(|v| v.data()).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec1 = x.iter().map(|v| (v - &Val::new(max)).exp()).collect();
    let sum = exps.iter().fold(Val::new(0.), |a, b| a + b);
    let inv = sum.pow(-1.);
    exps.iter().map(|v| v * &inv).collect()
}

pub fn rmsnorm(x: &[Val]) -> Vec1 {
    let sum_sq: Val = x.iter().map(|v| v * v).fold(Val::new(0.), |a, b| a + b);
    let rms = (sum_sq / &Val::new(x.len() as f64)).pow(-0.5);
    x.iter().map(|v| v * &rms).collect()
}

// Include all the other MicroGPT code here (omitted for brevity)
// This would include the full GPT implementation, training, etc.

pub const GENOME_FILE: &str = "genome.json";

#[derive(Clone, Debug, Default)]
pub struct TrainingConfig {
    pub n_emb: usize,
    pub n_ctx: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff_exp: usize,
    pub lr: f64,
    pub steps: usize,
    pub input_file: String,
    pub gen_samples: usize,
}

impl TrainingConfig {
    pub fn save_genome(&self, loss: f64, generation: usize) -> std::io::Result<()> {
        let json = format!(
            "{{\n  \"n_emb\": {},\n  \"n_ctx\": {},\n  \"n_layer\": {},\n  \"n_head\": {},\n  \"n_ff_exp\": {},\n  \"steps\": {},\n  \"lr\": {},\n  \"loss\": {},\n  \"generation\": {},\n  \"evolved\": true\n}}",
            self.n_emb, self.n_ctx, self.n_layer, self.n_head, self.n_ff_exp,
            self.steps, self.lr, loss, generation
        );
        std::fs::write(GENOME_FILE, json)
    }

    pub fn load_genome() -> Option<(TrainingConfig, f64, usize)> {
        let data = std::fs::read_to_string(GENOME_FILE).ok()?;
        let mut cfg = TrainingConfig::default();
        let mut loss = 0.0;
        let mut gen = 0;
        for line in data.lines() {
            let line = line.trim().trim_end_matches(',');
            if let Some((key, val)) = line.split_once(':') {
                let key = key.trim().trim_matches('"');
                let val = val.trim();
                match key {
                    "n_emb" => cfg.n_emb = val.parse().unwrap_or(cfg.n_emb),
                    "n_ctx" => cfg.n_ctx = val.parse().unwrap_or(cfg.n_ctx),
                    "n_layer" => cfg.n_layer = val.parse().unwrap_or(cfg.n_layer),
                    "n_head" => cfg.n_head = val.parse().unwrap_or(cfg.n_head),
                    "n_ff_exp" => cfg.n_ff_exp = val.parse().unwrap_or(cfg.n_ff_exp),
                    "steps" => cfg.steps = val.parse().unwrap_or(cfg.steps),
                    "lr" => cfg.lr = val.parse().unwrap_or(cfg.lr),
                    "loss" => loss = val.parse().unwrap_or(0.0),
                    "generation" => gen = val.parse().unwrap_or(0),
                    _ => {}
                }
            }
        }
        Some((cfg, loss, gen))
    }
}

pub struct TrainingResult {
    pub final_loss: f64,
}

pub fn load_training_data(_file: &str) {
    // Stub implementation
}

pub fn train_and_generate(_config: &TrainingConfig, _silent: bool) -> TrainingResult {
    // Stub implementation - would include full training logic
    TrainingResult { final_loss: 1.5 }
}

// Macro-based arithmetic operators (simplified)
macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for &Val {
            type Output = Val;
            fn $method(self, other: &Val) -> Val {
                let result = Val::new(self.data() $op other.data());
                result.0.borrow_mut().prev = vec![(self.clone(), 1.0), (other.clone(), 1.0)];
                result
            }
        }
    };
}

impl_binary_op!(Add, add, +);
impl_binary_op!(Mul, mul, *);

impl Neg for &Val {
    type Output = Val;
    fn neg(self) -> Val {
        let result = Val::new(-self.data());
        result.0.borrow_mut().prev = vec![(self.clone(), -1.0)];
        result
    }
}

impl Sub for &Val {
    type Output = Val;
    fn sub(self, other: &Val) -> Val {
        self + &(-other)
    }
}

impl Val {
    pub fn exp(&self) -> Val {
        let result = Val::new(self.data().exp());
        result.0.borrow_mut().prev = vec![(self.clone(), result.data())];
        result
    }
    
    pub fn pow(&self, exp: f64) -> Val {
        let result = Val::new(self.data().powf(exp));
        result.0.borrow_mut().prev = vec![(self.clone(), exp * self.data().powf(exp - 1.0))];
        result
    }
}
