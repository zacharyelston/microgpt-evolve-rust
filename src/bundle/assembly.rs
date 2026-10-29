use crate::{GPT, Val, Vec1};
use std::collections::HashMap;

pub struct SynapticConnection {
    pub weight: Val,
    pub source_node_id: String,
}

pub struct Assembly {
    pub upstream_nodes: HashMap<String, GPT>,
    pub evaluator: GPT,
    pub synaptic_weights: Vec<SynapticConnection>,
    pub l1_penalty: f64,
    pub prune_threshold: f64,
    pub max_synapses: usize,
}

impl Assembly {
    pub fn new(
        upstream_config: Vec<(String, usize, usize, usize, usize, usize, usize)>,
        evaluator_vocab: usize,
        evaluator_ctx: usize,
        evaluator_emb: usize,
        evaluator_layers: usize,
        evaluator_heads: usize,
        evaluator_ff: usize,
        init_scale: f64,
        rms_eps: f64,
        l1_penalty: f64,
        prune_threshold: f64,
        max_synapses: usize,
    ) -> Self {
        let mut upstream_nodes = HashMap::new();
        let mut synaptic_weights = Vec::new();
        
        for (node_id, vocab, ctx, emb, layers, heads, ff) in upstream_config {
            let node = GPT::new(vocab, ctx, emb, layers, heads, ff, init_scale, rms_eps);
            upstream_nodes.insert(node_id.clone(), node);
            
            let weight = Val::new(rand::random::<f64>() * 0.4 + 0.3);
            synaptic_weights.push(SynapticConnection {
                weight,
                source_node_id: node_id,
            });
        }
        
        let evaluator = GPT::new(
            evaluator_vocab,
            evaluator_ctx,
            evaluator_emb,
            evaluator_layers,
            evaluator_heads,
            evaluator_ff,
            init_scale,
            rms_eps,
        );
        
        Assembly {
            upstream_nodes,
            evaluator,
            synaptic_weights,
            l1_penalty,
            prune_threshold,
            max_synapses,
        }
    }
    
    pub fn forward(
        &self,
        inputs: &HashMap<String, Vec<usize>>,
        k_cache: &mut HashMap<String, Vec<Vec<Vec1>>>,
        v_cache: &mut HashMap<String, Vec<Vec<Vec1>>>,
        eval_k: &mut Vec<Vec<Vec1>>,
        eval_v: &mut Vec<Vec<Vec1>>,
    ) -> Vec1 {
        let mut weighted_embeddings = Vec::new();
        
        for conn in &self.synaptic_weights {
            if conn.weight.data().abs() < self.prune_threshold {
                continue;
            }
            
            if let Some(node) = self.upstream_nodes.get(&conn.source_node_id) {
                if let Some(tokens) = inputs.get(&conn.source_node_id) {
                    let node_k = k_cache.entry(conn.source_node_id.clone()).or_insert_with(|| {
                        vec![vec![]; node.wq.len()]
                    });
                    let node_v = v_cache.entry(conn.source_node_id.clone()).or_insert_with(|| {
                        vec![vec![]; node.wq.len()]
                    });
                    
                    let mut embedding = Vec::new();
                    for (pos, &tok) in tokens.iter().enumerate() {
                        let logits = node.forward(tok, pos, node_k, node_v);
                        if embedding.is_empty() {
                            embedding = logits;
                        } else {
                            for (i, val) in logits.iter().enumerate() {
                                embedding[i] = &embedding[i] + val;
                            }
                        }
                    }
                    
                    let scale = Val::new((tokens.len() as f64).recip());
                    for val in &mut embedding {
                        *val = &*val * &scale;
                    }
                    
                    for val in &embedding {
                        weighted_embeddings.push(val * &conn.weight);
                    }
                }
            }
        }
        
        if weighted_embeddings.is_empty() {
            return vec![Val::new(0.0); self.evaluator.lm_head.len()];
        }
        
        let target_dim = self.evaluator.wte[0].len();
        
        let mut combined_embedding = vec![Val::new(0.0); target_dim];
        for (i, val) in weighted_embeddings.iter().enumerate() {
            combined_embedding[i % target_dim] = &combined_embedding[i % target_dim] + val;
        }
        
        use crate::{rmsnorm, linear};
        let normalized = rmsnorm(&combined_embedding, self.evaluator.rms_eps);
        linear(&normalized, &self.evaluator.lm_head)
    }
    
    pub fn params(&self) -> Vec<Val> {
        let mut p = vec![];
        
        for node in self.upstream_nodes.values() {
            p.extend(node.params());
        }
        
        p.extend(self.evaluator.params());
        
        for conn in &self.synaptic_weights {
            p.push(conn.weight.clone());
        }
        
        p
    }
    
    pub fn synaptic_params(&self) -> Vec<Val> {
        self.synaptic_weights.iter().map(|c| c.weight.clone()).collect()
    }
    
    pub fn prune_synapses(&mut self) -> usize {
        let initial_count = self.synaptic_weights.len();
        
        self.synaptic_weights.retain(|conn| {
            conn.weight.data().abs() >= self.prune_threshold
        });
        
        if self.synaptic_weights.len() > self.max_synapses {
            self.synaptic_weights.sort_by(|a, b| {
                b.weight.data().abs().partial_cmp(&a.weight.data().abs()).unwrap()
            });
            self.synaptic_weights.truncate(self.max_synapses);
        }
        
        initial_count - self.synaptic_weights.len()
    }
    
    pub fn l1_regularization(&self) -> Val {
        let mut l1_loss = Val::new(0.0);
        for conn in &self.synaptic_weights {
            let abs_weight = if conn.weight.data() >= 0.0 {
                conn.weight.clone()
            } else {
                -conn.weight.clone()
            };
            l1_loss = &l1_loss + &abs_weight;
        }
        &l1_loss * &Val::new(self.l1_penalty)
    }
    
    pub fn active_synapses(&self) -> usize {
        self.synaptic_weights.iter()
            .filter(|c| c.weight.data().abs() >= self.prune_threshold)
            .count()
    }
}

pub fn build_vocab_for_visual(max_brightness: usize) -> Vec<char> {
    (0..=max_brightness).map(|i| char::from_digit(i as u32, 10).unwrap()).collect()
}

pub fn tokenize_visual_grid(grid_str: &str, chars: &[char]) -> Vec<usize> {
    grid_str.chars()
        .filter_map(|c| chars.iter().position(|&x| x == c))
        .collect()
}

pub fn tokenize_label(label: char) -> usize {
    match label {
        'L' => 0,
        'R' => 1,
        'T' => 2,
        'B' => 3,
        'U' => 4,
        _ => 4,
    }
}

pub fn extract_quadrant(tokens: &[usize], slice: &[usize]) -> Vec<usize> {
    let x_start = slice[0];
    let x_end = slice[1];
    let y_start = slice[2];
    let y_end = slice[3];
    
    let mut quadrant = Vec::new();
    for y in y_start..y_end {
        for x in x_start..x_end {
            let idx = y * 8 + x;
            if idx < tokens.len() {
                quadrant.push(tokens[idx]);
            }
        }
    }
    quadrant
}
