use candle_core::{Device, Result, Tensor, DType, IndexOp};
use candle_nn::{Linear, Module, VarBuilder, VarMap, Embedding, LayerNorm, ops::softmax};

#[derive(Debug, Clone)]
pub struct CandleGPTConfig {
    pub vocab_size: usize,
    pub n_emb: usize,
    pub n_ctx: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff_exp: usize,
}

pub struct CandleGPT {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    config: CandleGPTConfig,
}

struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

struct MultiHeadAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_emb: usize,
}

struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl CandleGPT {
    pub fn new(config: CandleGPTConfig, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(config.vocab_size, config.n_emb, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(config.n_ctx, config.n_emb, vb.pp("wpe"))?;
        
        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(TransformerBlock::new(&config, vb.pp(format!("h.{}", i)))?);
        }
        
        let ln_f = candle_nn::layer_norm(config.n_emb, 1e-5, vb.pp("ln_f"))?;
        let lm_head = candle_nn::linear(config.n_emb, config.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            config,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let input_ids = if input_ids.dims().len() == 1 {
            input_ids.unsqueeze(0)?
        } else {
            input_ids.clone()
        };
        
        let positions = if positions.dims().len() == 1 {
            positions.unsqueeze(0)?
        } else {
            positions.clone()
        };
        
        let token_embeddings = self.wte.forward(&input_ids)?;
        let position_embeddings = self.wpe.forward(&positions)?;
        
        let mut hidden_states = token_embeddings.broadcast_add(&position_embeddings)?;
        
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }
        
        let hidden_states = self.ln_f.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        
        Ok(logits)
    }
    
    pub fn get_embeddings(&self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let input_ids = if input_ids.dims().len() == 1 {
            input_ids.unsqueeze(0)?
        } else {
            input_ids.clone()
        };
        
        let positions = if positions.dims().len() == 1 {
            positions.unsqueeze(0)?
        } else {
            positions.clone()
        };
        
        let token_embeddings = self.wte.forward(&input_ids)?;
        let position_embeddings = self.wpe.forward(&positions)?;
        
        let mut hidden_states = token_embeddings.broadcast_add(&position_embeddings)?;
        
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }
        
        Ok(hidden_states)
    }
}

impl TransformerBlock {
    fn new(config: &CandleGPTConfig, vb: VarBuilder) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(config.n_emb, 1e-5, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::new(config.n_emb, config.n_head, vb.pp("attn"))?;
        let ln_2 = candle_nn::layer_norm(config.n_emb, 1e-5, vb.pp("ln_2"))?;
        let mlp = MLP::new(config.n_emb, config.n_ff_exp, vb.pp("mlp"))?;
        
        Ok(Self { ln_1, attn, ln_2, mlp })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        
        Ok(x)
    }
}

impl MultiHeadAttention {
    fn new(n_emb: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let c_attn = candle_nn::linear(n_emb, 3 * n_emb, vb.pp("c_attn"))?;
        let c_proj = candle_nn::linear(n_emb, n_emb, vb.pp("c_proj"))?;
        
        Ok(Self { c_attn, c_proj, n_head, n_emb })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        
        let qkv = self.c_attn.forward(x)?;
        let q = qkv.narrow(2, 0, self.n_emb)?;
        let k = qkv.narrow(2, self.n_emb, self.n_emb)?;
        let v = qkv.narrow(2, 2 * self.n_emb, self.n_emb)?;
        
        let head_dim = self.n_emb / self.n_head;
        
        let q = q.reshape((b, t, self.n_head, head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, t, self.n_head, head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, t, self.n_head, head_dim))?.transpose(1, 2)?;
        
        let scale = (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? / scale)?;
        let att = softmax(&att, candle_core::D::Minus1)?;
        
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.reshape((b, t, c))?;
        
        self.c_proj.forward(&y)
    }
}

impl MLP {
    fn new(n_emb: usize, n_ff_exp: usize, vb: VarBuilder) -> Result<Self> {
        let c_fc = candle_nn::linear(n_emb, n_ff_exp * n_emb, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear(n_ff_exp * n_emb, n_emb, vb.pp("c_proj"))?;
        
        Ok(Self { c_fc, c_proj })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.gelu()?;
        self.c_proj.forward(&x)
    }
}

pub struct CandleAssembly {
    pub upstream_nodes: Vec<CandleGPT>,
    pub evaluator: CandleGPT,
    pub synaptic_weights: Tensor,
    pub varmap: VarMap,
    pub device: Device,
}

impl CandleAssembly {
    pub fn new(
        upstream_configs: Vec<CandleGPTConfig>,
        evaluator_config: CandleGPTConfig,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let mut upstream_nodes = Vec::new();
        for (i, config) in upstream_configs.iter().enumerate() {
            let node = CandleGPT::new(config.clone(), vb.pp(format!("upstream_{}", i)))?;
            upstream_nodes.push(node);
        }
        
        let evaluator = CandleGPT::new(evaluator_config, vb.pp("evaluator"))?;
        
        let num_upstream = upstream_nodes.len();
        let synaptic_weights = vb.get((num_upstream,), "synaptic_weights")?;
        
        Ok(Self {
            upstream_nodes,
            evaluator,
            synaptic_weights,
            varmap,
            device,
        })
    }
    
    pub fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let mut weighted_embeddings = Vec::new();
        
        for (i, (node, input)) in self.upstream_nodes.iter().zip(inputs.iter()).enumerate() {
            let seq_len = input.dim(0)?;
            let positions = Tensor::arange(0u32, seq_len as u32, &self.device)?;
            
            let embeddings = node.get_embeddings(input, &positions)?;
            
            let mean_embedding = if embeddings.dims().len() == 3 {
                embeddings.mean(1)?
            } else {
                embeddings.mean(0)?.unsqueeze(0)?
            };
            
            let weight = self.synaptic_weights.get(i)?;
            let weighted = mean_embedding.broadcast_mul(&weight)?;
            weighted_embeddings.push(weighted);
        }
        
        let combined = Tensor::stack(&weighted_embeddings, 0)?;
        let combined_flat = combined.flatten_all()?;
        
        let dummy_input = combined_flat.unsqueeze(0)?;
        let seq_len = dummy_input.dim(1)?;
        let positions = Tensor::arange(0u32, seq_len as u32, &self.device)?.unsqueeze(0)?;
        
        let logits = self.evaluator.forward(&dummy_input, &positions)?;
        
        Ok(logits.i((.., 0, ..))?)
    }
}
