//! Individual 3x3 Vision Block
//! 
//! A single 3x3 vision processing unit that can detect light direction
//! and intensity. This is the fundamental building block for larger assemblies.

use candle_core::{Device, Result, Tensor, Module, DType};
use candle_nn::{VarBuilder, VarMap, Linear};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct VisionBlock3x3 {
    /// Linear layer for processing 3x3 pixel grid
    pub processor: Linear,
    /// Learnable synaptic weight for this block
    pub synaptic_weight: Tensor,
    /// Device where the block is running
    pub device: Device,
    /// Block identifier for assembly tracking
    pub block_id: String,
}

impl VisionBlock3x3 {
    /// Create a new 3x3 vision block
    pub fn new(block_id: &str, embedding_dim: usize, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // Input: 3x3 grid with pixel values 0-9 (one-hot encoded: 3*3*10 = 90 features)
        let processor = candle_nn::linear(90, embedding_dim, vb.pp("processor"))?;
        
        // Single synaptic weight for this block
        let synaptic_weight = vb.get(1, "synaptic_weight")?;
        
        Ok(Self {
            processor,
            synaptic_weight,
            device: device.clone(),
            block_id: block_id.to_string(),
        })
    }
    
    /// Forward pass through the 3x3 block
    pub fn forward(&self, grid_pixels: &[u8]) -> Result<Tensor> {
        // Convert 3x3 grid to one-hot encoded tensor
        let mut features = vec![0.0f32; 90]; // 3*3*10
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            if pixel <= 9 {
                features[i * 10 + pixel as usize] = 1.0;
            }
        }
        
        // Create input tensor: flattened to (90,) then reshape to (1, 90) for batch processing
        let input = Tensor::from_vec(features, 90, &self.device)?.unsqueeze(0)?;
        
        // Forward through processor
        let embedding = self.processor.forward(&input)?;
        
        // Global average pooling across spatial dimensions
        let pooled = embedding.mean(0)?;
        
        // Apply synaptic weight
        let weighted = pooled.broadcast_mul(&self.synaptic_weight)?;
        
        Ok(weighted)
    }
    
    /// Get block parameters for optimization
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.processor.weight().clone(),
            (*self.processor.bias().as_ref().unwrap()).clone(),
            self.synaptic_weight.clone(),
        ]
    }
    
    /// Save block weights to a genome
    pub fn extract_genome(&self) -> Result<BlockGenome> {
        let weights = self.processor.weight().to_vec2::<f32>()?;
        let bias = self.processor.bias().as_ref().unwrap().to_vec1::<f32>()?;
        let synaptic = self.synaptic_weight.to_vec1::<f32>()?[0];
        
        Ok(BlockGenome {
            block_id: self.block_id.clone(),
            processor_weights: weights,
            processor_bias: bias,
            synaptic_weight: synaptic,
            embedding_dim: self.processor.weight().dims()[1],
        })
    }
    
    /// Load block from a saved genome
    pub fn from_genome(genome: &BlockGenome, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        let processor = candle_nn::linear(90, genome.embedding_dim, vb.pp("processor"))?;
        let synaptic_weight = vb.get(1, "synaptic_weight")?;
        
        // Load weights
        let weight_tensor = Tensor::from_vec(genome.processor_weights.clone().into_iter().flatten().collect(), 
                                           (90, genome.embedding_dim), device)?;
        let bias_tensor = Tensor::from_vec(genome.processor_bias.clone(), genome.embedding_dim, device)?;
        let synaptic_tensor = Tensor::from_vec(vec![genome.synaptic_weight], 1, device)?;
        
        // Note: In a real implementation, we'd need to set these weights in the VarMap
        // For now, we'll create the block and assume weights are loaded externally
        
        Ok(Self {
            processor,
            synaptic_weight: synaptic_tensor,
            device: device.clone(),
            block_id: genome.block_id.clone(),
        })
    }
}

/// Serializable genome for a 3x3 vision block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockGenome {
    pub block_id: String,
    pub processor_weights: Vec<Vec<f32>>,
    pub processor_bias: Vec<f32>,
    pub synaptic_weight: f32,
    pub embedding_dim: usize,
}

impl BlockGenome {
    /// Save genome to file
    pub fn save_to_file(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Load genome from file
    pub fn load_from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let genome: BlockGenome = serde_json::from_str(&json)?;
        Ok(genome)
    }
    
    /// Get genome hash for identification
    pub fn hash(&self) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.block_id.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
