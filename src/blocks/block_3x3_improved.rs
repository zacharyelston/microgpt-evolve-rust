//! Improved 3x3 Vision Block
//! 
//! Enhanced 3x3 vision processing with spatial awareness and better architecture
//! for direction detection in planarian eyespot experiments.

use candle_core::{Device, Result, Tensor, Module, DType};
use candle_nn::{VarBuilder, VarMap, Linear, Conv2d, LayerNorm, Conv2dConfig};

#[derive(Debug, Clone)]
pub struct VisionBlock3x3Improved {
    /// Pixel embedding layer (10 values per pixel)
    pixel_embedding: Linear,
    /// 2D convolution for spatial processing
    conv1: Conv2d,
    /// Second convolution layer
    conv2: Conv2d, 
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Final processing layer
    processor: Linear,
    /// Learnable synaptic weight for this block
    synaptic_weight: Tensor,
    /// Device where the block is running
    device: Device,
    /// Block identifier for assembly tracking
    pub block_id: String,
}

impl VisionBlock3x3Improved {
    /// Create a new improved 3x3 vision block
    pub fn new(block_id: &str, embedding_dim: usize, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // Pixel embedding: 10 values (0-9) -> embedding_dim
        let pixel_embedding = candle_nn::linear(10, embedding_dim, vb.pp("pixel_embedding"))?;
        
        // Convolution layers for spatial processing
        // Input: (batch, embedding_dim, 3, 3)
        let conv1 = candle_nn::conv2d(embedding_dim, embedding_dim, 3, Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        }, vb.pp("conv1"))?;
        
        let conv2 = candle_nn::conv2d(embedding_dim, embedding_dim, 3, Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        }, vb.pp("conv2"))?;
        
        // Layer norm for stability
        let layer_norm = candle_nn::layer_norm(embedding_dim, 1e-5, vb.pp("layer_norm"))?;
        
        // Final processor: (3*3*embedding_dim) -> embedding_dim
        let final_input_dim = 3 * 3 * embedding_dim;
        let processor = candle_nn::linear(final_input_dim, embedding_dim, vb.pp("processor"))?;
        
        // Single synaptic weight for this block
        let synaptic_weight = vb.get(1, "synaptic_weight")?;
        
        Ok(Self {
            pixel_embedding,
            conv1,
            conv2,
            layer_norm,
            processor,
            synaptic_weight,
            device: device.clone(),
            block_id: block_id.to_string(),
        })
    }
    
    /// Forward pass through the improved 3x3 block
    pub fn forward(&self, grid_pixels: &[u8]) -> Result<Tensor> {
        // Convert 3x3 grid to one-hot encoded features
        let mut features = vec![0.0f32; 9 * 10]; // 9 pixels * 10 values each
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            if pixel <= 9 {
                features[i * 10 + pixel as usize] = 1.0;
            }
        }
        
        // Create input tensor and embed
        let input = Tensor::from_vec(features, (9, 10), &self.device)?;
        let embedded = self.pixel_embedding.forward(&input)?;
        
        // Reshape to spatial format: (1, embedding_dim, 3, 3)
        let spatial = embedded.reshape((1, embedded.dims()[1], 3, 3))?;
        
        // Apply convolutions with spatial awareness
        let conv1_out = self.conv1.forward(&spatial)?;
        let conv1_activated = conv1_out.relu()?;
        
        let conv2_out = self.conv2.forward(&conv1_activated)?;
        let conv2_activated = conv2_out.relu()?;
        
        // Layer normalization - apply to embedding dimension
        let conv2_flat = conv2_activated.flatten_all()?;
        let embedding_dim = self.processor.weight().dims()[0] / 9; // Get embedding dim from processor
        let reshaped = conv2_flat.reshape((1, 9, embedding_dim))?; // (1, 9, embedding_dim)
        let normalized = self.layer_norm.forward(&reshaped)?;
        
        // Final processing
        let processed = self.processor.forward(&normalized)?;
        
        // Apply synaptic weight
        let weighted = processed.broadcast_mul(&self.synaptic_weight)?;
        
        Ok(weighted.squeeze(0)?)
    }
    
    /// Get block parameters for optimization
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.pixel_embedding.weight().clone(),
            (*self.pixel_embedding.bias().as_ref().unwrap()).clone(),
            self.conv1.weight().clone(),
            (*self.conv1.bias().as_ref().unwrap()).clone(),
            self.conv2.weight().clone(),
            (*self.conv2.bias().as_ref().unwrap()).clone(),
            self.layer_norm.weight().clone(),
            (*self.layer_norm.bias().as_ref().unwrap()).clone(),
            self.processor.weight().clone(),
            (*self.processor.bias().as_ref().unwrap()).clone(),
            self.synaptic_weight.clone(),
        ]
    }
    
    /// Save block weights to a genome
    pub fn extract_genome(&self) -> Result<ImprovedBlockGenome> {
        let pixel_weights = self.pixel_embedding.weight().to_vec2::<f32>()?;
        let pixel_bias = self.pixel_embedding.bias().as_ref().unwrap().to_vec1::<f32>()?;
        
        let conv1_weights = self.conv1.weight().to_vec2::<f32>()?;
        let conv1_bias = self.conv1.bias().as_ref().unwrap().to_vec1::<f32>()?;
        
        let conv2_weights = self.conv2.weight().to_vec2::<f32>()?;
        let conv2_bias = self.conv2.bias().as_ref().unwrap().to_vec1::<f32>()?;
        
        let ln_weights = self.layer_norm.weight().to_vec1::<f32>()?;
        let ln_bias = self.layer_norm.bias().as_ref().unwrap().to_vec1::<f32>()?;
        
        let processor_weights = self.processor.weight().to_vec2::<f32>()?;
        let processor_bias = self.processor.bias().as_ref().unwrap().to_vec1::<f32>()?;
        
        let synaptic = self.synaptic_weight.to_vec1::<f32>()?[0];
        
        Ok(ImprovedBlockGenome {
            block_id: self.block_id.clone(),
            pixel_embedding_weights: pixel_weights,
            pixel_embedding_bias: pixel_bias,
            conv1_weights,
            conv1_bias,
            conv2_weights,
            conv2_bias,
            layer_norm_weights: ln_weights,
            layer_norm_bias: ln_bias,
            processor_weights,
            processor_bias,
            synaptic_weight: synaptic,
            embedding_dim: self.pixel_embedding.weight().dims()[1],
        })
    }
    
    /// Load block from a saved genome
    pub fn from_genome(genome: &ImprovedBlockGenome, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        let pixel_embedding = candle_nn::linear(10, genome.embedding_dim, vb.pp("pixel_embedding"))?;
        let conv1 = candle_nn::conv2d(genome.embedding_dim, genome.embedding_dim, 3, 
            candle_nn::Conv2dConfig { padding: 1, stride: 1, ..Default::default() }, 
            vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(genome.embedding_dim, genome.embedding_dim, 3,
            candle_nn::Conv2dConfig { padding: 1, stride: 1, ..Default::default() },
            vb.pp("conv2"))?;
        let layer_norm = candle_nn::layer_norm(genome.embedding_dim, 1e-5, vb.pp("layer_norm"))?;
        let processor = candle_nn::linear(3 * 3 * genome.embedding_dim, genome.embedding_dim, vb.pp("processor"))?;
        let synaptic_weight = vb.get(1, "synaptic_weight")?;
        
        // Note: In a real implementation, we'd load the actual weights here
        // For now, we create the structure and assume weights are loaded externally
        
        Ok(Self {
            pixel_embedding,
            conv1,
            conv2,
            layer_norm,
            processor,
            synaptic_weight,
            device: device.clone(),
            block_id: genome.block_id.clone(),
        })
    }
}

/// Serializable genome for improved 3x3 vision block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImprovedBlockGenome {
    pub block_id: String,
    pub pixel_embedding_weights: Vec<Vec<f32>>,
    pub pixel_embedding_bias: Vec<f32>,
    pub conv1_weights: Vec<Vec<f32>>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weights: Vec<Vec<f32>>,
    pub conv2_bias: Vec<f32>,
    pub layer_norm_weights: Vec<f32>,
    pub layer_norm_bias: Vec<f32>,
    pub processor_weights: Vec<Vec<f32>>,
    pub processor_bias: Vec<f32>,
    pub synaptic_weight: f32,
    pub embedding_dim: usize,
}

impl ImprovedBlockGenome {
    /// Save genome to file
    pub fn save_to_file(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Load genome from file
    pub fn load_from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let genome: ImprovedBlockGenome = serde_json::from_str(&json)?;
        Ok(genome)
    }
}
