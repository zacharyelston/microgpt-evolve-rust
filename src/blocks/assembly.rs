//! Block Assembly System
//! 
//! Assembles multiple 3x3 vision blocks into larger grids with configurable
//! layouts and shared evaluators.

use candle_core::{Device, Result, Tensor, Module, DType};
use candle_nn::{VarBuilder, VarMap, Linear};
use super::block_3x3::VisionBlock3x3;

#[derive(Debug, Clone)]
pub struct BlockAssembly {
    /// Individual 3x3 blocks
    pub blocks: Vec<VisionBlock3x3>,
    /// Layout specification (rows, cols)
    pub layout: (usize, usize),
    /// Evaluator for combining block outputs
    pub evaluator: Linear,
    /// Device where assembly runs
    pub device: Device,
    /// Assembly identifier
    pub assembly_id: String,
}

impl BlockAssembly {
    /// Create a new assembly with specified layout
    pub fn new(assembly_id: &str, layout: (usize, usize), embedding_dim: usize, evaluator_dim: usize, device: &Device) -> Result<Self> {
        let (rows, cols) = layout;
        let num_blocks = rows * cols;
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // Create individual blocks
        let mut blocks = Vec::new();
        for i in 0..num_blocks {
            let row = i / cols;
            let col = i % cols;
            let block_id = format!("{}_block_{}_{}", assembly_id, row, col);
            let block = VisionBlock3x3::new(&block_id, embedding_dim, device)?;
            blocks.push(block);
        }
        
        // Evaluator combines all block outputs
        let evaluator_input_dim = num_blocks * embedding_dim;
        let evaluator = candle_nn::linear(evaluator_input_dim, 9, vb.pp("evaluator"))?; // 9 directions
        
        Ok(Self {
            blocks,
            layout,
            evaluator,
            device: device.clone(),
            assembly_id: assembly_id.to_string(),
        })
    }
    
    /// Forward pass through the entire assembly
    pub fn forward(&self, grid_data: &[&[u8]]) -> Result<Tensor> {
        if grid_data.len() != self.blocks.len() {
            return Err(candle_core::Error::Msg(format!(
                "Expected {} blocks, got {}", self.blocks.len(), grid_data.len()
            )));
        }
        
        // Process each block
        let mut block_outputs = Vec::new();
        for (block, grid) in self.blocks.iter().zip(grid_data.iter()) {
            let output = block.forward(grid)?;
            block_outputs.push(output);
        }
        
        // Combine all block outputs
        let combined = Tensor::stack(&block_outputs, 0)?;
        let flattened = combined.flatten_all()?;
        
        // Forward through evaluator
        let logits = self.evaluator.forward(&flattened.unsqueeze(0)?)?;
        logits.squeeze(0)
    }
    
    /// Get all parameters for optimization
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        
        params.push(self.evaluator.weight().clone());
        if let Some(bias) = self.evaluator.bias() {
            params.push(bias.clone());
        }
        
        params
    }
    
    /// Get block at specific position
    pub fn get_block(&self, row: usize, col: usize) -> Option<&VisionBlock3x3> {
        let index = row * self.layout.1 + col;
        self.blocks.get(index)
    }
    
    /// Replace block at specific position
    pub fn replace_block(&mut self, row: usize, col: usize, new_block: VisionBlock3x3) -> Result<()> {
        let index = row * self.layout.1 + col;
        if index >= self.blocks.len() {
            return Err(candle_core::Error::Msg("Block position out of bounds".to_string()));
        }
        self.blocks[index] = new_block;
        Ok(())
    }
    
    /// Get assembly dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        self.layout
    }
    
    /// Get total number of blocks
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
    
    /// Create assembly from saved block genomes
    pub fn from_genomes(assembly_id: &str, genomes: &[super::block_3x3::BlockGenome], evaluator_dim: usize, device: &Device) -> Result<Self> {
        let num_blocks = genomes.len();
        let layout = (num_blocks, 1); // Default to single column layout
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // Create blocks from genomes
        let mut blocks = Vec::new();
        for genome in genomes {
            let block = VisionBlock3x3::from_genome(genome, device)?;
            blocks.push(block);
        }
        
        // Create evaluator
        let embedding_dim = genomes[0].embedding_dim;
        let evaluator_input_dim = num_blocks * embedding_dim;
        let evaluator = candle_nn::linear(evaluator_input_dim, 9, vb.pp("evaluator"))?;
        
        Ok(Self {
            blocks,
            layout,
            evaluator,
            device: device.clone(),
            assembly_id: assembly_id.to_string(),
        })
    }
}

/// Layout configurations for different grid sizes
#[derive(Debug, Clone)]
pub enum LayoutConfig {
    /// Single 3x3 block
    Single,
    /// 2x2 grid of 3x3 blocks (6x6 total)
    Grid2x2,
    /// 3x3 grid of 3x3 blocks (9x9 total)
    Grid3x3,
    /// 4x4 grid of 3x3 blocks (12x12 total)
    Grid4x4,
    /// Custom layout
    Custom { rows: usize, cols: usize },
}

impl LayoutConfig {
    /// Get dimensions for this layout
    pub fn dimensions(&self) -> (usize, usize) {
        match self {
            LayoutConfig::Single => (1, 1),
            LayoutConfig::Grid2x2 => (2, 2),
            LayoutConfig::Grid3x3 => (3, 3),
            LayoutConfig::Grid4x4 => (4, 4),
            LayoutConfig::Custom { rows, cols } => (*rows, *cols),
        }
    }
    
    /// Get total grid size for this layout
    pub fn total_grid_size(&self) -> usize {
        let (rows, cols) = self.dimensions();
        rows * cols * 3 // Each block is 3x3
    }
}
