use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub grid_size: usize,
    pub quadrant_size: usize,
    pub quadrants_per_row: usize,
    pub total_quadrants: usize,
}

impl GridConfig {
    pub fn new(grid_size: usize, quadrant_size: usize) -> Self {
        assert!(grid_size % quadrant_size == 0, "Grid size must be divisible by quadrant size");
        let quadrants_per_row = grid_size / quadrant_size;
        let total_quadrants = quadrants_per_row * quadrants_per_row;
        
        GridConfig {
            grid_size,
            quadrant_size,
            quadrants_per_row,
            total_quadrants,
        }
    }
    
    pub fn quadrant_3x3() -> Self {
        Self::new(3, 1)
    }
    
    pub fn quadrant_8x8() -> Self {
        Self::new(8, 4)
    }
    
    pub fn quadrant_12x12() -> Self {
        Self::new(12, 3)
    }
    
    pub fn quadrant_16x16() -> Self {
        Self::new(16, 4)
    }
    
    pub fn quadrant_32x32() -> Self {
        Self::new(32, 8)
    }
    
    pub fn get_quadrant_slice(&self, quadrant_idx: usize) -> [usize; 4] {
        let row = quadrant_idx / self.quadrants_per_row;
        let col = quadrant_idx % self.quadrants_per_row;
        
        let x_start = col * self.quadrant_size;
        let x_end = x_start + self.quadrant_size;
        let y_start = row * self.quadrant_size;
        let y_end = y_start + self.quadrant_size;
        
        [x_start, x_end, y_start, y_end]
    }
    
    pub fn extract_quadrant(&self, tokens: &[usize], quadrant_idx: usize) -> Vec<usize> {
        let slice = self.get_quadrant_slice(quadrant_idx);
        let x_start = slice[0];
        let x_end = slice[1];
        let y_start = slice[2];
        let y_end = slice[3];
        
        let mut quadrant = Vec::new();
        for y in y_start..y_end {
            for x in x_start..x_end {
                let idx = y * self.grid_size + x;
                if idx < tokens.len() {
                    quadrant.push(tokens[idx]);
                }
            }
        }
        quadrant
    }
    
    pub fn total_pixels(&self) -> usize {
        self.grid_size * self.grid_size
    }
    
    pub fn pixels_per_quadrant(&self) -> usize {
        self.quadrant_size * self.quadrant_size
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BundlePattern {
    Uniform {
        n_emb: usize,
        n_ctx: usize,
        n_layer: usize,
        n_head: usize,
        n_ff_exp: usize,
    },
    CenterFocus {
        center_n_emb: usize,
        center_n_layer: usize,
        peripheral_n_emb: usize,
        peripheral_n_layer: usize,
        n_head: usize,
        n_ff_exp: usize,
    },
    Custom {
        quadrant_configs: Vec<QuadrantConfig>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadrantConfig {
    pub quadrant_idx: usize,
    pub n_emb: usize,
    pub n_ctx: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff_exp: usize,
}

impl BundlePattern {
    pub fn get_quadrant_config(&self, quadrant_idx: usize, grid_config: &GridConfig) -> (usize, usize, usize, usize, usize) {
        match self {
            BundlePattern::Uniform { n_emb, n_ctx, n_layer, n_head, n_ff_exp } => {
                (*n_emb, *n_ctx, *n_layer, *n_head, *n_ff_exp)
            }
            BundlePattern::CenterFocus { 
                center_n_emb, center_n_layer, 
                peripheral_n_emb, peripheral_n_layer,
                n_head, n_ff_exp 
            } => {
                let is_center = Self::is_center_quadrant(quadrant_idx, grid_config);
                let n_ctx = grid_config.pixels_per_quadrant();
                
                if is_center {
                    (*center_n_emb, n_ctx, *center_n_layer, *n_head, *n_ff_exp)
                } else {
                    (*peripheral_n_emb, n_ctx, *peripheral_n_layer, *n_head, *n_ff_exp)
                }
            }
            BundlePattern::Custom { quadrant_configs } => {
                if let Some(config) = quadrant_configs.iter().find(|c| c.quadrant_idx == quadrant_idx) {
                    (config.n_emb, config.n_ctx, config.n_layer, config.n_head, config.n_ff_exp)
                } else {
                    (16, grid_config.pixels_per_quadrant(), 1, 4, 4)
                }
            }
        }
    }
    
    fn is_center_quadrant(quadrant_idx: usize, grid_config: &GridConfig) -> bool {
        let row = quadrant_idx / grid_config.quadrants_per_row;
        let col = quadrant_idx % grid_config.quadrants_per_row;
        let qpr = grid_config.quadrants_per_row;
        
        if qpr < 2 {
            return true;
        }
        
        let center_start = qpr / 4;
        let center_end = qpr - center_start;
        
        row >= center_start && row < center_end && col >= center_start && col < center_end
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub grid_config: GridConfig,
    pub bundle_pattern: BundlePattern,
    pub evaluator_n_emb: usize,
    pub evaluator_n_layer: usize,
    pub evaluator_n_head: usize,
    pub l1_penalty: f64,
    pub prune_threshold: f64,
    pub max_synapses: usize,
    pub training_steps: usize,
    pub learning_rate: f64,
}

impl ExperimentConfig {
    pub fn eyespot_3x3() -> Self {
        ExperimentConfig {
            name: "eyespot_3x3_building_block".to_string(),
            grid_config: GridConfig::quadrant_3x3(),
            bundle_pattern: BundlePattern::Uniform {
                n_emb: 16,
                n_ctx: 9,
                n_layer: 1,
                n_head: 4,
                n_ff_exp: 4,
            },
            evaluator_n_emb: 24,
            evaluator_n_layer: 1,
            evaluator_n_head: 4,
            l1_penalty: 0.01,
            prune_threshold: 0.05,
            max_synapses: 1,
            training_steps: 5000,
            learning_rate: 0.02,
        }
    }
    
    pub fn eyespot_8x8() -> Self {
        ExperimentConfig {
            name: "eyespot_8x8_uniform".to_string(),
            grid_config: GridConfig::quadrant_8x8(),
            bundle_pattern: BundlePattern::Uniform {
                n_emb: 16,
                n_ctx: 16,
                n_layer: 1,
                n_head: 4,
                n_ff_exp: 4,
            },
            evaluator_n_emb: 16,
            evaluator_n_layer: 1,
            evaluator_n_head: 4,
            l1_penalty: 0.01,
            prune_threshold: 0.05,
            max_synapses: 4,
            training_steps: 3000,
            learning_rate: 0.01,
        }
    }
    
        
    pub fn eyespot_12x12() -> Self {
        ExperimentConfig {
            name: "eyespot_12x12_uniform".to_string(),
            grid_config: GridConfig::quadrant_12x12(),
            bundle_pattern: BundlePattern::Uniform {
                n_emb: 16,
                n_ctx: 9,
                n_layer: 1,
                n_head: 4,
                n_ff_exp: 4,
            },
            evaluator_n_emb: 24,
            evaluator_n_layer: 1,
            evaluator_n_head: 4,
            l1_penalty: 0.01,
            prune_threshold: 0.05,
            max_synapses: 16,
            training_steps: 2000,
            learning_rate: 0.015,
        }
    }
    
    pub fn eyespot_16x16() -> Self {
        ExperimentConfig {
            name: "eyespot_16x16_uniform".to_string(),
            grid_config: GridConfig::quadrant_16x16(),
            bundle_pattern: BundlePattern::Uniform {
                n_emb: 16,
                n_ctx: 16,
                n_layer: 1,
                n_head: 4,
                n_ff_exp: 4,
            },
            evaluator_n_emb: 24,
            evaluator_n_layer: 2,
            evaluator_n_head: 4,
            l1_penalty: 0.01,
            prune_threshold: 0.05,
            max_synapses: 16,
            training_steps: 2000,
            learning_rate: 0.015,
        }
    }
    
    pub fn eyespot_32x32_center_focus() -> Self {
        ExperimentConfig {
            name: "eyespot_32x32_center_focus".to_string(),
            grid_config: GridConfig::quadrant_32x32(),
            bundle_pattern: BundlePattern::CenterFocus {
                center_n_emb: 32,
                center_n_layer: 2,
                peripheral_n_emb: 16,
                peripheral_n_layer: 1,
                n_head: 4,
                n_ff_exp: 4,
            },
            evaluator_n_emb: 32,
            evaluator_n_layer: 2,
            evaluator_n_head: 4,
            l1_penalty: 0.005,
            prune_threshold: 0.05,
            max_synapses: 16,
            training_steps: 1000,
            learning_rate: 0.02,
        }
    }
}
