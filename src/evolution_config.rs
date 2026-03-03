/*
    Evolution Configuration System
    
    This module provides configuration-driven parameter ranges
    for the evolution engine, allowing experimental control
    over min/max values for all hyperparameters.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRange {
    pub min: usize,
    pub max: usize,
    #[serde(default)]
    pub values: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrRange {
    pub min: f64,
    pub max: f64,
    #[serde(default)]
    pub log_scale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepsRange {
    pub min: usize,
    pub max: usize,
    #[serde(default)]
    pub values: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRanges {
    pub n_emb: ParameterRange,
    pub n_ctx: ParameterRange,
    pub n_layer: ParameterRange,
    pub n_head: ParameterRange,
    pub n_ff_exp: ParameterRange,
    pub lr: LrRange,
    pub steps: StepsRange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSettings {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub target_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisConfig {
    pub n_emb: usize,
    pub n_ctx: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff_exp: usize,
    pub lr: f64,
    pub steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub parameter_ranges: ParameterRanges,
    pub evolution_settings: EvolutionSettings,
    pub genesis_config: GenesisConfig,
}

impl EvolutionConfig {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: EvolutionConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn load_or_default(path: &str) -> Self {
        match Self::load_from_file(path) {
            Ok(config) => {
                println!("✅ Loaded evolution config from {}", path);
                config
            }
            Err(e) => {
                println!("⚠️  Failed to load config from {}: {}, using defaults", path, e);
                Self::default()
            }
        }
    }

    pub fn get_n_emb_choices(&self) -> Vec<usize> {
        self.parameter_ranges.n_emb.values
            .clone()
            .unwrap_or_else(|| (self.parameter_ranges.n_emb.min..=self.parameter_ranges.n_emb.max).collect())
    }

    pub fn get_n_ctx_choices(&self) -> Vec<usize> {
        self.parameter_ranges.n_ctx.values
            .clone()
            .unwrap_or_else(|| (self.parameter_ranges.n_ctx.min..=self.parameter_ranges.n_ctx.max).collect())
    }

    pub fn get_n_head_choices(&self) -> Vec<usize> {
        self.parameter_ranges.n_head.values
            .clone()
            .unwrap_or_else(|| (self.parameter_ranges.n_head.min..=self.parameter_ranges.n_head.max).collect())
    }

    pub fn get_steps_choices(&self) -> Vec<usize> {
        self.parameter_ranges.steps.values
            .clone()
            .unwrap_or_else(|| (self.parameter_ranges.steps.min..=self.parameter_ranges.steps.max).collect())
    }

    pub fn random_lr(&self) -> f64 {
        if self.parameter_ranges.lr.log_scale {
            let log_min = self.parameter_ranges.lr.min.log10();
            let log_max = self.parameter_ranges.lr.max.log10();
            let random_log = rand::random::<f64>() * (log_max - log_min) + log_min;
            10.0_f64.powf(random_log)
        } else {
            rand::random::<f64>() * (self.parameter_ranges.lr.max - self.parameter_ranges.lr.min) + self.parameter_ranges.lr.min
        }
    }

    pub fn random_n_layer(&self) -> usize {
        rand::random::<usize>() % (self.parameter_ranges.n_layer.max - self.parameter_ranges.n_layer.min + 1) + self.parameter_ranges.n_layer.min
    }

    pub fn random_n_ff_exp(&self) -> usize {
        rand::random::<usize>() % (self.parameter_ranges.n_ff_exp.max - self.parameter_ranges.n_ff_exp.min + 1) + self.parameter_ranges.n_ff_exp.min
    }

    pub fn validate_config(&self) -> Result<(), String> {
        // Validate parameter ranges are logical
        if self.parameter_ranges.n_emb.min > self.parameter_ranges.n_emb.max {
            return Err("n_emb min > max".to_string());
        }
        if self.parameter_ranges.n_ctx.min > self.parameter_ranges.n_ctx.max {
            return Err("n_ctx min > max".to_string());
        }
        if self.parameter_ranges.n_layer.min > self.parameter_ranges.n_layer.max {
            return Err("n_layer min > max".to_string());
        }
        if self.parameter_ranges.n_head.min > self.parameter_ranges.n_head.max {
            return Err("n_head min > max".to_string());
        }
        if self.parameter_ranges.n_ff_exp.min > self.parameter_ranges.n_ff_exp.max {
            return Err("n_ff_exp min > max".to_string());
        }
        if self.parameter_ranges.lr.min > self.parameter_ranges.lr.max {
            return Err("lr min > max".to_string());
        }
        if self.parameter_ranges.steps.min > self.parameter_ranges.steps.max {
            return Err("steps min > max".to_string());
        }

        // Validate genesis config is within ranges
        if self.genesis_config.n_emb < self.parameter_ranges.n_emb.min || self.genesis_config.n_emb > self.parameter_ranges.n_emb.max {
            return Err(format!("genesis n_emb {} out of range [{}, {}]", 
                self.genesis_config.n_emb, self.parameter_ranges.n_emb.min, self.parameter_ranges.n_emb.max));
        }
        if self.genesis_config.n_ctx < self.parameter_ranges.n_ctx.min || self.genesis_config.n_ctx > self.parameter_ranges.n_ctx.max {
            return Err(format!("genesis n_ctx {} out of range [{}, {}]", 
                self.genesis_config.n_ctx, self.parameter_ranges.n_ctx.min, self.parameter_ranges.n_ctx.max));
        }

        Ok(())
    }
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            parameter_ranges: ParameterRanges {
                n_emb: ParameterRange { min: 1, max: 12, values: Some(vec![1, 2, 3, 4, 6, 8, 10, 12]) },
                n_ctx: ParameterRange { min: 1, max: 16, values: Some(vec![1, 2, 3, 4, 6, 8, 12, 16]) },
                n_layer: ParameterRange { min: 1, max: 3, values: None },
                n_head: ParameterRange { min: 1, max: 2, values: Some(vec![1, 2]) },
                n_ff_exp: ParameterRange { min: 1, max: 3, values: None },
                lr: LrRange { min: 0.001, max: 0.1, log_scale: true },
                steps: StepsRange { min: 200, max: 1000, values: Some(vec![200, 400, 600, 800, 1000]) },
            },
            evolution_settings: EvolutionSettings {
                population_size: 12,
                generations: 20,
                mutation_rate: 0.3,
                target_loss: 1.2,
            },
            genesis_config: GenesisConfig {
                n_emb: 1,
                n_ctx: 1,
                n_layer: 1,
                n_head: 1,
                n_ff_exp: 1,
                lr: 0.01,
                steps: 1000,
            },
        }
    }
}
