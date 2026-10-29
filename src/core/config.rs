//! Core configuration management
//! 
//! Handles loading and validating experiment configurations

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub eval_frequency: usize,
    pub early_stopping: EarlyStoppingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_improvement: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArchitectureConfig {
    pub input_dim: usize,
    pub hidden_layers: Vec<usize>,
    pub output_dim: usize,
    pub activation: String,
    pub dropout: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub source: String,
    pub validation_split: f32,
    pub augmentation: Option<AugmentationConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AugmentationConfig {
    pub noise_std: f32,
    pub intensity_shift: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub save_frequency: usize,
    pub tensorboard: bool,
    pub debug: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptimizationConfig {
    pub optimizer: String,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NeuronConfig {
    pub training: TrainingConfig,
    pub architecture: ArchitectureConfig,
    pub data: DataConfig,
    pub logging: LoggingConfig,
    pub optimization: OptimizationConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BundleConfig {
    pub training: TrainingConfig,
    pub architecture: BundleArchitectureConfig,
    pub data: BundleDataConfig,
    pub evolution: Option<EvolutionConfig>,
    pub logging: LoggingConfig,
    pub optimization: OptimizationConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BundleArchitectureConfig {
    pub spatial_input_dim: usize,
    pub spatial_hidden_dim: usize,
    pub direction_output_dim: usize,
    pub neuron_count: usize,
    pub feature_engineering: FeatureEngineeringConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeatureEngineeringConfig {
    pub grayscale_features: bool,
    pub spatial_features: bool,
    pub global_features: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BundleDataConfig {
    pub source: String,
    pub validation_split: f32,
    pub neuron_models: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvolutionConfig {
    pub enabled: bool,
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f32,
    pub mutation_strength: f32,
    pub elite_ratio: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExperimentConfig {
    pub experiment: ExperimentMeta,
    pub pipeline: Vec<PipelineStage>,
    pub evaluation: EvaluationConfig,
    pub output: OutputConfig,
    pub system: SystemConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExperimentMeta {
    pub name: String,
    pub description: String,
    pub seed: u64,
    pub device: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineStage {
    pub stage: String,
    pub config: String,
    pub parallel: bool,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvaluationConfig {
    pub metrics: Vec<String>,
    pub test_datasets: Vec<String>,
    pub target_accuracy: f32,
    pub benchmark_comparison: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    pub save_models: bool,
    pub save_logs: bool,
    pub save_metrics: bool,
    pub export_format: Vec<String>,
    pub results_dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemConfig {
    pub max_memory_gb: usize,
    pub gpu_memory_fraction: f32,
    pub num_workers: usize,
    pub checkpoint_frequency: usize,
}

impl NeuronConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: NeuronConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.training.steps == 0 {
            return Err("Training steps must be > 0".to_string());
        }
        if self.training.batch_size == 0 {
            return Err("Batch size must be > 0".to_string());
        }
        if self.architecture.hidden_layers.is_empty() {
            return Err("At least one hidden layer required".to_string());
        }
        Ok(())
    }
}

impl BundleConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: BundleConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.training.steps == 0 {
            return Err("Training steps must be > 0".to_string());
        }
        if self.architecture.neuron_count != self.data.neuron_models.len() {
            return Err("Neuron count must match neuron models length".to_string());
        }
        Ok(())
    }
}

impl ExperimentConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: ExperimentConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.pipeline.is_empty() {
            return Err("Pipeline cannot be empty".to_string());
        }
        Ok(())
    }
}
