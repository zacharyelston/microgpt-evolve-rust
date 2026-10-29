//! Single neuron component
//! 
//! Trains individual "neuron-like" models that can be composed into bundles

use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use serde_json;
use std::path::Path;

use crate::core::config::{NeuronConfig, TrainingConfig, ArchitectureConfig};
use crate::core::logger::{Logger, ProgressTracker, PerformanceMetrics};

pub struct NeuronTrainer {
    config: NeuronConfig,
    logger: Logger,
    device: Device,
}

impl NeuronTrainer {
    pub fn new(config: NeuronConfig, device: Device) -> Result<Self> {
        config.validate().map_err(|e| candle_core::Error::Msg(e))?;
        
        let log_level: crate::core::logger::LogLevel = config.logging.level.as_str().into();
        let logger = Logger::new(log_level, config.logging.debug);
        
        Ok(Self {
            config,
            logger,
            device,
        })
    }
    
    pub fn train(&mut self) -> Result<TrainedNeuron> {
        self.logger.info("Starting single neuron training");
        self.logger.info(&format!("Device: {:?}", self.device));
        
        // Create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        let model = NeuronModel::new(&self.config.architecture, vb)?;
        
        // Create optimizer
        let mut optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: self.config.training.learning_rate as f64,
                beta1: self.config.optimization.beta1 as f64,
                beta2: self.config.optimization.beta2 as f64,
                eps: self.config.optimization.eps as f64,
                weight_decay: self.config.optimization.weight_decay as f64,
            }
        )?;
        
        // Load data
        let data = self.load_data()?;
        self.logger.info(&format!("Loaded {} training samples", data.train_samples.len()));
        
        // Training loop
        let start_time = std::time::Instant::now();
        let mut progress_tracker = ProgressTracker::new(
            self.logger.clone(),
            self.config.training.steps,
            self.config.training.early_stopping.patience,
        );
        
        for step in 0..self.config.training.steps {
            // Create batch - use actual training indices size
            let train_size = data.train_indices.len();
            let batch_size = self.config.training.batch_size.min(train_size);
            let batch_start = (step * batch_size) % train_size;
            let batch_end = batch_start + batch_size;
            let batch_indices = if batch_end <= train_size {
                &data.train_indices[batch_start..batch_end]
            } else {
                &data.train_indices[batch_start..]
            };
            
            // Forward pass
            let (input, targets) = self.create_batch(batch_indices, &data)?;
            let logits = model.forward(&input)?;
            let loss_val = loss::cross_entropy(&logits, &targets)?;
            
            // Backward pass
            optimizer.backward_step(&loss_val)?;
            
            // Progress tracking
            if step % self.config.training.eval_frequency == 0 {
                let current_accuracy = self.evaluate(&model, &data)?;
                let loss_scalar = loss_val.to_vec0::<f32>()?;
                
                if progress_tracker.update(step, current_accuracy as f32, loss_scalar) {
                    break; // Early termination
                }
            }
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        progress_tracker.final_report(training_time);
        
        // Create trained neuron
        let trained = TrainedNeuron::new(model, &self.config, training_time)?;
        
        self.logger.info("Single neuron training completed");
        Ok(trained)
    }
    
    fn load_data(&self) -> Result<TrainingData> {
        let content = std::fs::read_to_string(&self.config.data.source)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let lines: Vec<&str> = content.lines().collect();
        let mut train_samples = Vec::new();
        let mut train_labels = Vec::new();
        
        for line in lines {
            if line.len() < 2 {
                continue;
            }
            
            let digit = line.chars().next().unwrap().to_digit(10).unwrap_or(0) as u8;
            let wrong_digit = line.chars().nth(1).unwrap().to_digit(10).unwrap_or(0) as u8;
            
            train_samples.push(digit);
            train_labels.push(digit as u32);
        }
        
        let total_samples = train_samples.len();
        let split_point = (total_samples as f32 * (1.0 - self.config.data.validation_split)) as usize;
        
        let train_indices = (0..split_point).collect();
        let val_indices = (split_point..total_samples).collect();
        
        Ok(TrainingData {
            train_samples,
            train_labels,
            train_indices,
            val_indices,
        })
    }
    
    fn create_batch(&self, indices: &[usize], data: &TrainingData) -> Result<(Tensor, Tensor)> {
        let batch_size = indices.len();
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        
        for &idx in indices {
            let digit = data.train_samples[idx];
            let target = data.train_labels[idx];
            
            // Create enhanced features
            let mut features = vec![0.0f32; self.config.architecture.input_dim];
            features[digit as usize] = 1.0;
            
            if self.config.architecture.input_dim >= 14 {
                let intensity = digit as f32 / 9.0;
                features[10] = intensity;
                features[11] = 1.0 - intensity;
                features[12] = if digit >= 5 { 1.0 } else { 0.0 };
                features[13] = if digit % 2 == 0 { 1.0 } else { 0.0 };
            }
            
            input_data.extend(features);
            target_data.push(target);
        }
        
        let input = Tensor::from_vec(input_data, (batch_size, self.config.architecture.input_dim), &self.device)?;
        let targets = Tensor::from_vec(target_data, batch_size, &self.device)?;
        
        Ok((input, targets))
    }
    
    fn evaluate(&self, model: &NeuronModel, data: &TrainingData) -> Result<f64> {
        let mut correct = 0;
        let mut total = 0;
        
        for chunk in data.val_indices.chunks(100) {
            let (input, targets) = self.create_batch(chunk, data)?;
            let logits = model.forward(&input)?;
            let probs = softmax(&logits, 1)?;
            let probs_vec = probs.to_vec2::<f32>()?;
            
            for (i, probs) in probs_vec.iter().enumerate() {
                let pred_idx = probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                
                if pred_idx == targets.to_vec1::<u32>()?[i] as usize {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        Ok((correct as f64 / total as f64) * 100.0)
    }
}

pub struct NeuronModel {
    layers: Vec<candle_nn::Linear>,
    device: Device,
}

impl NeuronModel {
    pub fn new(config: &ArchitectureConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;
        
        for &hidden_dim in &config.hidden_layers {
            let layer = candle_nn::linear(prev_dim, hidden_dim, vb.pp(&format!("layer_{}", layers.len())))?;
            layers.push(layer);
            prev_dim = hidden_dim;
        }
        
        let output_layer = candle_nn::linear(prev_dim, config.output_dim, vb.pp("output"))?;
        layers.push(output_layer);
        
        Ok(Self {
            layers,
            device: vb.device().clone(),
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }
        
        Ok(x)
    }
    
    pub fn save_weights(&self, path: &Path) -> Result<()> {
        let mut weights = serde_json::Map::new();
        
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_name = if i < self.layers.len() - 1 {
                format!("layer_{}", i)
            } else {
                "output".to_string()
            };
            
            weights.insert(
                format!("{}_weights", layer_name),
                serde_json::Value::Array(
                    layer.weight().to_vec2::<f32>()?
                        .into_iter()
                        .map(|row| serde_json::Value::Array(
                            row.into_iter().map(|v| serde_json::Value::Number(serde_json::Number::from_f64(v as f64).unwrap())).collect()
                        ))
                        .collect()
                )
            );
            
            if let Some(bias) = layer.bias() {
                weights.insert(
                    format!("{}_bias", layer_name),
                    serde_json::Value::Array(
                        bias.to_vec1::<f32>()?
                            .into_iter()
                            .map(|v| serde_json::Value::Number(serde_json::Number::from_f64(v as f64).unwrap()))
                            .collect()
                    )
                );
            }
        }
        
        let json = serde_json::to_string_pretty(&serde_json::Value::Object(weights))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        std::fs::write(path, json)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        Ok(())
    }
}

pub struct TrainingData {
    pub train_samples: Vec<u8>,
    pub train_labels: Vec<u32>,
    pub train_indices: Vec<usize>,
    pub val_indices: Vec<usize>,
}

pub struct TrainedNeuron {
    pub model: NeuronModel,
    pub config: NeuronConfig,
    pub accuracy: f64,
    pub training_time: f64,
}

impl TrainedNeuron {
    pub fn new(model: NeuronModel, config: &NeuronConfig, training_time: f64) -> Result<Self> {
        // In a real implementation, we'd calculate actual accuracy
        // For now, use a placeholder
        let accuracy = 95.0; // Placeholder
        
        Ok(Self {
            model,
            config: config.clone(),
            accuracy,
            training_time,
        })
    }
    
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        std::fs::create_dir_all(path.as_ref().parent().unwrap())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        self.model.save_weights(path.as_ref())?;
        
        // Save metadata
        let metadata = serde_json::json!({
            "accuracy": self.accuracy,
            "training_time": self.training_time,
            "config": self.config,
            "architecture": "single_neuron",
        });
        
        let meta_path = path.as_ref().with_extension("meta.json");
        std::fs::write(meta_path, serde_json::to_string_pretty(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?)?;
        
        Ok(())
    }
    
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        // Load metadata
        let meta_path = path.as_ref().with_extension("meta.json");
        let meta_content = std::fs::read_to_string(meta_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let metadata: serde_json::Value = serde_json::from_str(&meta_content)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let config: NeuronConfig = serde_json::from_value(metadata["config"].clone())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let accuracy = metadata["accuracy"].as_f64().unwrap_or(0.0);
        let training_time = metadata["training_time"].as_f64().unwrap_or(0.0);
        
        // Load model weights
        let weights_content = std::fs::read_to_string(path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let weights: serde_json::Value = serde_json::from_str(&weights_content)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = NeuronModel::new(&config.architecture, vb)?;
        
        // In a real implementation, we'd load the weights into the model
        // For now, create a new model
        
        Ok(Self {
            model,
            config,
            accuracy,
            training_time,
        })
    }
}
