//! Color-aware 1x1 neuron species with RGB and HSV primitives
//! Extends basic grayscale with color detection capabilities

use candle_core::{Device, Result, Tensor};
use candle_nn::{VarBuilder, Linear, Module};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::core::config::{NeuronConfig, ArchitectureConfig};
use crate::core::logger::{Logger, ProgressTracker};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorNeuronConfig {
    pub base_config: NeuronConfig,
    pub color_space: ColorSpace,
    pub color_features: ColorFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorSpace {
    RGB,
    HSV,
    Lab,
    Combined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorFeatures {
    pub enable_rgb: bool,
    pub enable_hsv: bool,
    pub enable_intensity: bool,
    pub enable_saturation: bool,
    pub enable_hue_bins: bool,
    pub hue_bins: usize,
}

impl Default for ColorNeuronConfig {
    fn default() -> Self {
        Self {
            base_config: NeuronConfig::default(),
            color_space: ColorSpace::Combined,
            color_features: ColorFeatures {
                enable_rgb: true,
                enable_hsv: true,
                enable_intensity: true,
                enable_saturation: true,
                enable_hue_bins: true,
                hue_bins: 8,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorNeuronModel {
    pub layers: Vec<Linear>,
    pub device: Device,
    pub config: ColorNeuronConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorTrainingData {
    pub samples: Vec<ColorPixel>,
    pub labels: Vec<u8>,
    pub train_indices: Vec<usize>,
    pub val_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPixel {
    pub r: u8,
    pub g: u8, 
    pub b: u8,
    pub label: u8,
}

impl ColorPixel {
    pub fn new(r: u8, g: u8, b: u8, label: u8) -> Self {
        Self { r, g, b, label }
    }
    
    pub fn to_grayscale(&self) -> u8 {
        // Standard RGB to grayscale conversion
        (0.299 * self.r as f32 + 0.587 * self.g as f32 + 0.114 * self.b as f32) as u8
    }
    
    pub fn to_hsv(&self) -> (f32, f32, f32) {
        let r_norm = self.r as f32 / 255.0;
        let g_norm = self.g as f32 / 255.0;
        let b_norm = self.b as f32 / 255.0;
        
        let max = r_norm.max(g_norm).max(b_norm);
        let min = r_norm.min(g_norm).min(b_norm);
        let delta = max - min;
        
        // Value
        let v = max;
        
        // Saturation
        let s = if max == 0.0 { 0.0 } else { delta / max };
        
        // Hue
        let h = if delta == 0.0 {
            0.0
        } else {
            if max == r_norm {
                ((g_norm - b_norm) / delta + if g_norm < b_norm { 6.0 } else { 0.0 }) / 6.0
            } else if max == g_norm {
                ((b_norm - r_norm) / delta + 2.0) / 6.0
            } else {
                ((r_norm - g_norm) / delta + 4.0) / 6.0
            }
        };
        
        (h, s, v)
    }
}

pub struct ColorNeuronTrainer {
    pub config: ColorNeuronConfig,
    pub logger: Logger,
}

impl ColorNeuronTrainer {
    pub fn new(config: ColorNeuronConfig, logger: Logger) -> Self {
        Self { config, logger }
    }
    
    pub fn calculate_input_dim(&self) -> usize {
        let mut dim = 10; // Basic grayscale features
        
        if self.config.color_features.enable_rgb {
            dim += 3; // R, G, B
        }
        
        if self.config.color_features.enable_hsv {
            dim += 3; // H, S, V
        }
        
        if self.config.color_features.enable_intensity {
            dim += 1;
        }
        
        if self.config.color_features.enable_saturation {
            dim += 1;
        }
        
        if self.config.color_features.enable_hue_bins {
            dim += self.config.color_features.hue_bins;
        }
        
        dim
    }
    
    pub fn create_color_features(&self, pixel: &ColorPixel) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Basic grayscale one-hot (0-9)
        let gray = pixel.to_grayscale();
        for i in 0..10 {
            features.push(if gray == i as u8 { 1.0 } else { 0.0 });
        }
        
        // RGB features
        if self.config.color_features.enable_rgb {
            features.push(pixel.r as f32 / 255.0);
            features.push(pixel.g as f32 / 255.0);
            features.push(pixel.b as f32 / 255.0);
        }
        
        // HSV features
        if self.config.color_features.enable_hsv {
            let (h, s, v) = pixel.to_hsv();
            features.push(h);
            features.push(s);
            features.push(v);
        }
        
        // Intensity feature
        if self.config.color_features.enable_intensity {
            let intensity = (pixel.r as f32 + pixel.g as f32 + pixel.b as f32) / (3.0 * 255.0);
            features.push(intensity);
        }
        
        // Saturation feature
        if self.config.color_features.enable_saturation {
            let (_, s, _) = pixel.to_hsv();
            features.push(s);
        }
        
        // Hue bins
        if self.config.color_features.enable_hue_bins {
            let (h, _, _) = pixel.to_hsv();
            let bin_size = 1.0 / self.config.color_features.hue_bins as f32;
            for i in 0..self.config.color_features.hue_bins {
                let bin_start = i as f32 * bin_size;
                let bin_end = (i + 1) as f32 * bin_size;
                features.push(if h >= bin_start && h < bin_end { 1.0 } else { 0.0 });
            }
        }
        
        features
    }
    
    pub fn load_color_dataset(&self, file_path: &str) -> Result<ColorTrainingData> {
        let content = std::fs::read_to_string(file_path)?;
        let mut samples = Vec::new();
        let mut labels = Vec::new();
        
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            
            // Parse format: "r,g,b,label"
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 4 {
                let r = parts[0].parse::<u8>().unwrap_or(0);
                let g = parts[1].parse::<u8>().unwrap_or(0);
                let b = parts[2].parse::<u8>().unwrap_or(0);
                let label = parts[3].parse::<u8>().unwrap_or(0);
                
                samples.push(ColorPixel::new(r, g, b, label));
                labels.push(label);
            }
        }
        
        let total_samples = samples.len();
        let split_point = (total_samples as f32 * (1.0 - self.config.base_config.data.validation_split)) as usize;
        
        let train_indices = (0..split_point).collect();
        let val_indices = (split_point..total_samples).collect();
        
        Ok(ColorTrainingData {
            samples,
            labels,
            train_indices,
            val_indices,
        })
    }
    
    pub fn create_model(&self, vb: VarBuilder) -> Result<ColorNeuronModel> {
        let input_dim = self.calculate_input_dim();
        let mut layers = Vec::new();
        
        let mut prev_dim = input_dim;
        for &hidden_dim in &self.config.base_config.architecture.hidden_layers {
            layers.push(candle_nn::linear(prev_dim, hidden_dim, vb.pp("layer_0"))?);
            prev_dim = hidden_dim;
        }
        
        layers.push(candle_nn::linear(prev_dim, self.config.base_config.architecture.output_dim, vb.pp("output"))?);
        
        Ok(ColorNeuronModel {
            layers,
            device: vb.device().clone(),
            config: self.config.clone(),
        })
    }
    
    pub fn train(&self, model: &mut ColorNeuronModel, data: &ColorTrainingData) -> Result<f32> {
        // Training implementation similar to basic neuron but with color features
        let mut best_accuracy = 0.0f32;
        
        for step in 0..self.config.base_config.training.steps {
            // Create batch
            let batch_size = self.config.base_config.training.batch_size.min(data.train_indices.len());
            let batch_start = (step * batch_size) % data.train_indices.len();
            let batch_end = batch_start + batch_size;
            let batch_indices = if batch_end <= data.train_indices.len() {
                &data.train_indices[batch_start..batch_end]
            } else {
                &data.train_indices[batch_start..]
            };
            
            // Create color features batch
            let mut input_data = Vec::new();
            let mut target_data = Vec::new();
            
            for &idx in batch_indices {
                let pixel = &data.samples[idx];
                let features = self.create_color_features(pixel);
                input_data.extend(features);
                
                // One-hot target
                let mut target = vec![0.0f32; self.config.base_config.architecture.output_dim];
                target[data.labels[idx] as usize] = 1.0;
                target_data.extend(target);
            }
            
            // Forward pass
            let input = Tensor::from_vec(input_data, (batch_size, self.calculate_input_dim()), &model.device)?;
            let targets = Tensor::from_vec(target_data, (batch_size, self.config.base_config.architecture.output_dim), &model.device)?;
            
            let logits = self.forward(model, &input)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
            
            // Backward pass (simplified - would need optimizer)
            // optimizer.backward_step(&loss)?;
            
            // Calculate accuracy
            if step % 100 == 0 {
                let accuracy = self.calculate_accuracy(model, data)?;
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                }
            }
        }
        
        Ok(best_accuracy)
    }
    
    pub fn forward(&self, model: &ColorNeuronModel, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        
        for (i, layer) in model.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < model.layers.len() - 1 {
                x = x.relu()?;
            }
        }
        
        Ok(x)
    }
    
    pub fn calculate_accuracy(&self, model: &ColorNeuronModel, data: &ColorTrainingData) -> Result<f32> {
        let mut correct = 0;
        let mut total = 0;
        
        // Sample validation set
        for &idx in &data.val_indices.iter().take(1000).collect::<Vec<_>>() {
            let pixel = &data.samples[idx];
            let features = self.create_color_features(pixel);
            
            let input = Tensor::from_vec(features, (1, self.calculate_input_dim()), &model.device)?;
            let logits = self.forward(model, &input)?;
            let probs = logits.softmax(1)?;
            let predicted = probs.argmax(1)?.to_vec1::<u32>()?[0] as u8;
            
            if predicted == data.labels[idx] {
                correct += 1;
            }
            total += 1;
        }
        
        Ok(correct as f32 / total as f32)
    }
}
