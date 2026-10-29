//! Genome Management System
//! 
//! Handles saving, loading, and managing collections of 3x3 block genomes
//! for reuse in different assemblies.

use candle_core::{Device, Result};
use super::block_3x3::BlockGenome;
use std::collections::HashMap;
use std::path::Path;

/// Collection of block genomes with metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenomeCollection {
    /// Collection identifier
    pub collection_id: String,
    /// Individual block genomes
    pub genomes: Vec<BlockGenome>,
    /// Performance metrics for each genome
    pub metrics: HashMap<String, GenomeMetrics>,
    /// Training metadata
    pub training_info: TrainingInfo,
}

/// Performance metrics for a genome
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenomeMetrics {
    /// Accuracy on test set
    pub accuracy: f64,
    /// Loss on test set
    pub loss: f64,
    /// Training steps to convergence
    pub training_steps: usize,
    /// Training time in seconds
    pub training_time: f64,
    /// Date created
    pub created_at: String,
}

/// Training information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingInfo {
    /// Dataset used
    pub dataset: String,
    /// Training configuration
    pub config: TrainingConfig,
    /// Date trained
    pub trained_at: String,
}

/// Training configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub training_steps: usize,
    pub embedding_dim: usize,
    pub optimizer: String,
}

impl GenomeCollection {
    /// Create a new genome collection
    pub fn new(collection_id: &str, dataset: &str, config: TrainingConfig) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            genomes: Vec::new(),
            metrics: HashMap::new(),
            training_info: TrainingInfo {
                dataset: dataset.to_string(),
                config,
                trained_at: chrono::Utc::now().to_rfc3339(),
            },
        }
    }
    
    /// Add a genome to the collection
    pub fn add_genome(&mut self, genome: BlockGenome, metrics: GenomeMetrics) {
        self.metrics.insert(genome.block_id.clone(), metrics);
        self.genomes.push(genome);
    }
    
    /// Get genome by ID
    pub fn get_genome(&self, block_id: &str) -> Option<&BlockGenome> {
        self.genomes.iter().find(|g| g.block_id == block_id)
    }
    
    /// Get best performing genome
    pub fn get_best_genome(&self) -> Option<(&BlockGenome, &GenomeMetrics)> {
        self.genomes.iter()
            .filter_map(|g| {
                self.metrics.get(&g.block_id).map(|m| (g, m))
            })
            .max_by(|(_, a), (_, b)| a.accuracy.partial_cmp(&b.accuracy).unwrap())
    }
    
    /// Get genomes sorted by performance
    pub fn get_genomes_by_performance(&self) -> Vec<(&BlockGenome, &GenomeMetrics)> {
        let mut genomes: Vec<_> = self.genomes.iter()
            .filter_map(|g| {
                self.metrics.get(&g.block_id).map(|m| (g, m))
            })
            .collect();
        
        genomes.sort_by(|(_, a), (_, b)| b.accuracy.partial_cmp(&a.accuracy).unwrap());
        genomes
    }
    
    /// Save collection to file
    pub fn save_to_file(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Load collection from file
    pub fn load_from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let collection: GenomeCollection = serde_json::from_str(&json)?;
        Ok(collection)
    }
    
    /// Export genomes for assembly
    pub fn export_for_assembly(&self, block_ids: &[String]) -> std::result::Result<Vec<BlockGenome>, Box<dyn std::error::Error>> {
        let mut exported = Vec::new();
        for block_id in block_ids {
            if let Some(genome) = self.get_genome(block_id) {
                exported.push(genome.clone());
            } else {
                return Err(format!("Genome '{}' not found in collection", block_id).into());
            }
        }
        Ok(exported)
    }
    
    /// Get collection statistics
    pub fn get_statistics(&self) -> CollectionStats {
        let mut accuracies = Vec::new();
        let mut losses = Vec::new();
        let mut training_times = Vec::new();
        
        for metrics in self.metrics.values() {
            accuracies.push(metrics.accuracy);
            losses.push(metrics.loss);
            training_times.push(metrics.training_time);
        }
        
        CollectionStats {
            total_genomes: self.genomes.len(),
            avg_accuracy: accuracies.iter().sum::<f64>() / accuracies.len().max(1) as f64,
            best_accuracy: accuracies.iter().fold(0.0, |a, &b| a.max(b)),
            avg_loss: losses.iter().sum::<f64>() / losses.len().max(1) as f64,
            avg_training_time: training_times.iter().sum::<f64>() / training_times.len().max(1) as f64,
        }
    }
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub total_genomes: usize,
    pub avg_accuracy: f64,
    pub best_accuracy: f64,
    pub avg_loss: f64,
    pub avg_training_time: f64,
}

/// Genome manager for handling multiple collections
pub struct GenomeManager {
    collections: HashMap<String, GenomeCollection>,
    base_path: String,
}

impl GenomeManager {
    /// Create a new genome manager
    pub fn new(base_path: &str) -> Self {
        Self {
            collections: HashMap::new(),
            base_path: base_path.to_string(),
        }
    }
    
    /// Load all collections from base path
    pub fn load_collections(&mut self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let path = Path::new(&self.base_path);
        if !path.exists() {
            std::fs::create_dir_all(path)?;
            return Ok(());
        }
        
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let collection = GenomeCollection::load_from_file(path.to_str().unwrap())?;
                self.collections.insert(collection.collection_id.clone(), collection);
            }
        }
        
        Ok(())
    }
    
    /// Save a collection
    pub fn save_collection(&self, collection: &GenomeCollection) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let path = format!("{}/{}.json", self.base_path, collection.collection_id);
        collection.save_to_file(&path)
    }
    
    /// Get a collection by ID
    pub fn get_collection(&self, collection_id: &str) -> Option<&GenomeCollection> {
        self.collections.get(collection_id)
    }
    
    /// List all collections
    pub fn list_collections(&self) -> Vec<&String> {
        self.collections.keys().collect()
    }
    
    /// Create a new collection
    pub fn create_collection(&mut self, collection_id: &str, dataset: &str, config: TrainingConfig) -> &mut GenomeCollection {
        let collection = GenomeCollection::new(collection_id, dataset, config);
        self.collections.insert(collection_id.to_string(), collection);
        self.collections.get_mut(collection_id).unwrap()
    }
}
