use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod assembly;
pub mod grid_config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMetadata {
    pub name: String,
    pub version: String,
    pub category: String,
    pub author: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorCore {
    pub n_emb: usize,
    pub n_ctx: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff_exp: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plasticity {
    pub l1_prune_penalty: f64,
    pub prune_threshold: f64,
    pub max_surviving_synapses: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    pub evaluator_core: EvaluatorCore,
    pub plasticity: Plasticity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interface {
    pub input_type: String,
    pub output_type: String,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "binding_strategy")]
pub enum UpstreamNodes {
    #[serde(rename = "static_allocation")]
    StaticAllocation { nodes: HashMap<String, StaticNode> },
    #[serde(rename = "dynamic_query")]
    DynamicQuery { queries: Vec<DynamicQuery> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticNode {
    pub path: String,
    pub n_ctx: usize,
    pub slice: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicQuery {
    pub path: String,
    pub select: String,
    pub k: usize,
    pub metric: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bundle {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub kind: String,
    pub metadata: BundleMetadata,
    pub architecture: Architecture,
    pub interface: Interface,
    pub upstream_nodes: UpstreamNodes,
}

impl Bundle {
    pub fn from_yaml_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let first_doc = contents.split("\n---").next().unwrap_or(&contents);
        let bundle: Bundle = serde_yaml::from_str(first_doc)?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.api_version != "microgpt.evolve/v1alpha1" {
            return Err(format!("Unsupported API version: {}", self.api_version));
        }
        if self.kind != "Bundle" {
            return Err(format!("Expected kind 'Bundle', got '{}'", self.kind));
        }
        if self.architecture.evaluator_core.n_emb % self.architecture.evaluator_core.n_head != 0 {
            return Err(format!(
                "n_emb ({}) must be divisible by n_head ({})",
                self.architecture.evaluator_core.n_emb,
                self.architecture.evaluator_core.n_head
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalHyperparameters {
    pub rms_eps: f64,
    pub init_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hemisphere {
    pub bundle_ref: String,
    pub input_stream: Option<String>,
    pub upstream_target: Option<String>,
    pub upstream_targets: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    #[serde(flatten)]
    pub hemispheres: HashMap<String, Hemisphere>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology {
    #[serde(flatten)]
    pub layers: HashMap<String, Layer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroStackMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuroStack {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub kind: String,
    pub metadata: NeuroStackMetadata,
    pub global_hyperparameters: GlobalHyperparameters,
    pub topology: Topology,
}

impl NeuroStack {
    pub fn from_yaml_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let stack: NeuroStack = serde_yaml::from_str(&contents)?;
        Ok(stack)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.api_version != "microgpt.evolve/v1alpha1" {
            return Err(format!("Unsupported API version: {}", self.api_version));
        }
        if self.kind != "Neurostack" {
            return Err(format!("Expected kind 'Neurostack', got '{}'", self.kind));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_validation() {
        let bundle = Bundle {
            api_version: "microgpt.evolve/v1alpha1".to_string(),
            kind: "Bundle".to_string(),
            metadata: BundleMetadata {
                name: "test_bundle".to_string(),
                version: "1.0.0".to_string(),
                category: "test".to_string(),
                author: "test".to_string(),
                description: None,
            },
            architecture: Architecture {
                evaluator_core: EvaluatorCore {
                    n_emb: 16,
                    n_ctx: 16,
                    n_layer: 1,
                    n_head: 4,
                    n_ff_exp: 4,
                },
                plasticity: Plasticity {
                    l1_prune_penalty: 0.01,
                    prune_threshold: 0.05,
                    max_surviving_synapses: 4,
                },
            },
            interface: Interface {
                input_type: "test".to_string(),
                output_type: "test".to_string(),
                output_dim: 5,
            },
            upstream_nodes: UpstreamNodes::StaticAllocation {
                nodes: HashMap::new(),
            },
        };

        assert!(bundle.validate().is_ok());
    }
}
