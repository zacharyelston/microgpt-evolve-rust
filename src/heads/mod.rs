use std::collections::HashSet;
use crate::TrainingResult;

pub mod weaver;
pub mod mirror;
pub mod spark;
pub mod origin;

pub trait Head: Send + Sync {
    fn name(&self) -> &'static str;
    fn calculate_fitness(&self, result: &TrainingResult, training_data: &HashSet<String>) -> f64;
}

// Re-export specific heads for easier access
pub use weaver::Weaver;
pub use mirror::Mirror;
pub use spark::Spark;
pub use origin::Origin;
