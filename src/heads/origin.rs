use std::collections::HashSet;
use crate::TrainingResult;
use super::Head;

pub struct Origin;

impl Head for Origin {
    fn name(&self) -> &'static str {
        "Origin"
    }

    fn calculate_fitness(&self, result: &TrainingResult, _training_data: &HashSet<String>) -> f64 {
        // Original MicroGPT goal: minimize loss.
        // We invert loss to get fitness (lower loss = higher fitness).
        1.0 / result.final_loss.max(0.0001)
    }
}
