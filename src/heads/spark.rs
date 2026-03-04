use std::collections::HashSet;
use crate::TrainingResult;
use super::Head;

pub struct Spark;

impl Head for Spark {
    fn name(&self) -> &'static str {
        "Spark"
    }

    fn calculate_fitness(&self, result: &TrainingResult, training_data: &HashSet<String>) -> f64 {
        let names = &result.names;
        if names.is_empty() { return -100.0; }
        
        let mut total = 0.0;
        let mut valid = 0;

        for name in names {
            let name = name.trim().to_lowercase();
            if name.len() < 3 { continue; }

            // Novelty is the only goal here
            if training_data.contains(&name) {
                total -= 5.0; // Memorization is death
            } else {
                total += 2.0; // Novelty is life
            }
            
            // Reward length variance (we don't want all short names)
            if name.len() > 6 { total += 0.5; }

            valid += 1;
        }

        if valid == 0 { -100.0 } else { total / valid as f64 }
    }
}
