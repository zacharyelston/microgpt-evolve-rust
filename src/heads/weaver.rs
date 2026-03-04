use std::collections::HashSet;
use crate::TrainingResult;
use super::Head;

pub struct Weaver;

impl Head for Weaver {
    fn name(&self) -> &'static str {
        "Weaver"
    }

    fn calculate_fitness(&self, result: &TrainingResult, _training_data: &HashSet<String>) -> f64 {
        let names = &result.names;
        if names.is_empty() { return -100.0; }
        
        let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u', 'y'].iter().cloned().collect();
        
        let mut total = 0.0;
        let mut valid = 0;

        for name in names {
            let name = name.trim().to_lowercase();
            // Basic validity check
            if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) { continue; }

            let mut score = 0.0;
            let mut cons_v = 0;
            let mut cons_c = 0;

            for c in name.chars() {
                if vowels.contains(&c) {
                    cons_v += 1;
                    cons_c = 0;
                } else {
                    cons_c += 1;
                    cons_v = 0;
                }
                // Penalize clusters of 3+ vowels or consonants
                if cons_v > 2 || cons_c > 2 { score -= 1.0; }
            }
            
            // Reward ideal length (4-8 chars)
            if name.len() >= 4 && name.len() <= 8 { score += 1.0; }
            
            total += score;
            valid += 1;
        }
        
        if valid == 0 { -100.0 } else { total / valid as f64 }
    }
}
