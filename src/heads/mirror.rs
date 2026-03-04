use std::collections::HashSet;
use crate::TrainingResult;
use super::Head;

pub struct Mirror;

impl Head for Mirror {
    fn name(&self) -> &'static str {
        "Mirror"
    }

    fn calculate_fitness(&self, result: &TrainingResult, _training_data: &HashSet<String>) -> f64 {
        let names = &result.names;
        if names.is_empty() { return -100.0; }
        
        let mut total = 0.0;
        let mut valid = 0;

        for name in names {
            let name = name.trim().to_lowercase();
            if name.len() < 3 { continue; }

            let mut score = 0.0;
            let chars: Vec<char> = name.chars().collect();

            // Palindrome
            if chars.iter().eq(chars.iter().rev()) { score += 3.0; }

            // Repeating halves (e.g., "baba")
            if name.len() >= 4 {
                let mid = name.len() / 2;
                if name[..mid] == name[mid..mid*2] { score += 2.0; }
            }

            // Rhyme-friendly endings
            if name.ends_with('a') || name.ends_with('n') || name.ends_with('y') { score += 0.5; }

            total += score;
            valid += 1;
        }

        if valid == 0 { -100.0 } else { total / valid as f64 }
    }
}
