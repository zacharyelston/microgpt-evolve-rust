use rand::Rng;
use std::fs::File;
use std::io::Write;

fn generate_gradient(direction: &str, intensity: f64) -> Vec<u8> {
    let mut grid = vec![vec![0u8; 12]; 12];
    
    for y in 0..12 {
        for x in 0..12 {
            let brightness = match direction {
                "L" => {
                    let dist = x as f64 / 11.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                "R" => {
                    let dist = x as f64 / 11.0;
                    (dist * intensity * 9.0) as u8
                }
                "T" => {
                    let dist = y as f64 / 11.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                "B" => {
                    let dist = y as f64 / 11.0;
                    (dist * intensity * 9.0) as u8
                }
                "U" => {
                    (intensity * 5.0) as u8
                }
                _ => 0,
            };
            grid[y][x] = brightness.min(9);
        }
    }
    
    grid.into_iter().flatten().collect()
}

fn add_noise(grid: &mut [u8], noise_level: f64) {
    let mut rng = rand::thread_rng();
    for pixel in grid.iter_mut() {
        let noise = rng.gen_range(-noise_level..noise_level);
        let noisy = (*pixel as f64 + noise).max(0.0).min(9.0);
        *pixel = noisy as u8;
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let output_file = "eyespot_training_12x12.txt";
    let num_samples = 10000;
    
    let directions = ["L", "R", "T", "B", "U"];
    let mut file = File::create(output_file).expect("Failed to create output file");
    
    println!("Generating {} training samples for 12x12 planarian eyespot...", num_samples);
    
    for i in 0..num_samples {
        let direction = directions[rng.gen_range(0..directions.len())];
        let intensity = rng.gen_range(0.6..1.0);
        let noise_level = rng.gen_range(0.0..0.5);
        
        let mut grid = generate_gradient(direction, intensity);
        add_noise(&mut grid, noise_level);
        
        let grid_str: String = grid.iter().map(|&b| char::from_digit(b as u32, 10).unwrap()).collect();
        writeln!(file, "{}{}", grid_str, direction).expect("Failed to write to file");
        
        if (i + 1) % 1000 == 0 {
            print!("\rGenerated {}/{} samples", i + 1, num_samples);
            std::io::stdout().flush().unwrap();
        }
    }
    
    println!("\n\nDataset saved to: {}", output_file);
    println!("\nSample format:");
    println!("  144 digits (0-9) representing 12x12 grid + direction label (L/R/T/B/U)");
    
    let sample_file = std::fs::read_to_string(output_file).unwrap();
    println!("\nDistribution:");
    let mut counts = std::collections::HashMap::new();
    for line in sample_file.lines() {
        if let Some(label) = line.chars().last() {
            *counts.entry(label).or_insert(0) += 1;
        }
    }
    for (label, count) in counts.iter() {
        println!("  {}: {} samples ({:.1}%)", label, count, (*count as f64 / num_samples as f64) * 100.0);
    }
}
