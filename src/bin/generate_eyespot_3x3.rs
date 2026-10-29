use rand::Rng;
use std::fs::File;
use std::io::Write;

// Generate 3x3 gradient with all 8 directions + uniform
fn generate_gradient(direction: &str, intensity: f64) -> Vec<u8> {
    let mut grid = vec![vec![0u8; 3]; 3];
    
    for y in 0..3 {
        for x in 0..3 {
            let brightness = match direction {
                // Cardinal directions
                "L" => {
                    let dist = x as f64 / 2.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                "R" => {
                    let dist = x as f64 / 2.0;
                    (dist * intensity * 9.0) as u8
                }
                "T" => {
                    let dist = y as f64 / 2.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                "B" => {
                    let dist = y as f64 / 2.0;
                    (dist * intensity * 9.0) as u8
                }
                // Diagonal directions
                "TL" => {
                    let x_dist = x as f64 / 2.0;
                    let y_dist = y as f64 / 2.0;
                    let dist = (x_dist + y_dist) / 2.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                "TR" => {
                    let x_dist = x as f64 / 2.0;
                    let y_dist = y as f64 / 2.0;
                    let dist = ((1.0 - x_dist) + y_dist) / 2.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                "BL" => {
                    let x_dist = x as f64 / 2.0;
                    let y_dist = y as f64 / 2.0;
                    let dist = (x_dist + (1.0 - y_dist)) / 2.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                "BR" => {
                    let x_dist = x as f64 / 2.0;
                    let y_dist = y as f64 / 2.0;
                    let dist = ((1.0 - x_dist) + (1.0 - y_dist)) / 2.0;
                    ((1.0 - dist) * intensity * 9.0) as u8
                }
                // Uniform
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

// Flip grid horizontally
fn flip_horizontal(grid: &[u8]) -> Vec<u8> {
    let mut flipped = vec![0u8; 9];
    for y in 0..3 {
        for x in 0..3 {
            let old_idx = y * 3 + x;
            let new_x = 2 - x;
            let new_idx = y * 3 + new_x;
            flipped[new_idx] = grid[old_idx];
        }
    }
    flipped
}

// Flip grid vertically
fn flip_vertical(grid: &[u8]) -> Vec<u8> {
    let mut flipped = vec![0u8; 9];
    for y in 0..3 {
        for x in 0..3 {
            let old_idx = y * 3 + x;
            let new_y = 2 - y;
            let new_idx = new_y * 3 + x;
            flipped[new_idx] = grid[old_idx];
        }
    }
    flipped
}

// Rotate grid 90 degrees clockwise
fn rotate_90(grid: &[u8]) -> Vec<u8> {
    let mut rotated = vec![0u8; 9];
    for y in 0..3 {
        for x in 0..3 {
            let old_idx = y * 3 + x;
            let new_y = x;
            let new_x = 2 - y;
            let new_idx = new_y * 3 + new_x;
            rotated[new_idx] = grid[old_idx];
        }
    }
    rotated
}

// Add noise to simulate real-world conditions
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
    let output_file = "eyespot_training_3x3.txt";
    let num_samples = 15000; // Focused dataset for building block
    
    // All 9 directions: 4 cardinal + 4 diagonal + uniform
    let directions = ["L", "R", "T", "B", "TL", "TR", "BL", "BR", "U"];
    let mut file = File::create(output_file).expect("Failed to create output file");
    
    println!("Generating {} training samples for 3x3 planarian eyespot...", num_samples);
    println!("This is the fundamental building block for larger assemblies");
    println!("Directions: {:?}", directions);
    println!("Including all orientations and intensity variations...");
    
    for i in 0..num_samples {
        let direction = directions[rng.gen_range(0..directions.len())];
        let intensity = rng.gen_range(0.3..1.0); // Wide intensity range
        let noise_level = rng.gen_range(0.0..0.5); // Moderate noise
        
        let mut grid = generate_gradient(direction, intensity);
        add_noise(&mut grid, noise_level);
        
        // Apply random transformations for orientation invariance
        let transform = rng.gen_range(0..4);
        let final_grid = match transform {
            1 => flip_horizontal(&grid),
            2 => flip_vertical(&grid),
            3 => rotate_90(&grid),
            _ => grid,
        };
        
        let grid_str: String = final_grid.iter().map(|&b| char::from_digit(b as u32, 10).unwrap()).collect();
        writeln!(file, "{}{}", grid_str, direction).expect("Failed to write to file");
        
        if (i + 1) % 1500 == 0 {
            print!("\rGenerated {}/{} samples", i + 1, num_samples);
            std::io::stdout().flush().unwrap();
        }
    }
    
    println!("\n\nDataset saved to: {}", output_file);
    println!("\nSample format:");
    println!("  9 digits (0-9) representing 3x3 grid + direction label");
    println!("  Labels: L,R,T,B (cardinal), TL,TR,BL,BR (diagonal), U (uniform)");
    
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
    
    // Show some example grids for verification
    println!("\nExample grids (ASCII visualization):");
    let mut examples = Vec::new();
    for direction in directions.iter() {
        let grid = generate_gradient(direction, 0.8);
        examples.push((direction, grid));
    }
    
    for (direction, grid) in examples {
        println!("\nDirection: {} ({})", direction, match *direction {
            "L" => "Light from LEFT",
            "R" => "Light from RIGHT", 
            "T" => "Light from TOP",
            "B" => "Light from BOTTOM",
            "TL" => "Light from TOP-LEFT",
            "TR" => "Light from TOP-RIGHT",
            "BL" => "Light from BOTTOM-LEFT",
            "BR" => "Light from BOTTOM-RIGHT",
            "U" => "UNIFORM lighting",
            _ => "Unknown",
        });
        println!("┌───┐");
        for y in 0..3 {
            print!("│");
            for x in 0..3 {
                let idx = y * 3 + x;
                let brightness = grid[idx];
                let symbol = match brightness {
                    0..=1 => " ",
                    2..=3 => "░",
                    4..=5 => "▒",
                    6..=7 => "▓",
                    _ => "█",
                };
                print!("{}", symbol);
            }
            println!("│");
        }
        println!("└───┘");
    }
    
    println!("\n=== 3x3 Building Block Strategy ===");
    println!("✅ Prove 3x3 works perfectly with GPU acceleration");
    println!("✅ Extract and reuse the proven genome");
    println!("✅ Assemble multiple 3x3 blocks for larger grids");
    println!("✅ No need to test higher resolutions for fitness");
}
