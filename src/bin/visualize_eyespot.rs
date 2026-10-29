use std::env;

fn visualize_grid(line: &str) {
    if line.len() < 65 {
        println!("Invalid line length: {}", line.len());
        return;
    }
    
    let grid_str = &line[0..64];
    let label = &line[64..65];
    
    println!("\nDirection: {} ({})", label, match label {
        "L" => "Light from LEFT",
        "R" => "Light from RIGHT",
        "T" => "Light from TOP",
        "B" => "Light from BOTTOM",
        "U" => "UNIFORM lighting",
        _ => "Unknown",
    });
    
    println!("┌────────────────┐");
    for y in 0..8 {
        print!("│");
        for x in 0..8 {
            let idx = y * 8 + x;
            let brightness = grid_str.chars().nth(idx).unwrap().to_digit(10).unwrap();
            let symbol = match brightness {
                0..=1 => " ",
                2..=3 => "░",
                4..=5 => "▒",
                6..=7 => "▓",
                _ => "█",
            };
            print!("{}", symbol);
            print!("{}", symbol);
        }
        println!("│");
    }
    println!("└────────────────┘");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = if args.len() > 1 {
        &args[1]
    } else {
        "eyespot_training.txt"
    };
    
    let contents = std::fs::read_to_string(filename)
        .expect("Failed to read training file");
    
    let lines: Vec<&str> = contents.lines().collect();
    
    println!("=== Planarian Eyespot Dataset Visualizer ===");
    println!("Total samples: {}\n", lines.len());
    
    let samples_to_show = if args.len() > 2 {
        args[2].parse().unwrap_or(5)
    } else {
        5
    };
    
    println!("Showing {} random samples:\n", samples_to_show);
    
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut sample_lines = lines.clone();
    sample_lines.shuffle(&mut rng);
    
    for line in sample_lines.iter().take(samples_to_show) {
        visualize_grid(line);
    }
    
    println!("\n=== Label Distribution ===");
    let mut counts = std::collections::HashMap::new();
    for line in &lines {
        if let Some(label) = line.chars().nth(64) {
            *counts.entry(label).or_insert(0) += 1;
        }
    }
    
    for (label, count) in [('L', "LEFT"), ('R', "RIGHT"), ('T', "TOP"), ('B', "BOTTOM"), ('U', "UNIFORM")] {
        let c = counts.get(&label).unwrap_or(&0);
        let pct = (*c as f64 / lines.len() as f64) * 100.0;
        println!("  {}: {:5} samples ({:5.2}%)", count, c, pct);
    }
}
