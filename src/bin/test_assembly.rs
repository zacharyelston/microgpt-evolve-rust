//! Assembly Testing
//! 
//! Tests assemblies of 3x3 blocks in different configurations.

use microgpt_rust::blocks::{BlockAssembly, LayoutConfig, GenomeCollection, GenomeManager};
use candle_core::{Device, Result, Tensor, Module};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    let collection_id = if args.len() > 1 { &args[1] } else { "eyespot_3x3_v1" };
    let layout_name = if args.len() > 2 { &args[2] } else { "single" };
    
    println!("=== Assembly Testing ===");
    println!("Collection: {}", collection_id);
    println!("Layout: {}", layout_name);
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Load genome collection
    let mut manager = GenomeManager::new("genomes");
    manager.load_collections()?;
    
    let collection = match manager.get_collection(collection_id) {
        Some(c) => c,
        None => {
            eprintln!("Error: Collection '{}' not found", collection_id);
            eprintln!("Available collections: {:?}", manager.list_collections());
            std::process::exit(1);
        }
    };
    
    println!("Loaded collection with {} genomes", collection.genomes.len());
    
    // Parse layout
    let layout = match layout_name {
        "single" => LayoutConfig::Single,
        "2x2" => LayoutConfig::Grid2x2,
        "3x3" => LayoutConfig::Grid3x3,
        "4x4" => LayoutConfig::Grid4x4,
        _ => {
            eprintln!("Error: Unknown layout '{}'", layout_name);
            eprintln!("Available: single, 2x2, 3x3, 4x4");
            std::process::exit(1);
        }
    };
    
    let (rows, cols) = layout.dimensions();
    let required_blocks = rows * cols;
    
    println!("Layout: {}x{} ({} blocks needed)", rows, cols, required_blocks);
    
    if collection.genomes.len() < required_blocks {
        eprintln!("Error: Need {} blocks but only {} available", required_blocks, collection.genomes.len());
        std::process::exit(1);
    }
    
    // Create assembly
    println!("\n--- Creating Assembly ---");
    
    let assembly_id = format!("{}_{}", collection_id, layout_name);
    let assembly = BlockAssembly::new(
        &assembly_id,
        (rows, cols),
        24, // embedding_dim
        48, // evaluator_dim
        &device
    )?;
    
    println!("Created assembly: {}", assembly_id);
    println!("Dimensions: {}x{}", assembly.dimensions().0, assembly.dimensions().1);
    println!("Total blocks: {}", assembly.block_count());
    
    // Load test data
    let data_path = "eyespot_training_3x3.txt";
    let raw = std::fs::read_to_string(data_path).expect("Failed to read test data");
    let lines: Vec<&str> = raw.lines().collect();
    
    // Test assembly with sample data
    println!("\n--- Testing Assembly ---");
    
    let mut correct = 0;
    let mut total = 0;
    let test_samples = 100.min(lines.len() / 10);
    
    for line in lines.iter().take(test_samples) {
        if line.len() < 10 {
            continue;
        }
        
        // For single block test, use the actual 3x3 grid
        // For larger assemblies, we'd need to tile or combine grids
        let grid_data = if required_blocks == 1 {
            let grid_str = &line[0..9];
            let mut grid_pixels = vec![0u8; 9];
            for (i, ch) in grid_str.chars().enumerate() {
                grid_pixels[i] = ch.to_digit(10).unwrap_or(0) as u8;
            }
            vec![grid_pixels.as_slice()]
        } else {
            // For multi-block assemblies, duplicate the single grid
            // In a real scenario, you'd have larger grids or tile patterns
            let grid_str = &line[0..9];
            let mut grid_pixels = vec![0u8; 9];
            for (i, ch) in grid_str.chars().enumerate() {
                grid_pixels[i] = ch.to_digit(10).unwrap_or(0) as u8;
            }
            
            let mut grid_data = Vec::new();
            for _ in 0..required_blocks {
                grid_data.push(grid_pixels.as_slice());
            }
            grid_data
        };
        
        let label_str = &line[9..];
        let target_label = match label_str {
            "L" => 0, "R" => 1, "T" => 2, "B" => 3,
            "TL" => 4, "TR" => 5, "BL" => 6, "BR" => 7,
            "U" => 8, _ => 8,
        };
        
        // Forward pass through assembly
        let logits = assembly.forward(&grid_data)?;
        
        // Get prediction
        let pred_idx = logits.to_vec1::<f32>()?
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        if pred_idx == target_label {
            correct += 1;
        }
        total += 1;
        
        if total <= 10 {
            let pred_label = match pred_idx {
                0 => "L", 1 => "R", 2 => "T", 3 => "B",
                4 => "TL", 5 => "TR", 6 => "BL", 7 => "BR",
                _ => "U",
            };
            println!("Sample {}: {} -> {} {}", total, label_str, pred_label, 
                if pred_idx == target_label { "✓" } else { "✗" });
        }
    }
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    
    println!("\n--- Assembly Test Results ---");
    println!("Test accuracy: {}/{} ({:.2}%)", correct, total, accuracy);
    println!("Assembly size: {}x{} blocks", rows, cols);
    println!("Effective grid size: {}x{} pixels", rows * 3, cols * 3);
    
    // Performance estimation
    println!("\n--- Performance Estimates ---");
    let single_block_time = 0.5; // seconds per inference (estimated)
    let estimated_time = single_block_time * required_blocks as f64;
    println!("Estimated inference time: {:.2}ms", estimated_time * 1000.0);
    
    println!("\n=== Assembly Testing Complete ===");
    
    if accuracy > 80.0 {
        println!("✅ Assembly performs well ({:.1}% accuracy)", accuracy);
        println!("✅ Ready for production use");
    } else {
        println!("⚠️  Assembly performance needs improvement ({:.1}% accuracy)", accuracy);
        println!("💡 Consider: more training, better block selection, or assembly tuning");
    }
    
    Ok(())
}
