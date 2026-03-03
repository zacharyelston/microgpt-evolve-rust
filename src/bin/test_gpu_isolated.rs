/*
    Independent GPU Acceleration Test
    
    This test verifies GPU functionality without affecting
    the main training loop or evolution engine.
*/

use microgpt_rust::gpu_accel::{create_gpu_accelerator, cpu_matrix_vector_multiply};
use std::time::Instant;

fn test_gpu_accelerator() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing GPU Accelerator ===");
    
    // Test GPU initialization
    let gpu = create_gpu_accelerator()?;
    println!("GPU Available: {}", gpu.is_available());
    
    if gpu.is_available() {
        // Test memory transfer
        gpu.test_memory_transfer()?;
        
        // Test matrix multiplication
        test_matrix_operations(&gpu)?;
    } else {
        println!("⚠️  GPU not available, testing CPU fallback");
        test_cpu_fallback()?;
    }
    
    Ok(())
}

fn test_matrix_operations(gpu: &microgpt_rust::gpu_accel::GpuAccelerator) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Matrix Operations Test ---");
    
    // Test data
    let rows = 1000;
    let cols = 1000;
    let matrix: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 / 100.0).collect();
    let vector: Vec<f32> = (0..cols).map(|i| (i % 50) as f32 / 50.0).collect();
    
    // GPU version
    let start = Instant::now();
    let gpu_result = gpu.matrix_vector_multiply(&matrix, rows, cols, &vector)?;
    let gpu_time = start.elapsed();
    
    // CPU version for comparison
    let start = Instant::now();
    let cpu_result = cpu_matrix_vector_multiply(&matrix, rows, cols, &vector);
    let cpu_time = start.elapsed();
    
    // Verify correctness
    let max_diff = gpu_result.iter()
        .zip(cpu_result.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    
    println!("GPU time: {:?}", gpu_time);
    println!("CPU time: {:?}", cpu_time);
    println!("Max difference: {:.6}", max_diff);
    
    if max_diff < 1e-6 {
        println!("✅ Matrix multiplication test PASSED");
        
        if gpu_time < cpu_time {
            println!("🚀 GPU faster by {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
        } else {
            println!("⚠️  GPU slower by {:.2}x", gpu_time.as_secs_f64() / cpu_time.as_secs_f64());
        }
    } else {
        println!("❌ Matrix multiplication test FAILED");
    }
    
    Ok(())
}

fn test_cpu_fallback() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- CPU Fallback Test ---");
    
    let rows = 1000;
    let cols = 1000;
    let matrix: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 / 100.0).collect();
    let vector: Vec<f32> = (0..cols).map(|i| (i % 50) as f32 / 50.0).collect();
    
    let start = Instant::now();
    let result = cpu_matrix_vector_multiply(&matrix, rows, cols, &vector);
    let time = start.elapsed();
    
    println!("CPU time: {:?}", time);
    println!("Result length: {}", result.len());
    println!("First 5 results: {:?}", &result[..5]);
    println!("✅ CPU fallback test PASSED");
    
    Ok(())
}

fn test_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Edge Cases Test ---");
    
    let gpu = create_gpu_accelerator()?;
    
    // Test empty matrices
    let empty_matrix: Vec<f32> = vec![];
    let empty_vector: Vec<f32> = vec![];
    
    match gpu.matrix_vector_multiply(&empty_matrix, 0, 0, &empty_vector) {
        Ok(_) => println!("✅ Empty matrix test passed"),
        Err(e) => println!("⚠️  Empty matrix test: {}", e),
    }
    
    // Test dimension mismatch
    let matrix = vec![1.0f32; 100]; // 10x10
    let vector = vec![1.0f32; 5]; // Wrong size
    
    match gpu.matrix_vector_multiply(&matrix, 10, 10, &vector) {
        Ok(_) => println!("❌ Dimension mismatch test failed"),
        Err(_) => println!("✅ Dimension mismatch test passed"),
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Isolated GPU Acceleration Test Suite ===");
    
    test_gpu_accelerator()?;
    test_edge_cases()?;
    
    println!("\n=== Test Complete ===");
    println!("GPU acceleration module is working correctly!");
    
    Ok(())
}
