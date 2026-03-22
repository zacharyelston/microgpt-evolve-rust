// Simple GPU initialization proof of concept
use cudarc::driver::CudaDevice;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Initialization Test ===");
    
    // Test 1: Basic GPU initialization
    let start = Instant::now();
    let gpu = CudaDevice::new(0)?;
    let init_time = start.elapsed();
    
    println!("✅ GPU initialized in {:?}", init_time);
    println!("GPU Name: {}", gpu.name()?);
    println!("GPU is ready for memory operations!");
    
    // Test 2: Simple memory allocation
    let start = Instant::now();
    let size = 1024 * 1024; // 1M floats = 4MB
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let gpu_data = gpu.htod_sync_copy(&data)?;
    let alloc_time = start.elapsed();
    
    println!("✅ Allocated {} floats in {:?}", size, alloc_time);
    
    // Test 3: Simple memory copy back
    let start = Instant::now();
    let result = gpu.dtoh_sync_copy(&gpu_data)?;
    let copy_time = start.elapsed();
    
    println!("✅ Copied back to CPU in {:?}", copy_time);
    
    // Verify correctness
    let max_diff = data.iter()
        .zip(result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("Max difference: {:.6}", max_diff);
    
    if max_diff < 1e-6 {
        println!("✅ Memory transfer test PASSED!");
    } else {
        println!("❌ Memory transfer test FAILED!");
    }
    
    // Test 4: Performance comparison
    println!("\n--- Performance Test ---");
    let size = 4096 * 4096; // 16M floats = 64MB
    
    // CPU allocation
    let start = Instant::now();
    let cpu_data: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
    let cpu_time = start.elapsed();
    
    // GPU allocation and transfer
    let start = Instant::now();
    let gpu_data = gpu.htod_sync_copy(&cpu_data)?;
    let gpu_time = start.elapsed();
    
    println!("CPU allocation ({} floats): {:?}", size, cpu_time);
    println!("GPU transfer ({} floats): {:?}", size, gpu_time);
    
    if gpu_time < cpu_time {
        println!("✅ GPU transfer faster by {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    } else {
        println!("⚠️  GPU transfer slower by {:.2}x", gpu_time.as_secs_f64() / cpu_time.as_secs_f64());
    }
    
    println!("\n=== GPU Test Complete ===");
    println!("GPU is working and ready for matrix operations!");
    
    Ok(())
}
