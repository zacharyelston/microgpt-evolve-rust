/*
    GPU Detection Utility
    Lists all available CUDA devices on the system
*/

#[cfg(feature = "gpu")]
use cudarc::driver::CudaDevice;

fn main() {
    #[cfg(feature = "gpu")]
    {
        println!("🔍 Scanning for CUDA devices...");
        
        // Try to detect multiple GPUs
        for device_id in 0..4 {
            match CudaDevice::new(device_id) {
                Ok(device) => {
                    let name = device.name().unwrap_or("Unknown".to_string());
                    println!("✅ GPU {}: {}", device_id, name);
                }
                Err(_) => {
                    if device_id == 0 {
                        println!("❌ No CUDA devices found");
                        break;
                    }
                    // Don't print errors for higher device IDs that don't exist
                }
            }
        }
        
        println!("\n💡 Usage:");
        println!("  cargo run --release --features gpu --bin gpu_detect");
        println!("  cargo run --release --features gpu --bin evolve_large_gpu -- 100");
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        println!("⚠️  GPU feature not enabled");
        println!("Build with: cargo build --release --features gpu");
    }
}
