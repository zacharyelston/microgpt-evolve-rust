//! Modular 3x3 Vision Building Blocks
//! 
//! This module provides reusable 3x3 vision processing blocks that can be
//! trained independently and then assembled into larger grids.

pub mod block_3x3;
pub mod block_3x3_improved;
pub mod assembly;
pub mod genome;

pub use block_3x3::VisionBlock3x3;
pub use block_3x3::BlockGenome;
pub use block_3x3_improved::VisionBlock3x3Improved;
pub use block_3x3_improved::ImprovedBlockGenome;
pub use assembly::BlockAssembly;
pub use assembly::LayoutConfig;
pub use genome::GenomeCollection;
pub use genome::GenomeManager;
pub use genome::GenomeMetrics;
pub use genome::TrainingConfig;
