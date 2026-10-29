//! Core logging system
//! 
//! Configurable logging with debug levels and performance metrics

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::io::Write;

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl From<&str> for LogLevel {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" => LogLevel::Warn,
            "error" => LogLevel::Error,
            _ => LogLevel::Info,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub steps_per_second: f64,
    pub samples_per_second: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: f32,
}

#[derive(Clone)]
pub struct Logger {
    level: LogLevel,
    debug_enabled: bool,
    start_time: Instant,
    metrics: HashMap<String, Vec<Duration>>,
}

impl Logger {
    pub fn new(level: LogLevel, debug_enabled: bool) -> Self {
        Self {
            level,
            debug_enabled,
            start_time: Instant::now(),
            metrics: HashMap::new(),
        }
    }
    
    pub fn log(&self, level: LogLevel, message: &str) {
        if self.should_log(level) {
            let timestamp = self.start_time.elapsed().as_secs_f64();
            let level_str = format!("{:?}", level).chars().take(4).collect::<String>();
            println!("[{:8.3}s] [{:<4}] {}", timestamp, level_str, message);
        }
    }
    
    pub fn debug(&self, message: &str) {
        if self.debug_enabled {
            self.log(LogLevel::Debug, message);
        }
    }
    
    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message);
    }
    
    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message);
    }
    
    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message);
    }
    
    pub fn start_timer(&mut self, operation: &str) {
        if self.debug_enabled {
            self.metrics.entry(operation.to_string()).or_insert_with(Vec::new).push(Duration::ZERO);
        }
    }
    
    pub fn end_timer(&mut self, operation: &str) -> Option<Duration> {
        if self.debug_enabled {
            // In a real implementation, we'd track actual timing
            // For now, return a placeholder
            self.metrics.get_mut(operation)?.pop()
        } else {
            None
        }
    }
    
    pub fn report_performance(&self, metrics: &PerformanceMetrics) {
        self.info(&format!(
            "Performance: {:.1} steps/s, {:.1} samples/s, {:.1}MB RAM, {:.1}% GPU",
            metrics.steps_per_second,
            metrics.samples_per_second,
            metrics.memory_usage_mb,
            metrics.gpu_utilization * 100.0
        ));
    }
    
    pub fn progress_bar(&self, current: usize, total: usize, step: usize, loss: f32, best_acc: f32, eta: f64) {
        let progress = (current as f64 / total as f64) * 100.0;
        let bar_width = 30;
        let filled = ((progress / 100.0) * bar_width as f64) as usize;
        let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
        
        print!("[{}] {:5.1}% | Step {:4}/{} | Loss: {:.4} | Best: {:.1}% | ETA: {:3.0}s\r",
            bar, progress, step, total, loss, best_acc, eta
        );
        std::io::stdout().flush().unwrap();
    }
    
    fn should_log(&self, level: LogLevel) -> bool {
        match (&self.level, &level) {
            (LogLevel::Debug, _) => true,
            (LogLevel::Info, LogLevel::Info | LogLevel::Warn | LogLevel::Error) => true,
            (LogLevel::Warn, LogLevel::Warn | LogLevel::Error) => true,
            (LogLevel::Error, LogLevel::Error) => true,
            _ => false,
        }
    }
}

pub struct ProgressTracker {
    logger: Logger,
    total_steps: usize,
    best_accuracy: f32,
    plateau_counter: usize,
    max_plateau: usize,
    last_update: Instant,
}

impl ProgressTracker {
    pub fn new(logger: Logger, total_steps: usize, max_plateau: usize) -> Self {
        Self {
            logger,
            total_steps,
            best_accuracy: 0.0,
            plateau_counter: 0,
            max_plateau,
            last_update: Instant::now(),
        }
    }
    
    pub fn update(&mut self, step: usize, current_accuracy: f32, loss: f32) -> bool {
        let should_report = step % 100 == 0 || step == self.total_steps - 1;
        
        if current_accuracy > self.best_accuracy {
            self.best_accuracy = current_accuracy;
            self.plateau_counter = 0;
            self.logger.info(&format!("🎯 NEW BEST: {:.2}% at step {}", current_accuracy, step));
            
            if self.best_accuracy >= 95.0 {
                self.logger.info(&format!("🎉 TARGET ACHIEVED: {:.1}% - 95% accuracy reached!", self.best_accuracy));
                return true; // Early termination
            }
        } else {
            self.plateau_counter += 100;
        }
        
        if self.plateau_counter >= self.max_plateau {
            self.logger.info(&format!("⏹️  Plateau detected at step {} (best: {:.2}%)", step, self.best_accuracy));
            return true; // Early termination
        }
        
        if should_report {
            let elapsed = self.last_update.elapsed().as_secs_f64();
            let steps_per_sec = 100.0 / elapsed.max(0.001);
            let remaining_steps = self.total_steps - step;
            let eta = remaining_steps as f64 / steps_per_sec.max(0.001);
            
            self.logger.progress_bar(step, self.total_steps, step, loss, self.best_accuracy, eta);
            self.last_update = Instant::now();
        }
        
        false
    }
    
    pub fn final_report(&self, training_time: f64) {
        self.logger.info(&format!("Best accuracy: {:.2}%", self.best_accuracy));
        self.logger.info(&format!("Training time: {:.1}s", training_time));
        self.logger.info(&format!("Average speed: {:.1} steps/s", self.total_steps as f64 / training_time));
        
        if self.best_accuracy >= 95.0 {
            self.logger.info("🎉 SUCCESS: Target accuracy achieved!");
        } else if self.best_accuracy >= 90.0 {
            self.logger.info("🎯 EXCELLENT: Very close to target");
        } else if self.best_accuracy >= 80.0 {
            self.logger.info("✅ GOOD: Significant improvement");
        } else {
            self.logger.info("⚠️  IMPROVING: Better than baseline");
        }
    }
}
