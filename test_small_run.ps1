# PowerShell script for testing small evolution runs
# Tests the configurable evolution system with a small number of generations

Write-Host "🧪 Testing Small Evolution Run" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# Check if Rust is installed
try {
    $null = cargo --version
    Write-Host "✅ Rust/Cargo found" -ForegroundColor Green
} catch {
    Write-Host "❌ Cargo not found. Please install Rust first." -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "Cargo.toml")) {
    Write-Host "❌ Cargo.toml not found. Please run from microgpt-evolve-rust directory." -ForegroundColor Red
    exit 1
}

Write-Host "✅ In correct directory" -ForegroundColor Green

# Create a small test config (5 generations)
Write-Host ""
Write-Host "📝 Creating test configuration..." -ForegroundColor Blue

$testConfig = @{
    parameter_ranges = @{
        n_emb = @{ min = 1; max = 3; values = @(1, 2, 3) }
        n_ctx = @{ min = 1; max = 3; values = @(1, 2, 3) }
        n_layer = @{ min = 1; max = 3 }
        n_head = @{ min = 1; max = 2; values = @(1, 2) }
        n_ff_exp = @{ min = 1; max = 3 }
        lr = @{ min = 0.001; max = 0.1; log_scale = $true }
        steps = @{ min = 200; max = 600; values = @(200, 400, 600) }
    }
    evolution_settings = @{
        population_size = 6
        generations = 5
        mutation_rate = 0.3
        target_loss = 1.5
    }
    genesis_config = @{
        n_emb = 1
        n_ctx = 1
        n_layer = 1
        n_head = 1
        n_ff_exp = 1
        lr = 0.01
        steps = 400
    }
}

$testConfig | ConvertTo-Json -Depth 10 | Out-File -FilePath "evolution_config_test.json" -Encoding utf8
Write-Host "✅ Test config created: evolution_config_test.json" -ForegroundColor Green

# Build the evolution binary
Write-Host ""
Write-Host "🔨 Building evolution binary..." -ForegroundColor Blue

$buildResult = cargo build --release --bin evolve_loss_config
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build successful" -ForegroundColor Green

# Create results directory
New-Item -ItemType Directory -Force -Path "results" | Out-Null

# Run the test evolution
Write-Host ""
Write-Host "🚀 Running test evolution (5 generations)..." -ForegroundColor Blue
Write-Host "This should take 2-5 minutes" -ForegroundColor Gray

$startTime = Get-Date
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$env:RUST_LOG = "info"
$testResult = cargo run --release --bin evolve_loss_config -- evolution_config_test.json 2>&1 | Tee-Object -FilePath "results/test_evolution_${timestamp}.log"

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "🎉 Test Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host "Duration: $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor White
Write-Host "Log file: results/test_evolution_${timestamp}.log" -ForegroundColor White

# Check if genome was created
if (Test-Path "genome.json") {
    Write-Host ""
    Write-Host "🏆 Best Result:" -ForegroundColor Yellow
    $genome = Get-Content "genome.json" | ConvertFrom-Json
    Write-Host "Generation: $($genome.generation)" -ForegroundColor White
    Write-Host "Loss: $($genome.loss)" -ForegroundColor White
    Write-Host "Config: Emb:$($genome.config.n_emb) Head:$($genome.config.n_head) Lay:$($genome.config.n_layer) Ctx:$($genome.config.n_ctx) FF:$($genome.config.n_ff_exp) LR:$($genome.config.lr) Steps:$($genome.config.steps)" -ForegroundColor Gray
} else {
    Write-Host "⚠️  No genome.json file created" -ForegroundColor Yellow
}

# Show final log entries
Write-Host ""
Write-Host "📋 Final log entries:" -ForegroundColor Blue
Get-Content "results/test_evolution_${timestamp}.log" | Select-Object -Last 10

Write-Host ""
Write-Host "✅ Test successful! Ready for large runs." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run 100 generations: ./run_evolution_100.sh" -ForegroundColor White
Write-Host "  2. Run 1000 generations: ./run_evolution_1000.sh" -ForegroundColor White
Write-Host "  3. Monitor with: ./checkpoint_1000.sh" -ForegroundColor White
