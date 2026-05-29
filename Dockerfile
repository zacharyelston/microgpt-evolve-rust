# Stage 1: Build the Rust binary with CUDA support
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --features gpu --bin evolve_genesis_config \
    && cargo build --release --features gpu --bin evolve_gpu_matrix \
    && cargo build --release --features gpu --bin gpu_detect \
    && cargo build --release --bin evolve_genesis_config

# Stage 2: Runtime image
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip libssl3 \
    && pip3 install --no-cache-dir numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/evolve_genesis_config ./evolve_genesis_config_cpu
COPY --from=builder /build/target/release/evolve_genesis_config ./evolve_genesis_config_gpu
COPY --from=builder /build/target/release/evolve_gpu_matrix ./evolve_gpu_matrix
COPY --from=builder /build/target/release/gpu_detect ./gpu_detect
COPY analyze_results.py ./
COPY analyze_efficiency.py ./
COPY evolution_config.json ./
COPY evolution_config_100.json ./
COPY genomes ./genomes
COPY scripts ./scripts

ENV RUST_LOG=info

ENTRYPOINT ["/bin/bash"]
