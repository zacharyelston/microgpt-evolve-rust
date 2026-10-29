use microgpt_rust::bundle::Bundle;

fn main() {
    println!("=== Bundle Loader Test ===\n");
    
    let bundle_path = "research/bundles/bundle-1-planarian-eye/planarian_eyespot-prd.yaml";
    
    println!("Loading bundle from: {}", bundle_path);
    
    match Bundle::from_yaml_file(bundle_path) {
        Ok(bundle) => {
            println!("✓ Bundle loaded successfully!\n");
            
            println!("Metadata:");
            println!("  Name: {}", bundle.metadata.name);
            println!("  Version: {}", bundle.metadata.version);
            println!("  Category: {}", bundle.metadata.category);
            println!("  Author: {}", bundle.metadata.author);
            if let Some(desc) = &bundle.metadata.description {
                println!("  Description: {}", desc);
            }
            
            println!("\nArchitecture:");
            println!("  Evaluator Core:");
            println!("    n_emb: {}", bundle.architecture.evaluator_core.n_emb);
            println!("    n_ctx: {}", bundle.architecture.evaluator_core.n_ctx);
            println!("    n_layer: {}", bundle.architecture.evaluator_core.n_layer);
            println!("    n_head: {}", bundle.architecture.evaluator_core.n_head);
            println!("    n_ff_exp: {}", bundle.architecture.evaluator_core.n_ff_exp);
            
            println!("\n  Plasticity:");
            println!("    L1 prune penalty: {}", bundle.architecture.plasticity.l1_prune_penalty);
            println!("    Prune threshold: {}", bundle.architecture.plasticity.prune_threshold);
            println!("    Max surviving synapses: {}", bundle.architecture.plasticity.max_surviving_synapses);
            
            println!("\nInterface:");
            println!("  Input type: {}", bundle.interface.input_type);
            println!("  Output type: {}", bundle.interface.output_type);
            println!("  Output dim: {}", bundle.interface.output_dim);
            
            println!("\nUpstream Nodes:");
            match &bundle.upstream_nodes {
                microgpt_rust::bundle::UpstreamNodes::StaticAllocation { nodes } => {
                    println!("  Strategy: Static Allocation");
                    println!("  Node count: {}", nodes.len());
                    for (name, node) in nodes {
                        println!("\n  Node: {}", name);
                        println!("    Path: {}", node.path);
                        println!("    Context: {}", node.n_ctx);
                        println!("    Slice: {:?}", node.slice);
                    }
                }
                microgpt_rust::bundle::UpstreamNodes::DynamicQuery { queries } => {
                    println!("  Strategy: Dynamic Query");
                    println!("  Query count: {}", queries.len());
                    for (i, query) in queries.iter().enumerate() {
                        println!("\n  Query {}:", i + 1);
                        println!("    Path: {}", query.path);
                        println!("    Select: {}", query.select);
                        println!("    K: {}", query.k);
                        println!("    Metric: {}", query.metric);
                    }
                }
            }
            
            println!("\n--- Validation ---");
            match bundle.validate() {
                Ok(_) => println!("✓ Bundle validation passed!"),
                Err(e) => println!("✗ Bundle validation failed: {}", e),
            }
            
            println!("\n--- Architecture Summary ---");
            let params_per_quadrant = estimate_params(
                bundle.architecture.evaluator_core.n_emb,
                16,
                1,
                bundle.architecture.evaluator_core.n_head,
                bundle.architecture.evaluator_core.n_ff_exp,
            );
            let evaluator_params = estimate_params(
                bundle.architecture.evaluator_core.n_emb,
                bundle.architecture.evaluator_core.n_ctx,
                bundle.architecture.evaluator_core.n_layer,
                bundle.architecture.evaluator_core.n_head,
                bundle.architecture.evaluator_core.n_ff_exp,
            );
            
            println!("Estimated parameters:");
            println!("  Per quadrant node: ~{} params", params_per_quadrant);
            println!("  Evaluator core: ~{} params", evaluator_params);
            println!("  Total assembly: ~{} params", 4 * params_per_quadrant + evaluator_params);
            
        }
        Err(e) => {
            eprintln!("✗ Failed to load bundle: {}", e);
            std::process::exit(1);
        }
    }
}

fn estimate_params(n_emb: usize, n_ctx: usize, n_layer: usize, _n_head: usize, n_ff_exp: usize) -> usize {
    let vocab = 11;
    let wte = vocab * n_emb;
    let wpe = n_ctx * n_emb;
    let lm_head = vocab * n_emb;
    
    let per_layer = 4 * (n_emb * n_emb) + 2 * (n_emb * n_emb * n_ff_exp);
    
    wte + wpe + lm_head + (n_layer * per_layer)
}
