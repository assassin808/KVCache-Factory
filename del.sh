#!/bin/bash

# Create target directory if it doesn't exist
mkdir -p combined_results

# Loop through all bench directories
for bench_dir in results_long_bench*; do
    # Extract the bench value (0.3, 0.5, etc.)
    bench_value=${bench_dir#results_long_bench}
    
    # Loop through size directories
    for size_dir in "$bench_dir"/_*; do
        # Extract size number (256, 512, etc.)
        size=${size_dir##*_}
        
        # Define source and target paths
        source_file="$size_dir/results.csv"
        target_file="combined_results/result_${bench_value}_${size}.csv"
        
        # Copy and rename the file if it exists
        if [ -f "$source_file" ]; then
            cp "$source_file" "$target_file"
            echo "Copied: $source_file -> $target_file"
        else
            echo "Warning: $source_file not found"
        fi
    done
done

echo "All files collected in combined_results directory!"