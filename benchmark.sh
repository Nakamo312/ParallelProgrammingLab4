#!/bin/bash

# Default values
DEFAULT_SIZES="100 200 300 400 500"
DEFAULT_THREADS="1 2 4 8"
DEFAULT_BLOCKS="16 32"  # Block sizes for CUDA
DEFAULT_CSV="ParallelProgrammingLab2/results.csv"
DEFAULT_PLOT="ParallelProgrammingLab2/performance_plot.png"
DEFAULT_MODE="parallel"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --blocks)
            BLOCKS="$2"
            shift 2
            ;;
        --csv)
            CSV_FILE="ParallelProgrammingLab2/$2"
            shift 2
            ;;
        --plot)
            PLOT_FILE="ParallelProgrammingLab2/$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
SIZES=${SIZES:-$DEFAULT_SIZES}
THREADS=${THREADS:-$DEFAULT_THREADS}
BLOCKS=${BLOCKS:-$DEFAULT_BLOCKS}
CSV_FILE=${CSV_FILE:-$DEFAULT_CSV}
PLOT_FILE=${PLOT_FILE:-$DEFAULT_PLOT}
MODE=${MODE:-$DEFAULT_MODE}

# Validate mode
VALID_MODES=("serial" "parallel" "cuda" "mpi")
if [[ ! " ${VALID_MODES[@]} " =~ " ${MODE} " ]]; then
    echo "Invalid mode: ${MODE}. Valid modes are: ${VALID_MODES[@]}"
    exit 1
fi

# Setup Python environment
if [[ ! -d "ParallelProgrammingLab2/venv" ]]; then
    python3 -m venv "ParallelProgrammingLab2/venv"
fi


# Check and install Python dependencies
python -c "import numpy" 2>/dev/null || { 
    echo "Installing Python dependencies..."
    if [[ -f "ParallelProgrammingLab2/requirements.txt" ]]; then
        pip install -r "ParallelProgrammingLab2/requirements.txt"
    else
        pip install numpy matplotlib pandas seaborn
    fi
}

# Clear previous results
echo "size,param,time,gflops" > "$CSV_FILE"

# Main benchmark loop
for size in $SIZES; do
    echo "Processing size: ${size}x${size}"
    
    # Generate matrices with absolute paths
    ./matrix_mult generate "$size" "$size" "$size" "$size" \
        "ParallelProgrammingLab2/mat1.bin" "ParallelProgrammingLab2/mat2.bin" || {
        echo "Failed to generate matrices"
        exit 1
    }
    
    case "$MODE" in
        "mpi")
            # MPI mode
            echo -n "  MPI mode... "
            output=$(mpirun -np 4 ./matrix_mult multiply mpi \
                "ParallelProgrammingLab2/mat1.bin" "ParallelProgrammingLab2/mat2.bin" \
                "ParallelProgrammingLab2/result.bin" 2>&1)
            duration=$(echo "$output" | grep "Computation time:" | awk '{print $3}')
            gflops=$(echo "$output" | grep "GFLOPS:" | awk '{print $2}')
            echo "$size,4,$duration,$gflops" >> "$CSV_FILE"
            echo "OK (${duration}ms)"
            ;;
            
        "cuda")
            # CUDA mode with multiple block sizes
            for block in $BLOCKS; do
                echo -n "  CUDA block size: ${block}x${block}... "
                output=$(./matrix_mult multiply cuda \
                    "ParallelProgrammingLab2/mat1.bin" "ParallelProgrammingLab2/mat2.bin" \
                    "ParallelProgrammingLab2/result.bin" "$block" 2>&1)
                duration=$(echo "$output" | grep "Computation time:" | awk '{print $3}')
                gflops=$(echo "$output" | grep "GFLOPS:" | awk '{print $2}')
                echo "$size,$block,$duration,$gflops" >> "$CSV_FILE"
                echo "OK (${duration}ms)"
                rm -f "ParallelProgrammingLab2/result.bin"
            done
            ;;
            
        "serial")
            # Serial mode
            echo -n "  Serial mode... "
            output=$(./matrix_mult multiply serial \
                "ParallelProgrammingLab2/mat1.bin" "ParallelProgrammingLab2/mat2.bin" \
                "ParallelProgrammingLab2/result.bin" 2>&1)
            duration=$(echo "$output" | grep "Computation time:" | awk '{print $3}')
            gflops=$(echo "$output" | grep "GFLOPS:" | awk '{print $2}')
            echo "$size,1,$duration,$gflops" >> "$CSV_FILE"
            echo "OK (${duration}ms)"
            ;;
            
        "parallel")
            # Parallel mode with multiple thread counts
            for threads in $THREADS; do
                echo -n "  Threads: $threads... "
                output=$(./matrix_mult multiply parallel \
                    "ParallelProgrammingLab2/mat1.bin" "ParallelProgrammingLab2/mat2.bin" \
                    "ParallelProgrammingLab2/result.bin" "$threads" 2>&1)
                duration=$(echo "$output" | grep "Computation time:" | awk '{print $3}')
                gflops=$(echo "$output" | grep "GFLOPS:" | awk '{print $2}')
                echo "$size,$threads,$duration,$gflops" >> "$CSV_FILE"
                echo "OK (${duration}ms)"
                rm -f "ParallelProgrammingLab2/result.bin"
            done
            ;;
    esac
    
    # Clean up
    rm -f "ParallelProgrammingLab2/mat1.bin" "ParallelProgrammingLab2/mat2.bin" "ParallelProgrammingLab2/result.bin"
done

# Generate plot
python "ParallelProgrammingLab2/graph.py" --csv "$CSV_FILE" --output "$PLOT_FILE" --mode "$MODE" || {
    echo "Failed to generate plot"
    # Continue even if plot generation fails
}

# Deactivate virtual environment if it was activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

echo "Benchmark completed. Results saved to $CSV_FILE"
if [[ -f "$PLOT_FILE" ]]; then
    echo "Performance plot generated: $PLOT_FILE"
fi