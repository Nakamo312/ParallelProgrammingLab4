import numpy as np
import argparse
import csv
import os

def read_matrix_binary(filename):
    """Read matrix from binary file."""
    with open(filename, 'rb') as file:
        rows = np.fromfile(file, dtype=np.uint64, count=1)[0]
        cols = np.fromfile(file, dtype=np.uint64, count=1)[0]
        matrix = np.fromfile(file, dtype=np.float64).reshape(rows, cols)
    return matrix

def write_to_csv(filename, data):
    """Write or append data to CSV file."""
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'Dimension',
            'Threads',
            'Multiplication Time (s)',
            'Validation Result'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def main():
    parser = argparse.ArgumentParser(description='Validate matrix multiplication results and record performance.')
    parser.add_argument('matrix_a', help='Binary file for matrix A')
    parser.add_argument('matrix_b', help='Binary file for matrix B')
    parser.add_argument('matrix_c', help='Binary file for result matrix C')
    parser.add_argument('--csv', default='matrix_results.csv', help='CSV output file name')
    parser.add_argument('--duration', type=float, required=True, help='Multiplication time in milliseconds')
    parser.add_argument('--threads', type=int, required=True, help='Number of threads used for multiplication')
    args = parser.parse_args()

    try:
        A = read_matrix_binary(args.matrix_a)
        B = read_matrix_binary(args.matrix_b)
        C_expected = read_matrix_binary(args.matrix_c)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    if A.shape[1] != B.shape[0]:
        print("Error: Matrix dimensions are incompatible for multiplication.")
        print(f"A.shape: {A.shape}, B.shape: {B.shape}")
        return

    C_actual = np.dot(A, B)

    validation = True
    if C_expected.shape != C_actual.shape:
        validation = f"Size mismatch: {C_actual.shape} vs {C_expected.shape}"
    elif not np.allclose(C_expected, C_actual, rtol=1e-5, atol=1e-8):
        validation = False

    csv_data = {
        'Dimension': f"{A.shape[0]}",
        'Threads': args.threads,
        'Multiplication Time (s)': f"{args.duration * 1e-3:.6f}",  # Convert ms to seconds
        'Validation Result': validation
    }

    write_to_csv(args.csv, csv_data)

    print(f"- Threads: {args.threads}")
    print(f"- Validation Result: {csv_data['Validation Result']}")
    print(f"\nResults saved to: {args.csv}")

if __name__ == "__main__":
    main()