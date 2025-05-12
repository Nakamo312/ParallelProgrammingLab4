#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <string>
#include <omp.h>
#include <cuda_runtime.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyCUDA(double* A, double* B, double* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void generateRandomMatrix(vector<vector<double>>& matrix) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 100.0);

    for (auto& row : matrix) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }
}

vector<vector<double>> readMatrixBinary(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open file: " + filename);

    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (auto& row : matrix) {
        file.read(reinterpret_cast<char*>(row.data()), cols * sizeof(double));
        if (!file) throw runtime_error("Error reading matrix data");
    }

    return matrix;
}

void writeMatrixBinary(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot create file: " + filename);

    size_t rows = matrix.size();
    size_t cols = rows > 0 ? matrix[0].size() : 0;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

    for (const auto& row : matrix) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
}

vector<vector<double>> multiplyMatricesSerial(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw runtime_error("Matrix dimensions mismatch for multiplication");
    }

    size_t m = A.size();
    size_t n = A[0].size();
    size_t p = B[0].size();

    vector<vector<double>> C(m, vector<double>(p, 0.0));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

vector<vector<double>> multiplyMatricesParallel(const vector<vector<double>>& A,
    const vector<vector<double>>& B, int num_threads) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw runtime_error("Matrix dimensions mismatch for multiplication");
    }

    const size_t m = A.size();
    const size_t n = A[0].size();
    const size_t p = B[0].size();

    vector<vector<double>> C(m, vector<double>(p, 0.0));

    omp_set_num_threads(num_threads);
    omp_set_schedule(omp_sched_static, 0); 

    #pragma omp parallel for collapse(2) schedule(runtime)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

vector<vector<double>> multiplyMatricesCUDA(const vector<vector<double>>& A,
    const vector<vector<double>>& B, int block_size) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw runtime_error("Matrix dimensions mismatch for multiplication");
    }

    const size_t m = A.size();
    const size_t n = A[0].size();
    const size_t p = B[0].size();

    // Flatten matrices
    vector<double> A_flat(m * n);
    vector<double> B_flat(n * p);
    vector<double> C_flat(m * p, 0.0);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_flat[i * n + j] = A[i][j];
        }
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            B_flat[i * p + j] = B[i][j];
        }
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_B, n * p * sizeof(double));
    cudaMalloc(&d_C, m * p * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, A_flat.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), n * p * sizeof(double), cudaMemcpyHostToDevice);

    // Configure kernel with specified block size
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x, 
                 (m + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMultiplyCUDA<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(C_flat.data(), d_C, m * p * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Convert back to 2D vector
    vector<vector<double>> C(m, vector<double>(p));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            C[i][j] = C_flat[i * p + j];
        }
    }

    return C;
}

vector<vector<double>> multiplyMatricesMPI(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        if (rank == 0) {
            throw runtime_error("Matrix dimensions mismatch for multiplication");
        }
        return {};
    }

    const size_t m = A.size();
    const size_t n = A[0].size();
    const size_t p = B[0].size();

    vector<vector<double>> C(m, vector<double>(p, 0.0));

    if (rank == 0) {
        // Master process distributes work and collects results
        int rows_per_process = m / size;
        int extra_rows = m % size;

        // Send data to worker processes
        for (int dest = 1; dest < size; ++dest) {
            int start_row = dest * rows_per_process + min(dest, extra_rows);
            int end_row = start_row + rows_per_process + (dest < extra_rows ? 1 : 0);
            
            // Send number of rows
            int rows_to_send = end_row - start_row;
            MPI_Send(&rows_to_send, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            
            // Send matrix data
            for (int i = start_row; i < end_row; ++i) {
                MPI_Send(A[i].data(), n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
            MPI_Send(B[0].data(), n * p, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }

        // Process master's share
        int start_row = 0;
        int end_row = rows_per_process + (extra_rows > 0 ? 1 : 0);
        
        for (int i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < p; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        // Receive results from workers
        for (int src = 1; src < size; ++src) {
            int start_row = src * rows_per_process + min(src, extra_rows);
            int rows_to_recv = rows_per_process + (src < extra_rows ? 1 : 0);
            
            for (int i = 0; i < rows_to_recv; ++i) {
                MPI_Recv(C[start_row + i].data(), p, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Worker processes
        int rows_to_recv;
        MPI_Recv(&rows_to_recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        vector<vector<double>> local_A(rows_to_recv, vector<double>(n));
        vector<vector<double>> local_B(n, vector<double>(p));
        vector<vector<double>> local_C(rows_to_recv, vector<double>(p, 0.0));
        
        // Receive A rows
        for (int i = 0; i < rows_to_recv; ++i) {
            MPI_Recv(local_A[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Receive B matrix
        for (int i = 0; i < n; ++i) {
            MPI_Recv(local_B[i].data(), p, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Compute
        for (int i = 0; i < rows_to_recv; ++i) {
            for (size_t j = 0; j < p; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    local_C[i][j] += local_A[i][k] * local_B[k][j];
                }
            }
        }
        
        // Send results back
        for (int i = 0; i < rows_to_recv; ++i) {
            MPI_Send(local_C[i].data(), p, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    return C;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (argc < 6) {
            cerr << "Usage modes:\n"
                << "1. Generate and save matrices: " << argv[0]
                << " generate <rows1> <cols1> <rows2> <cols2> <file1> <file2>\n"
                << "2. Multiply matrices (serial): " << argv[0]
                << " multiply serial <matrixA> <matrixB> <output>\n"
                << "3. Multiply matrices (parallel OpenMP): " << argv[0]
                << " multiply parallel <matrixA> <matrixB> <output> <threads>\n"
                << "4. Multiply matrices (CUDA): " << argv[0]
                << " multiply cuda <matrixA> <matrixB> <output> <block_size>\n"
                << "5. Multiply matrices (MPI): " << argv[0]
                << " multiply mpi <matrixA> <matrixB> <output>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        try {
            string mode(argv[1]);

            if (mode == "generate" && argc == 8) {
                int rows1 = stoi(argv[2]);
                int cols1 = stoi(argv[3]);
                int rows2 = stoi(argv[4]);
                int cols2 = stoi(argv[5]);

                vector<vector<double>> A(rows1, vector<double>(cols1));
                vector<vector<double>> B(rows2, vector<double>(cols2));

                generateRandomMatrix(A);
                generateRandomMatrix(B);

                writeMatrixBinary(argv[6], A);
                writeMatrixBinary(argv[7], B);

                cout << "Matrix A (" << rows1 << "x" << cols1 << ") saved to " << argv[6] << "\n";
                cout << "Matrix B (" << rows2 << "x" << cols2 << ") saved to " << argv[7] << "\n";
            }
            else if (mode == "multiply") {
                if (argc < 6) {
                    throw runtime_error("Not enough arguments for multiply mode");
                }

                string impl_type(argv[2]);
                auto A = readMatrixBinary(argv[3]);
                auto B = readMatrixBinary(argv[4]);
                string output_file = argv[5];
                int param = 0; // threads or block_size

                vector<vector<double>> C;
                auto start = high_resolution_clock::now();

                if (impl_type == "serial") {
                    C = multiplyMatricesSerial(A, B);
                }
                else if (impl_type == "parallel") {
                    if (argc != 7) {
                        throw runtime_error("Missing thread count for parallel mode");
                    }
                    param = stoi(argv[6]);
                    C = multiplyMatricesParallel(A, B, param);
                    cout << "Using " << param << " OpenMP threads\n";
                }
                else if (impl_type == "cuda") {
                    if (argc >= 7) {
                        param = stoi(argv[6]);
                    } else {
                        param = 16; // default block size
                    }
                    C = multiplyMatricesCUDA(A, B, param);
                    cout << "Using CUDA with block size " << param << "x" << param << "\n";
                }
                else if (impl_type == "mpi") {
                    C = multiplyMatricesMPI(A, B);
                }
                else {
                    throw runtime_error("Invalid implementation type");
                }

                auto end = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(end - start).count();

                cout << "Computation time: " << duration << " ms\n";
                cout << "GFLOPS: " 
                     << (2.0 * A.size() * A[0].size() * B[0].size()) / (duration * 1e6)
                     << "\n";

                if (impl_type != "mpi" || rank == 0) {
                    writeMatrixBinary(output_file, C);
                }
            }
            else {
                throw runtime_error("Invalid mode");
            }
        }
        catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    } else {
        if (argc > 1 && string(argv[1]) == "multiply" && string(argv[2]) == "mpi") {
            multiplyMatricesMPI({}, {});
        }
    }

    MPI_Finalize();
    return 0;
}