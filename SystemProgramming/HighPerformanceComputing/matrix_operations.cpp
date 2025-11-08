/**
 * High-Performance Matrix Operations
 * Optimized linear algebra using SIMD, OpenMP, and cache-friendly algorithms
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <immintrin.h>  // AVX/AVX2 intrinsics
#include <omp.h>        // OpenMP
#include <memory>

// Matrix class with aligned memory for SIMD
template<typename T>
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::unique_ptr<T[], decltype(&_mm_free)> data_;

public:
    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols),
          data_(static_cast<T*>(_mm_malloc(rows * cols * sizeof(T), 64)),
                &_mm_free) {
        std::fill_n(data_.get(), rows_ * cols_, T(0));
    }

    ~Matrix() = default;

    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_),
          data_(static_cast<T*>(_mm_malloc(rows_ * cols_ * sizeof(T), 64)),
                &_mm_free) {
        std::copy_n(other.data_.get(), rows_ * cols_, data_.get());
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_.reset(static_cast<T*>(_mm_malloc(rows_ * cols_ * sizeof(T), 64)));
            std::copy_n(other.data_.get(), rows_ * cols_, data_.get());
        }
        return *this;
    }

    T& operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }
    const T& operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }
};

// Naive matrix multiplication (baseline)
template<typename T>
Matrix<T> matmul_naive(const Matrix<T>& A, const Matrix<T>& B) {
    Matrix<T> C(A.rows(), B.cols());

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            T sum = 0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

// Cache-optimized matrix multiplication (loop reordering)
template<typename T>
Matrix<T> matmul_cache_optimized(const Matrix<T>& A, const Matrix<T>& B) {
    Matrix<T> C(A.rows(), B.cols());

    // Reorder loops: i-k-j instead of i-j-k for better cache locality
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            T a_ik = A(i, k);
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }

    return C;
}

// SIMD-optimized matrix multiplication (AVX2)
Matrix<float> matmul_simd_avx2(const Matrix<float>& A, const Matrix<float>& B) {
    Matrix<float> C(A.rows(), B.cols());

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            __m256 a_ik = _mm256_set1_ps(A(i, k));

            size_t j = 0;
            // Process 8 elements at a time using AVX2
            for (; j + 7 < B.cols(); j += 8) {
                __m256 b = _mm256_loadu_ps(&B(k, j));
                __m256 c = _mm256_loadu_ps(&C(i, j));
                c = _mm256_fmadd_ps(a_ik, b, c);  // c += a_ik * b
                _mm256_storeu_ps(&C(i, j), c);
            }

            // Handle remaining elements
            for (; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return C;
}

// OpenMP parallelized matrix multiplication
template<typename T>
Matrix<T> matmul_parallel(const Matrix<T>& A, const Matrix<T>& B) {
    Matrix<T> C(A.rows(), B.cols());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            T a_ik = A(i, k);
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }

    return C;
}

// Combined SIMD + OpenMP for maximum performance
Matrix<float> matmul_hybrid(const Matrix<float>& A, const Matrix<float>& B) {
    Matrix<float> C(A.rows(), B.cols());

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            __m256 a_ik = _mm256_set1_ps(A(i, k));

            size_t j = 0;
            for (; j + 7 < B.cols(); j += 8) {
                __m256 b = _mm256_loadu_ps(&B(k, j));
                __m256 c = _mm256_loadu_ps(&C(i, j));
                c = _mm256_fmadd_ps(a_ik, b, c);
                _mm256_storeu_ps(&C(i, j), c);
            }

            for (; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return C;
}

// Blocked matrix multiplication for better cache utilization
template<typename T>
Matrix<T> matmul_blocked(const Matrix<T>& A, const Matrix<T>& B, size_t block_size = 64) {
    Matrix<T> C(A.rows(), B.cols());

    for (size_t ii = 0; ii < A.rows(); ii += block_size) {
        for (size_t jj = 0; jj < B.cols(); jj += block_size) {
            for (size_t kk = 0; kk < A.cols(); kk += block_size) {
                // Process block
                size_t i_max = std::min(ii + block_size, A.rows());
                size_t j_max = std::min(jj + block_size, B.cols());
                size_t k_max = std::min(kk + block_size, A.cols());

                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        T a_ik = A(i, k);
                        for (size_t j = jj; j < j_max; ++j) {
                            C(i, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }

    return C;
}

// Benchmark utility
template<typename Func>
double benchmark(Func&& func, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << name << ": " << duration << " ms\n";

    return duration;
}

int main() {
    const size_t N = 1024;  // Matrix size

    std::cout << "Initializing matrices (" << N << "x" << N << ")...\n";

    // Initialize random matrices
    Matrix<float> A(N, N);
    Matrix<float> B(N, N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A(i, j) = dist(gen);
            B(i, j) = dist(gen);
        }
    }

    std::cout << "\nRunning benchmarks...\n\n";

    // Benchmark different implementations
    Matrix<float> C;

    std::cout << "=== Naive Implementation ===\n";
    benchmark([&]() { C = matmul_naive(A, B); }, "Naive");

    std::cout << "\n=== Cache-Optimized ===\n";
    benchmark([&]() { C = matmul_cache_optimized(A, B); }, "Cache-Optimized");

    std::cout << "\n=== SIMD (AVX2) ===\n";
    benchmark([&]() { C = matmul_simd_avx2(A, B); }, "SIMD AVX2");

    std::cout << "\n=== OpenMP Parallel ===\n";
    benchmark([&]() { C = matmul_parallel(A, B); }, "OpenMP Parallel");

    std::cout << "\n=== Hybrid (SIMD + OpenMP) ===\n";
    benchmark([&]() { C = matmul_hybrid(A, B); }, "Hybrid");

    std::cout << "\n=== Blocked (Tiled) ===\n";
    benchmark([&]() { C = matmul_blocked(A, B, 64); }, "Blocked");

    return 0;
}
