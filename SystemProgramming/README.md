# System Programming Portfolio

High-performance system-level programming in C++ and Rust.

## Projects

### 1. High-Performance Computing (C++)
Matrix operations optimized with SIMD, OpenMP, and cache-friendly algorithms.

**Features:**
- AVX2 SIMD vectorization
- OpenMP parallelization
- Cache-optimized algorithms
- Blocked matrix multiplication
- Comprehensive benchmarks

**Tech Stack:** C++17, AVX2, OpenMP, SIMD intrinsics

### 2. Memory Allocator (Rust)
Custom memory allocators with different strategies.

**Features:**
- Bump allocator (fast linear allocation)
- Linked list allocator (general purpose)
- Pool allocator (fixed-size blocks)
- Thread-safe with atomics
- Zero-cost abstractions

**Tech Stack:** Rust, unsafe code, atomics

### 3. Concurrency Library (Coming Soon)
Lock-free data structures and concurrent algorithms

## Technologies

- **Languages:** C++17/20, Rust
- **Optimizations:** SIMD, OpenMP, lock-free algorithms
- **Tools:** CMake, Cargo, perf, valgrind

## Build

### C++
```bash
g++ -O3 -march=native -fopenmp matrix_operations.cpp -o matrix_ops
./matrix_ops
```

### Rust
```bash
cargo build --release
cargo run --release
```
