// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reservd.
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "CUDAStream.h"
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda/atomic>
#include <cuda/std/array>
#include <cuda/std/tuple>

static constexpr int max_sums = 512;

[[noreturn]] inline void error(char const* file, int line, cudaError_t e) {
  std::fprintf(stderr, "Error at %s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
  exit(e);
}

#define CU(EXPR) if (auto __e = (EXPR); __e != cudaSuccess) error(__FILE__, __LINE__, __e);

__host__ __device__ constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1)/b; }

cudaStream_t* stream(long long& s) { return (cudaStream_t*)&s; }

template <typename UnaryFunction>
__device__ int for_each(/*grid_group _,*/ int n, UnaryFunction&& f) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for (; i < n; i += gridDim.x * blockDim.x)
    f(i);
  return i;
}

template <typename T>
struct V {
  static constexpr int w = 16/sizeof(T);
  alignas(16) T v[w];
  __host__ __device__ constexpr T& operator[](int i) { return v[i]; }
  __host__ __device__ constexpr T operator[](int i) const { return v[i]; }
};

template <typename T, size_t N> using outs = cuda::std::array<T*, N>;
template <typename T, size_t N> using ins = cuda::std::array<T const*, N>;
template <typename T, size_t N> using vals = cuda::std::array<T, N>;

template <typename UnaryFunction, typename T, size_t N, size_t M>
__device__ void for_each_vec(/*grid_group _,*/ int n, outs<T, N> d, ins<T, M> s, UnaryFunction&& f) {
  using V = V<T>;
  constexpr int w = sizeof(V) / sizeof(T);

  cuda::std::array<V*, N> dv;
  for (int i = 0; i < N; ++i) dv[i] = (V*)d[i];
  cuda::std::array<V*, M> sv;
  for (int i = 0; i < M; ++i) sv[i] = (V*)s[i];

  const auto vl = ceil_div(n, w);
  int i = for_each(vl - 1, [&](int i) {
    cuda::std::array<V, M> svv;
    for (int j = 0; j < M; ++j) svv[j] = sv[j][i];
    cuda::std::array<V, N> dvv;
    for (int k = 0; k < w; ++k) {
      cuda::std::array<T, M> ins;
      for (int j = 0; j < M; ++j) ins[j] = svv[j][k];
      cuda::std::array<T, N> outs{cuda::std::apply(f, ins)};
      for (int j = 0; j < N; ++j) dvv[j][k] = outs[k];
    }
    for (int j = 0; j < N; ++j) dv[j][i] = dvv[j];
  });
  if (i == (vl - 1)) {
    for (int k = w * i; k < n; ++k) {
      cuda::std::array<T, M> ins;
      for (int j = 0; j < M; ++j) ins[j] = s[j][k];
      cuda::std::array<T, N> outs{cuda::std::apply(f, ins)};
      for (int j = 0; j < N; ++j) d[j][k] = outs[k];
    }
  }
}

template <typename T>
struct sum_t {
  alignas(512) cuda::atomic<T, cuda::thread_scope_device> data;
};

void blocks_and_threads(int& minGridSize, int& blockSize, size_t array_size, void* func, int esize,
			int maxBlockSize = 256, int maxWaveSize = 64) {
  auto dyn_smem = [] __host__ __device__ (int){ return 0; };
  CU(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, func, dyn_smem, 0));
  auto nthreads = minGridSize * blockSize;
  // Clamp at 256 threads:
  blockSize = std::min(blockSize, maxBlockSize);
  minGridSize = nthreads / blockSize;
  int vw = 16 / esize;
  int actualGridSize = ceil_div(array_size / vw, blockSize);
  if (maxWaveSize > -1) {
    // Clamp at n thread block waves:
    minGridSize = std::min(actualGridSize, maxWaveSize * minGridSize);
  } else {
    minGridSize = actualGridSize;
  }
}

template <typename F>
void autotune(char const* name, int& minGridSize, int& blockSize, int& num_dot_sums,
	      size_t array_size, void* func, int esize, F&& kernel) {
  constexpr int niter = 20;
  double dt = std::numeric_limits<double>::max();
  int minGridLocal = 0, minBlockLocal = 0, minSums = max_sums, minWaves = 0;
  std::vector<int> num_sums{-1};
  std::vector<int> block_sizes{128, 256, 512, 1024};
  bool with_sums = num_dot_sums != -1;
  if (with_sums) {
    block_sizes = std::vector<int>{512, 1024};
    num_sums = std::vector<int>{1, 2, 8, 32, 64, 128, 256, max_sums};
  }
  for (auto bs : block_sizes) {
    for (auto ws : {-1, 8, 16, 32, 64, 128, 256}) {
      for (auto ns : num_sums) {
	if (ns > max_sums) abort();
	if (with_sums) num_dot_sums = ns;

	using clk_t = std::chrono::high_resolution_clock;
	using dur_t = std::chrono::duration<double>;

	blocks_and_threads(minGridSize, blockSize, array_size, func, esize, bs, ws);

	kernel();
	auto s = clk_t::now();
	for (int it = 0; it < niter; ++it) kernel();
	auto t = (clk_t::now() - s).count();
	if (t < dt) {
	  minGridLocal = minGridSize;
	  minBlockLocal = blockSize;
	  minWaves = ws;
	  if (ns != -1) minSums = ns;
	  dt = t;
	}
      }
    }
  }
  minGridSize = minGridLocal;
  blockSize = minBlockLocal;
  if (with_sums) num_dot_sums = minSums;

  std::cout << name << " kernel config: " << minGridSize << " groups of (fixed) size " << blockSize
	    << " in " << minWaves << " waves ";
  if (with_sums)
    std::cout << " with " << num_dot_sums << " sums";
  std::cout << std::endl;
}

template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC, size_t array_size) {
  for_each(array_size, [=](int i) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  });
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC) {
  constexpr int threads_per_block = 256;
  size_t blocks = ceil_div(array_size, threads_per_block);
  init_kernel<<<blocks, threads_per_block, 0, *stream(s)>>>(d_a, d_b, d_c, initA, initB, initC, array_size);
  CU(cudaStreamSynchronize(*stream(s)));
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) {
  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  CU(cudaStreamSynchronize(*stream(s)));
  for (size_t i = 0; i < array_size; i++) {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  CU(cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost));
#endif
}

template <typename T>
__global__ void copy_kernel(const T * a, T * c, size_t array_size) {
  using V = V<T>;
  for_each_vec(array_size, outs<T,1>{c}, ins<T,1>{a}, [=](T a) {
    return a;
  });
}

template <class T>
void CUDAStream<T>::copy() {
  copy_kernel<<<num_blocks_copy, num_threads_copy, 0, *stream(s)>>>(d_a, d_c, array_size);
  CU(cudaStreamSynchronize(*stream(s)));
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c, size_t array_size) {
  const T scalar = startScalar;
  for_each_vec(array_size, outs<T, 1>{b}, ins<T, 1>{c}, [](T c) {
    return c * scalar;
  });
}

template <class T>
void CUDAStream<T>::mul() {
  mul_kernel<<<num_blocks_mul, num_threads_mul, 0, *stream(s)>>>(d_b, d_c, array_size);
  CU(cudaStreamSynchronize(*stream(s)));
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c, size_t array_size) {
  for_each_vec(array_size, outs<T, 1>{c}, ins<T, 2>{a, b}, [](T a, T b) {
    return a + b;
  });
}

template <class T>
void CUDAStream<T>::add() {
  add_kernel<<<num_blocks_add, num_threads_add, 0, *stream(s)>>>(d_a, d_b, d_c, array_size);
  CU(cudaStreamSynchronize(*stream(s)));
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c, size_t array_size) {
  const T scalar = startScalar;
  for_each_vec(array_size, outs<T, 1>{a}, ins<T, 2>{b, c}, [](T b, T c) {
    return b + c * scalar;
  });
}

template <class T>
void CUDAStream<T>::triad() {
  triad_kernel<<<num_blocks_triad, num_threads_triad, 0, *stream(s)>>>(d_a, d_b, d_c, array_size);
  CU(cudaStreamSynchronize(*stream(s)));
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c, size_t array_size) {
  const T scalar = startScalar;
  for_each_vec(array_size, outs<T, 1>{a}, ins<T, 3>{a, b, c}, [=](T a, T b, T c) {
    return a + b + scalar * c;
  });
}

template <class T>
void CUDAStream<T>::nstream() {
  nstream_kernel<<<num_blocks_nstream, num_threads_nstream, 0, *stream(s)>>>(d_a, d_b, d_c, array_size);
  CU(cudaStreamSynchronize(*stream(s)));
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, sum_t<T>* sums, int num_sums, int array_size) {
  namespace cg = cooperative_groups;
  using V = V<T>;
  __shared__ T data[32];

  T init = T{0};
  for_each_vec(array_size, outs<T, 0>{}, ins<T, 2>{a, b}, [&init](T a, T b) {
    init += a * b;
    return vals<T, 0>{};
  });

  auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  auto r = cg::reduce(tile, init, cg::plus<T>{});
  cg::invoke_one(tile, [&] {
    data[tile.meta_group_rank()] = r;
  });
  __syncthreads();
  if (threadIdx.x < tile.meta_group_size()) {
    auto g = cg::coalesced_threads();
    auto r = cg::reduce(g, data[threadIdx.x], cg::plus<T>{});
    cg::invoke_one(g, [&] {
      sums[blockIdx.x % num_sums].data.fetch_add(r, cuda::memory_order_relaxed);
    });
  }
}

template <class T>
T CUDAStream<T>::dot() {
  sum_t<T>* p = (sum_t<T>*)sums;
  for (int i = 0; i < num_dot_sums; ++i) p[i].data.store(T(0), cuda::memory_order_relaxed);
  dot_kernel<<<num_blocks_dot, num_threads_dot, 0, *stream(s)>>>(d_a, d_b, p, num_dot_sums, array_size);
  CU(cudaStreamSynchronize(*stream(s)));
  T sum = 0;
  for (int i = 0; i < num_dot_sums; ++i) sum += p[i].data.load(cuda::memory_order_relaxed);
  return sum;
}

template <class T>
CUDAStream<T>::CUDAStream(const int ARRAY_SIZE, const int device_index) {
  // Set device
  int count;
  CU(cudaGetDeviceCount(&count));
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  CU(cudaSetDevice(device_index));

  // Print out device information
  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
#if defined(MANAGED)
  std::cout << "Memory: MANAGED" << std::endl;
#elif defined(PAGEFAULT)
  std::cout << "Memory: PAGEFAULT" << std::endl;
#else
  std::cout << "Memory: DEFAULT" << std::endl;
#endif
  array_size = ARRAY_SIZE;

  CU(cudaStreamCreate(stream(s)));

  // Check buffers fit on the device
  size_t array_bytes = sizeof(T);
  array_bytes *= ARRAY_SIZE;
  size_t total_bytes = array_bytes * 4;
  
  cudaDeviceProp props;
  CU(cudaGetDeviceProperties(&props, device_index));
  if (props.totalGlobalMem < total_bytes)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
#if defined(MANAGED)
  CU(cudaMallocManaged(&d_a, array_bytes));
  CU(cudaMallocManaged(&d_b, array_bytes));
  CU(cudaMallocManaged(&d_c, array_bytes));
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(array_bytes);
  d_b = (T*)malloc(array_bytes);
  d_c = (T*)malloc(array_bytes);
#else
  CU(cudaMalloc(&d_a, array_bytes));
  CU(cudaMalloc(&d_b, array_bytes));
  CU(cudaMalloc(&d_c, array_bytes));
#endif
  sums = (long long*)malloc(sizeof(sum_t<T>) * max_sums);
  CU(cudaHostRegister(sums, sizeof(sum_t<T>) * max_sums, cudaHostRegisterDefault));
  num_dot_sums = -1;

  // Query sensible device properties for the different kernels
  autotune("Copy", num_blocks_copy, num_threads_copy, num_dot_sums, array_size, (void*)copy_kernel<T>, sizeof(T), [&] { copy(); });
  autotune("Mul", num_blocks_mul, num_threads_mul, num_dot_sums, array_size, (void*)mul_kernel<T>, sizeof(T), [&] { mul(); });
  autotune("Add", num_blocks_add, num_threads_add, num_dot_sums, array_size, (void*)add_kernel<T>, sizeof(T), [&] { add(); });
  autotune("Triad", num_blocks_triad, num_threads_triad, num_dot_sums, array_size, (void*)triad_kernel<T>, sizeof(T), [&] { triad(); });
  autotune("Nstream", num_blocks_nstream, num_threads_nstream, num_dot_sums, array_size, (void*)nstream_kernel<T>, sizeof(T), [&] { nstream(); });
  num_dot_sums = 1;
  autotune("Dot", num_blocks_dot, num_threads_dot, num_dot_sums, array_size, (void*)dot_kernel<T>, sizeof(T), [&] { dot(); });
}

template <class T>
CUDAStream<T>::~CUDAStream() {
#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
#else
  CU(cudaFree(d_a));
  CU(cudaFree(d_b));
  CU(cudaFree(d_c));
  CU(cudaStreamDestroy(*stream(s)));
  CU(cudaHostUnregister(sums));
#endif
  free(sums);
}


void listDevices(void) {
  // Get number of devices
  int count;
  CU(cudaGetDeviceCount(&count));

  // Print device names
  if (count == 0) {
    std::cerr << "No devices found." << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++) {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}

std::string getDeviceName(const int device) {
  cudaDeviceProp props;
  CU(cudaGetDeviceProperties(&props, device));
  return std::string(props.name);
}


std::string getDeviceDriver(const int device) {
  CU(cudaSetDevice(device));
  int driver;
  CU(cudaDriverGetVersion(&driver));
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
