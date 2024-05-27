// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "CUDAStream.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

[[noreturn]] inline void error(char const* file, int line, char const* expr, cudaError_t e) {
  std::fprintf(stderr, "Error at %s:%d: %s (%d)\n  %s\n", file, line, cudaGetErrorString(e), e, expr);
  exit(e);
}

// The do while is there to make sure you remember to put a semi-colon after calling CU
#define CU(EXPR) do { auto __e = (EXPR); if (__e != cudaSuccess) error(__FILE__, __LINE__, #EXPR, __e); } while(false)

// It is best practice to include __device__ and constexpr even though in BabelStream it only needs to be __host__ const
__host__ __device__ constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

cudaStream_t stream;

template <typename T>
T* alloc_device(const intptr_t array_size) {
  size_t array_bytes = sizeof(T) * array_size;
  T* p = nullptr;
#if defined(MANAGED)
  CU(cudaMallocManaged(&p, array_bytes));
#elif defined(PAGEFAULT)
  p = (T*)malloc(array_bytes);
#else
  CU(cudaMalloc(&p, array_bytes));
#endif
  if (p == nullptr) throw std::runtime_error("Failed to allocate device array");
  return p;
}

template <typename T>
T* alloc_host(const intptr_t array_size) {
  size_t array_bytes = sizeof(T) * array_size;
  T* p = nullptr;
#if defined(PAGEFAULT)
  p = (T*)malloc(array_bytes);
#else
  CU(cudaHostAlloc(&p, array_bytes, cudaHostAllocDefault));
#endif
  if (p == nullptr) throw std::runtime_error("Failed to allocate host array");
  return p;
}

template <typename T>
void free_device(T* p) {
#if defined(PAGEFAULT)
  free(p);
#else
  CU(cudaFree(p));
#endif
}

template <typename T>
void free_host(T* p) {
#if defined(PAGEFAULT)
  free(p);
#else
  CU(cudaFreeHost(p));
#endif
}

template <class T>
CUDAStream<T>::CUDAStream(const intptr_t array_size, const int device_index, const bool will_run_scan)
  : array_size(array_size), alloc_scan(will_run_scan)
{
  // Set device
  int count;
  CU(cudaGetDeviceCount(&count));
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  CU(cudaSetDevice(device_index));

  CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

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

  // Query device for sensible dot kernel block count
  cudaDeviceProp props;
  CU(cudaGetDeviceProperties(&props, device_index));
  dot_num_blocks = props.multiProcessorCount * 4;

  // Size of partial sums for dot kernels
  size_t sums_bytes = sizeof(T) * dot_num_blocks;
  size_t array_bytes = sizeof(T) * array_size;
  size_t scan_bytes = alloc_scan? size_t(2) * array_size * sizeof(scan_t<T>) : 0;
  size_t total_bytes = array_bytes * size_t(3) + scan_bytes + sums_bytes;
  std::cout << "Reduction kernel config: " << dot_num_blocks << " groups of (fixed) size " << TBSIZE << std::endl;

  // Check buffers fit on the device
  if (props.totalGlobalMem < total_bytes)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Allocate buffers:
  d_a = alloc_device<T>(array_size);
  d_b = alloc_device<T>(array_size);
  d_c = alloc_device<T>(array_size);
  sums = alloc_host<T>(dot_num_blocks);
  if (alloc_scan) {
    d_si = alloc_device<scan_t<T>>(array_size);
    d_so = alloc_device<scan_t<T>>(array_size);
  }
}

template <class T>
CUDAStream<T>::~CUDAStream()
{
  CU(cudaStreamDestroy(stream));
  free_device(d_a);
  free_device(d_b);
  free_device(d_c);
  free_host(sums);
  if (alloc_scan) {
    free_device(d_si);
    free_device(d_so);
  }
}

template <typename T>
__global__ void init_kernel(T* a, T* b, T* c, scan_t<T>* si, T initA, T initB, T initC, size_t array_size, bool alloc_scan)
{  
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
    if (alloc_scan) {
      si[i] = static_cast<scan_t<T>>(i);
    }
  }
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  init_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, d_si, initA, initB, initC, array_size, alloc_scan);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, std::vector<scan_t<T>>& s)
{
  if (alloc_scan && s.size() != array_size) {
    std::cerr << "Host Scan array size mismatch" << std::endl;
    std::terminate();
  }
  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  CU(cudaStreamSynchronize(stream));
  for (intptr_t i = 0; i < array_size; ++i)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
    if (alloc_scan) {
      s[i] = d_so[i];
    }
  }
#else
  CU(cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost));
  if (alloc_scan) {
    CU(cudaMemcpy(s.data(), d_so, c.size()*sizeof(scan_t<T>), cudaMemcpyDeviceToHost));
  }
#endif
}

template <typename T>
__global__ void copy_kernel(const T * a, T * c, size_t array_size)
{
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    c[i] = a[i];
  }
}

template <class T>
void CUDAStream<T>::copy()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  copy_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c, size_t array_size)
{
  const T scalar = startScalar;
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    b[i] = scalar * c[i];
  }
}

template <class T>
void CUDAStream<T>::mul()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  mul_kernel<<<blocks, TBSIZE, 0, stream>>>(d_b, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c, size_t array_size)
{
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void CUDAStream<T>::add()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  add_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, array_size);
  CU(cudaPeekAtLastError());  
  CU(cudaStreamSynchronize(stream));
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c, size_t array_size)
{
  const T scalar = startScalar;
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void CUDAStream<T>::triad()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  triad_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c, size_t array_size)
{
  const T scalar = startScalar;
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
void CUDAStream<T>::nstream()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  nstream_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T* sums, size_t array_size)
{
  __shared__ T smem[TBSIZE];
  T tmp = T(0.);
  const size_t tidx = threadIdx.x;
  for (size_t i = tidx + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    tmp += a[i] * b[i];
  }
  smem[tidx] = tmp;

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    __syncthreads();
    if (tidx < offset) smem[tidx] += smem[tidx+offset];
  }

  // First thread writes to host memory directly from the device
  if (tidx == 0) sums[blockIdx.x] = smem[tidx];
}

template <class T>
T CUDAStream<T>::dot()
{
  dot_kernel<<<dot_num_blocks, TBSIZE, 0, stream>>>(d_a, d_b, sums, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));

  T sum = 0.0;
  for (intptr_t i = 0; i < dot_num_blocks; ++i) sum += sums[i];

  return sum;
}

template <class T>
void CUDAStream<T>::scan()
{
  if (!alloc_scan) {
    std::cerr << "Trying to run scan but storage not allocated" << std::endl;
    std::terminate();
  }
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_si, d_si + array_size, d_so);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <typename T>
__global__ void read_kernel(T* a, size_t array_size)
{
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    T tmp = a[i];
    // Control-dependency on loading a[i]: never true, but checking it requires loading value:
    if (tmp == T(3.14)) {
      a[i] = 2 * tmp;
    }
  }
}

template <class T>
void CUDAStream<T>::read()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  read_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <typename T>
__global__ void write_kernel(T* a, T initA, size_t array_size)
{
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    a[i] = initA;
  }
}

template <class T>
void CUDAStream<T>::write(T initA)
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  write_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, initA, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

void listDevices(void)
{
  // Get number of devices
  int count;
  CU(cudaGetDeviceCount(&count));

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}

std::string getDeviceName(const int device)
{
  cudaDeviceProp props;
  CU(cudaGetDeviceProperties(&props, device));
  return std::string(props.name);
}

std::string getDeviceDriver(const int device)
{
  CU(cudaSetDevice(device));
  int driver;
  CU(cudaDriverGetVersion(&driver));
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
