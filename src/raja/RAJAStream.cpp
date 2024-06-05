
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include <stdexcept>
#include "RAJAStream.hpp"

using RAJA::forall;

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <typename T>
T* alloc(intptr_t n) {
#ifdef RAJA_TARGET_CPU
  return (T*)aligned_alloc(ALIGNMENT, sizeof(T)*n);
#else
  T* p;
  cudaMallocManaged((void**)&p, sizeof(T)*n, cudaMemAttachGlobal);
  return p;
#endif  
}

template <typename T>
void dealloc(T* p) {
#ifdef RAJA_TARGET_CPU
  free(p);
#else
  cudaFree(p);
#endif  
}

template <class T>
RAJAStream<T>::RAJAStream(BenchId bs, const intptr_t array_size, const int device_index,
			  T initA, T initB, T initC)
  : array_size(array_size), range(0, array_size)
{
  d_a = alloc<T>(array_size);
  d_b = alloc<T>(array_size);
  d_c = alloc<T>(array_size);

  if (needs_buffer(bs, 's')) {
    d_si = alloc<scan_t<T>>(array_size);
    d_so = alloc<scan_t<T>>(array_size);
  }

  init_arrays(initA, initB, initC);
}

template <class T>
RAJAStream<T>::~RAJAStream()
{
  dealloc(d_a);
  dealloc(d_b);
  dealloc(d_c);
  if (d_si) {
    dealloc(d_si);
    dealloc(d_so);
  }
}

template <class T>
void RAJAStream<T>::init_arrays(T initA, T initB, T initC)
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  scan_t<T>* RAJA_RESTRICT s = d_si;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    a[index] = initA;
    b[index] = initB;
    c[index] = initC;
    if (s) s[index] = index;
  });
}

template <class T>
void RAJAStream<T>::get_arrays(T const*& a, T const*& b, T const*& c, scan_t<T> const*& s)
{
  a = d_a;
  b = d_b;
  c = d_c;
  s = d_so;
}

template <class T>
void RAJAStream<T>::copy()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT c = d_c;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    b[index] = scalar*c[index];
  });
}

template <class T>
void RAJAStream<T>::add()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index] + b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    a[index] = b[index] + scalar*c[index];
  });
}

template <class T>
void RAJAStream<T>::nstream()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    a[index] += b[index] + scalar * c[index];;
  });
}

template <class T>
T RAJAStream<T>::dot()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;

  RAJA::ReduceSum<reduce_policy, T> sum(T{});

  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    sum += a[index] * b[index];
  });

  return T(sum);
}

template <class T>
void RAJAStream<T>::read()
{
  T* RAJA_RESTRICT a = d_a;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    T tmp = a[index];
    // Control-dependency on loading a[i]: never true, but checking it requires loading value:
    if (tmp == T(3.14)) {
      a[index] *= 2;
    }
  });
}

template <class T>
void RAJAStream<T>::write(T initA)
{
  T* RAJA_RESTRICT a = d_a;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    a[index] = initA;
  });
}

template <class T>
void RAJAStream<T>::scan()
{
  if (!d_si) throw std::runtime_error("trying to call scan without allocating memory");
  scan_t<T>* RAJA_RESTRICT si = d_si;
  scan_t<T>* RAJA_RESTRICT so = d_so;
  RAJA::exclusive_scan<policy>(RAJA::make_span(si, array_size), RAJA::make_span(so, array_size));
}

void listDevices(void)
{
  std::cout << "This is not the device you are looking for.";
}


std::string getDeviceName(const int device)
{
  return "RAJA";
}


std::string getDeviceDriver(const int device)
{
  return "RAJA";
}

template class RAJAStream<float>;
template class RAJAStream<double>;
