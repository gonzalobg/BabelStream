
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith, Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include "SerialStream.h"

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
SerialStream<T>::SerialStream(BenchId bs, const intptr_t array_size, const int device_id,
			      T initA, T initB, T initC)
  : array_size{array_size}
{
  // Allocate on the host
  this->a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  if (needs_buffer(bs, 's')) {
    this->si = (scan_t<T>*)aligned_alloc(ALIGNMENT, sizeof(scan_t<T>)*array_size);
    this->so = (scan_t<T>*)aligned_alloc(ALIGNMENT, sizeof(scan_t<T>)*array_size);
  }

  init_arrays(initA, initB, initC);
}

template <class T>
SerialStream<T>::~SerialStream()
{
  free(a);
  free(b);
  free(c);
  if (si) {
    free(si);
    free(so);
  }
}

template <class T>
void SerialStream<T>::init_arrays(T initA, T initB, T initC)
{
  intptr_t array_size = this->array_size;
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
    if (si) {
      si[i] = i;
    }
  }
}

template <class T>
void SerialStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c, scan_t<T> const*& h_s)
{
  h_a = a;
  h_b = b;
  h_c = c;
  h_s = so;
}

template <class T>
void SerialStream<T>::copy()
{
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void SerialStream<T>::mul()
{
  const T scalar = startScalar;
  for (intptr_t i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void SerialStream<T>::add()
{
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void SerialStream<T>::triad()
{
  const T scalar = startScalar;
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void SerialStream<T>::nstream()
{
  const T scalar = startScalar;
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
T SerialStream<T>::dot()
{
  T sum{};
  for (intptr_t i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }
  return sum;
}

template <class T>
void SerialStream<T>::read()
{
  for (intptr_t i = 0; i < array_size; i++)
  {
    T tmp = a[i];
    // Control-dependency on loading a[i]: never true, but checking it requires loading value:
    if (tmp == T(3.14)) {
      a[i] *= 2;
    }
  }
}

template <class T>
void SerialStream<T>::write(T initA)
{
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
  }
}

template <class T>
void SerialStream<T>::scan()
{
  scan_t<T> s = 0;
  for (intptr_t i = 0; i < array_size; i++)
  {
    so[i] = s;
    s += si[i];
  }
}

void listDevices(void)
{
  std::cout << "0: CPU" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class SerialStream<float>;
template class SerialStream<double>;
