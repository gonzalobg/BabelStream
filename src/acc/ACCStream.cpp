
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <numeric>
#include "ACCStream.h"

template <class T>
ACCStream<T>::ACCStream(BenchId bs, const intptr_t array_size, const int device_id,
			T initA, T initB, T initC)
  : array_size{array_size}
{
  acc_device_t device_type = acc_get_device_type();
  acc_set_device_num(device_id, device_type);

  // Set up data region on device
  this->a = new T[array_size];
  this->b = new T[array_size];
  this->c = new T[array_size];

  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;

  if (needs_buffer(bs, 's')) {
    s_i = new scan_t<T>[array_size];
    s_o = new scan_t<T>[array_size];
  }

  #pragma acc enter data create(a[0:array_size], b[0:array_size], c[0:array_size])
  {}

  init_arrays(initA, initB, initC);
}

template <class T>
ACCStream<T>::~ACCStream()
{
  // End data region on device
  intptr_t array_size = this->array_size;

  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;

  #pragma acc exit data delete(a[0:array_size], b[0:array_size], c[0:array_size])
  {}

  delete[] a;
  delete[] b;
  delete[] c;

  if (s_i) {
    delete[] s_i;
    delete[] s_o;
  }
}

template <class T>
void ACCStream<T>::init_arrays(T initA, T initB, T initC)
{
  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }

  if (s_i) {
    for (intptr_t i = 0; i < array_size; i++)
    {
      s_i[i] = scan_t<T>(i);
    }
  }
}

template <class T>
void ACCStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c, scan_t<T> const*& h_s)
{
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma acc update host(a[0:array_size], b[0:array_size], c[0:array_size])
  {}

  h_a = a;
  h_b = b;
  h_c = c;

  if (s_o) {
    h_s = s_o;
  }
}

template <class T>
void ACCStream<T>::copy()
{
  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], c[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void ACCStream<T>::mul()
{
  const T scalar = startScalar;

  intptr_t array_size = this->array_size;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(b[0:array_size], c[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void ACCStream<T>::add()
{
  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void ACCStream<T>::triad()
{
  const T scalar = startScalar;

  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void ACCStream<T>::nstream()
{
  const T scalar = startScalar;

  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size],  b[0:array_size], c[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
T ACCStream<T>::dot()
{
  T sum{};

  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  #pragma acc parallel loop reduction(+:sum) present(a[0:array_size], b[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }

  return sum;
}

template <class T>
void ACCStream<T>::read()
{
  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  #pragma acc parallel loop present(a[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    T tmp = a[i];
    if (tmp == T(3.14)) {
      a[i] *= 2;;
    }
  }
}

template <class T>
void ACCStream<T>::write(T initA)
{
  intptr_t array_size = this->array_size;
  T * restrict a = this->a;
  #pragma acc parallel loop present(a[0:array_size]) wait
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
  }
}

template <class T>
void ACCStream<T>::scan()
{
  if (!s_i) {
    throw std::runtime_error("Trying to run scan but storage not allocated");
  }
  
  // OpenAcc doesn't have scan; run sequentially
  std::exclusive_scan(s_i, s_i + array_size, s_o, scan_t<T>(0));
}

void listDevices(void)
{
  // Get number of devices
  acc_device_t device_type = acc_get_device_type();
  int count = acc_get_num_devices(device_type);

  // Print device list
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << "There are " << count << " devices." << std::endl;
  }
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class ACCStream<float>;
template class ACCStream<double>;
