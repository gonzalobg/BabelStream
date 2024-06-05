// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include "Stream.h"

#define IMPLEMENTATION_STRING "Kokkos"

template <class T>
class KokkosStream : public Stream<T>
{
  protected:
    // Size of arrays
    intptr_t array_size;

    // Device side pointers to arrays
    typename Kokkos::View<T*>* d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    typename Kokkos::View<scan_t<T>*>* d_si = nullptr, *d_so = nullptr;
    typename Kokkos::View<T*>::HostMirror* hm_a, *hm_b, *hm_c;
    typename Kokkos::View<scan_t<T>*>::HostMirror* hm_so;  

  public:

    KokkosStream(BenchId bs, const intptr_t array_size, const int device_id,
		 T initA, T initB, T initC);
    ~KokkosStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;
    void read() override;
    void write(T initA) override;
    void scan() override;

    void get_arrays(T const*& a, T const*& b, T const*& c, scan_t<T> const*& s) override;
    void init_arrays(T initA, T initB, T initC);
};

