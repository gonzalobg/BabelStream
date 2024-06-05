
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#include <openacc.h>

#define IMPLEMENTATION_STRING "OpenACC"

template <class T>
class ACCStream : public Stream<T>
{
    // Size of arrays
    intptr_t array_size;
    // Device side pointers
    T* restrict a;
    T* restrict b;
    T* restrict c;
    scan_t<T>* restrict s_i;
    scan_t<T>* restrict s_o;

  public:
    ACCStream(BenchId bs, const intptr_t array_size, const int device_id,
	      T initA, T initB, T initC);
    ~ACCStream();

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
