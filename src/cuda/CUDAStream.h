// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reservd.
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#define IMPLEMENTATION_STRING "CUDA"

template <class T>
class CUDAStream : public Stream<T>
{
  protected:
    // Size of arrays
    size_t array_size;

    // Host array for partial sums for dot kernel
    long long* sums;
    int num_dot_sums;

    // Device side pointers to arrays
    T *d_a, *d_b, *d_c;

    // Number of blocks per grid:
    int num_blocks_copy, num_blocks_mul, num_blocks_add,
        num_blocks_triad, num_blocks_dot, num_blocks_nstream;

    // Number of threads per block:
    int num_threads_copy, num_threads_mul, num_threads_add,
        num_threads_triad, num_threads_dot, num_threads_nstream;

    long long s;

  public:

    CUDAStream(const int, const int);
    ~CUDAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
