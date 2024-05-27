
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <type_traits>

// Array values
#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

template <typename T>
using scan_t = std::conditional_t<sizeof(T) == 4, std::uint32_t,
               std::conditional_t<sizeof(T) == 8, std::uint64_t, void>>;

template <class T>
class Stream
{
  public:
    virtual ~Stream(){}

    // Kernels
    // These must be blocking calls
    virtual void copy() = 0;
    virtual void mul() = 0;
    virtual void add() = 0;
    virtual void triad() = 0;
    virtual void nstream() = 0;
    virtual T dot() = 0;
    virtual void read() = 0;
    virtual void write(T initA) = 0;
    virtual void scan() = 0;

    // Copy memory between host and device
    virtual void init_arrays(T initA, T initB, T initC) = 0;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c,
                             std::vector<scan_t<T>>& s) = 0;
};

// Implementation specific device functions
void listDevices(void);
std::string getDeviceName(const int);
std::string getDeviceDriver(const int);

