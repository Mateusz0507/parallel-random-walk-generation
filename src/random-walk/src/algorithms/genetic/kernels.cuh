#pragma once

#ifndef GENETIC_IMPROVED_KERNELS_CUH
#define GENETIC_IMPROVED_KERNELS_CUH

#include "algorithms/model/particle.cuh"

#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"

__global__ void kernel_improved_fitness_function(vector3* dev_random_walk, int N1, const real_t distance, const real_t precision, int* dev_invalid_distances, int* dev_random_walk_idx);
__global__ void kernel_init_tables(int generation_size, int* dev_generation_idx, int* dev_fitness_function);

#endif // !GENETIC_IMPROVED_KERNELS_CUH