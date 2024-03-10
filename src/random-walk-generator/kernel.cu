
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

struct Particle {
    float x, y, z;
};

__host__ __device__ float distance(Particle p1, Particle p2)
{
    float x_distance = p1.x - p2.x;
    float y_distance = p1.y - p2.y;
    float z_distance = p1.z - p2.z;
    return sqrt(x_distance * x_distance + y_distance * y_distance + z_distance * z_distance);
}

void initialize_particles_locations(Particle* particles, const int n)
{
    particles[0].x = particles[0].y = particles[0].z = 0;

    srand((unsigned int)time(NULL));
    for (int i = 1; i < n; i++)
    {
        float versor_x = 2 * ((float)rand() / (float)(RAND_MAX)) - 1;
        float versor_y = 2 * ((float)rand() / (float)(RAND_MAX)) - 1;
        float versor_z = 2 * ((float)rand() / (float)(RAND_MAX)) - 1;
        float versor_length = sqrt(versor_x * versor_x + versor_y * versor_y + versor_z * versor_z);
        versor_x /= versor_length;
        versor_y /= versor_length;
        versor_z /= versor_length;
        particles[i].x = particles[i - 1].x + versor_x;
        particles[i].y = particles[i - 1].y + versor_y;
        particles[i].z = particles[i - 1].z + versor_z;
    }
}

void print_particles(Particle* particles, const int n)
{
    for (int i = 0; i < n; i++)
    {
        float distance_to_last_particle = distance(particles[i], particles[(i - 1 + n) % n]);
        printf("Particle %d: x: %f   y: %f   z : %f   distance to last: %f\n",
            i, particles[i].x, particles[i].y, particles[i].z, distance_to_last_particle);
    }
}

__global__ void check_correctness_kernel(Particle* particles, const int n, int* results)
{
    int i = threadIdx.x;
    float eps = 0.001;

    results[i] = 1;
    if (i != 0)
    {
        if (abs(distance(particles[i], particles[i - 1]) - 1) > eps)
            results[i] = 0;
    }

    for (int j = 0; j < n; j++)
    {
        if (j != i && abs(distance(particles[i], particles[j])) < 1 - eps)
            results[i] = 0;
    }
}

cudaError_t check_correctness(Particle* particles, const int n, bool* is_correct)
{
    int* dev_results = 0;
    Particle* dev_particles = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_results, n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_particles, n * sizeof(Particle));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_particles, particles, n * sizeof(Particle), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    check_correctness_kernel << <1, n >> > (dev_particles, n, dev_results);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fix_particles_locations launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    thrust::device_ptr<int> results_ptr = thrust::device_pointer_cast(dev_results);
    int result = thrust::reduce(results_ptr, results_ptr + n);
    *is_correct = result == n ? true : false;

Error:
    cudaFree(dev_particles);
    cudaFree(dev_results);

    return cudaStatus;
}

__global__ void fix_particles_locations_kernel(Particle* particles, const int n)
{
    int i = threadIdx.x;
    Particle particle = particles[i];

    particle.x = i + 0.6;
    particle.y = i + 0.7;
    particle.z = i + 0.8;

    particles[i] = particle;
}

cudaError_t fix_particles_locations(Particle* particles, const int n)
{
    Particle* dev_particles = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_particles, n * sizeof(Particle));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_particles, particles, n * sizeof(Particle), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    fix_particles_locations_kernel << <1, n >> > (dev_particles, n);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fix_particles_locations launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(particles, dev_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_particles);

    return cudaStatus;
}

int main()
{
    const int N = 3;
    Particle particles[N];
    initialize_particles_locations(particles, N);
    print_particles(particles, N);
    bool is_correct;
    check_correctness(particles, N, &is_correct);
    printf("Is correct? %s", is_correct ? "true" : "false");

    cudaError_t cudaStatus = fix_particles_locations(particles, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fix_particles_locations failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
