
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct Particle {
    float x, y, z;
};

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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
        float x_distance = particles[i].x - particles[(i - 1 + n) % n].x;
        float y_distance = particles[i].y - particles[(i - 1 + n) % n].y;
        float z_distance = particles[i].z - particles[(i - 1 + n) % n].z;
        float distance_to_last_particle = sqrt(x_distance * x_distance + y_distance * y_distance + z_distance * z_distance);
        printf("Particle %d: x: %f   y: %f   z : %f   distance to last: %f\n",
            i, particles[i].x, particles[i].y, particles[i].z, distance_to_last_particle);
    }
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
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    const int N = 10;
    Particle particles[N];
    initialize_particles_locations(particles, N);
    print_particles(particles, N);
    fix_particles_locations(particles, N);
    print_particles(particles, N);

    cudaError_t cudaStatus = fix_particles_locations(particles, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fix_particles_locations failed!");
        return 1;
    }

    // Add vectors in parallel.
    cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
