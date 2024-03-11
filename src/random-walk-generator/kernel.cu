
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

__host__ __device__ float vector_length(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
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
    check_correctness_kernel << <n/32 + 1, 32 >> > (dev_particles, n, dev_results);

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
    printf("There are %d bad particles\n", n - result);

Error:
    cudaFree(dev_particles);
    cudaFree(dev_results);

    return cudaStatus;
}

__global__ void fix_particles_locations_kernel(Particle* particles, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float spring_strength = 0.5;
    float force_strength = 0.5;
    float movement_x = 0, movement_y = 0, movement_z = 0;
    float defect, direction_x, direction_y, direction_z, length;

    if (i != 0)
    {
        defect = distance(particles[i], particles[i - 1]) - 1;
        direction_x = particles[i].x - particles[i - 1].x;
        direction_y = particles[i].y - particles[i - 1].y;
        direction_z = particles[i].z - particles[i - 1].z;
        length = vector_length(direction_x, direction_y, direction_z);
        if (length > 0)
        {
            movement_x -= direction_x / length * defect * spring_strength;
            movement_y -= direction_y / length * defect * spring_strength;
            movement_z -= direction_z / length * defect * spring_strength;
        }
    }

    if (i != n)
    {
        defect = distance(particles[i], particles[i + 1]) - 1;
        direction_x = particles[i].x - particles[i + 1].x;
        direction_y = particles[i].y - particles[i + 1].y;
        direction_z = particles[i].z - particles[i + 1].z;
        length = vector_length(direction_x, direction_y, direction_z);
        if (length > 0)
        {
            movement_x -= direction_x / length * defect * spring_strength;
            movement_y -= direction_y / length * defect * spring_strength;
            movement_z -= direction_z / length * defect * spring_strength;
        }
    }

    for (int j = 0; j < n; j++)
    {
        if (j < i - 1 || j > i + 1)
        {
            defect = distance(particles[i], particles[j]) - 1;
            if (defect < 0)
            {
                direction_x = particles[i].x - particles[j].x;
                direction_y = particles[i].y - particles[j].y;
                direction_z = particles[i].z - particles[j].z;
                length = vector_length(direction_x, direction_y, direction_z);
                if (length > 0)
                {
                    movement_x -= direction_x / length * defect * force_strength;
                    movement_y -= direction_y / length * defect * force_strength;
                    movement_z -= direction_z / length * defect * force_strength;
                }
            }
        }
    }

    //length = vector_length(movement_x, movement_y, movement_z);

    particles[i].x += movement_x;
    particles[i].y += movement_y;
    particles[i].z += movement_z;
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
    fix_particles_locations_kernel << <n/32 + 1, 32 >> > (dev_particles, n);

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
    const int N = 2000;
    Particle particles[N];

    initialize_particles_locations(particles, N);

    cudaError_t cudaStatus;
    int iteration = 0;
    bool is_correct;
    check_correctness(particles, N, &is_correct);
    while (is_correct != true)
    {
        printf("Iteration: %d\n", iteration);
        //print_particles(particles, N);

        cudaStatus = fix_particles_locations(particles, N);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "fix_particles_locations failed!");
            return 1;
        }

        check_correctness(particles, N, &is_correct);
        iteration++;
    }

    printf("END\n");
    print_particles(particles, N);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
