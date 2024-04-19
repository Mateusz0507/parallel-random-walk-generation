#include "algorithms/energetic/naive/energetic_naive.cuh"


__host__ __device__ float distance(algorithms::model::particle p1, algorithms::model::particle p2)
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

__global__ void iteration(algorithms::model::particle* particles, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
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

    if (i != N - 1)
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

    for (int j = 0; j < N; j++)
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

    particles[i].x += movement_x;
    particles[i].y += movement_y;
    particles[i].z += movement_z;
}

bool algorithms::energetic::naive_method::run(algorithms::model::particle** result, int N)
{
    if (allocate_memory(N))
    {
        generate_random_starting_points(N);


        /* Create pdb file with points position before the start of the algorithm */
        algorithms::model::particle* points_before_algorithm = new algorithms::model::particle[N];
        if (!cuda_check_continue(cudaMemcpy(points_before_algorithm, dev_points, N * sizeof(model::particle), cudaMemcpyDeviceToHost)))
        {
            release_memory();
            return false;
        }
        create_pdb_file(points_before_algorithm, N, "before");
        open_chimera("before");


        while (!validator.validate(dev_points, N, DISTANCE, EN_PRECISION))
        {
            iteration<<<N/32 + 1, 32>>>(dev_points, N);
        }

        if (!cuda_check_continue(cudaMemcpy(*result, dev_points, N * sizeof(model::particle), cudaMemcpyDeviceToHost)))
        {
            release_memory();
            return false;
        }

        release_memory();
        return true;
    }

    return false;
}

bool algorithms::energetic::naive_method::allocate_memory(int N)
{
    if (N < 0)
        return false;

    if (!cuda_check_continue(cudaMalloc(&dev_points, N * sizeof(model::particle))))
    {
        dev_points = nullptr;
        return false;
    }

    return true;
}

void algorithms::energetic::naive_method::release_memory()
{
    if (dev_points)
    {
        cuda_check_terminate(cudaFree(dev_points));
        dev_points = nullptr;
    }
}

bool algorithms::energetic::naive_method::generate_random_starting_points(int N)
{
    model::particle* starting_points = new model::particle[N];

    curandState* dev_states = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_states, N * sizeof(curandState))))
    {
        dev_states = nullptr;
        return false;
    }

    model::particle* dev_unit_vectors = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_unit_vectors, N * sizeof(model::particle))))
    {
        dev_unit_vectors = nullptr;
        return false;
    }

    /* Generate starting points */

    int number_of_blocks = (N + EN_BLOCK_SIZE - 1) / EN_BLOCK_SIZE;
    algorithms::directional_randomization::kernel_setup << <number_of_blocks, EN_BLOCK_SIZE >> > (dev_states, N, std::time(nullptr), 0);
    cuda_check_terminate(cudaDeviceSynchronize());

    algorithms::directional_randomization::kernel_generate_random_unit_vectors << <number_of_blocks, EN_BLOCK_SIZE >> > (dev_unit_vectors, dev_states, N);
    model::particle init = { 0.0, 0.0, 0.0 };

    // thrust no operator matches error resolved here https://stackoverflow.com/questions/18123407/cuda-thrust-reduction-with-double2-arrays
    // eventually thrust does not implement operator+ for float3 or double3
    thrust::device_ptr<model::particle> dev_unit_vectors_ptr = thrust::device_ptr<model::particle>(dev_unit_vectors);
    thrust::device_ptr<model::particle> dev_points_ptr = thrust::device_ptr<model::particle>(dev_points);
    model::add_particles add;
    cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_unit_vectors_ptr, dev_unit_vectors_ptr + N, dev_points_ptr, init, add));

    if (dev_unit_vectors)
    {
        cuda_check_terminate(cudaFree(dev_unit_vectors));
        dev_unit_vectors = nullptr;
    }
    if (dev_states)
    {
        cuda_check_terminate(cudaFree(dev_states));
        dev_states = nullptr;
    }

    return true;
}

algorithms::energetic::naive_method::naive_method(validators::abstract_validator& validator) : validator{ validator }
{
    this->dev_points = nullptr;
}
