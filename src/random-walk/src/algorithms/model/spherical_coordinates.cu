#include "algorithms/model/spherical_coordinates.cuh"

__device__ algorithms::model::spherical_coordinates::spherical_coordinates(float alpha, float beta)
{
	this->alpha = alpha;
	this->beta = beta;
}

__device__ float3 algorithms::model::spherical_coordinates::get_vector(float alpha, float beta)
{
	float x = sin(alpha) * cos(beta);
	float y = sin(alpha) * sin(beta);
	float z = cos(alpha);

	vector3 vector;
	vector.x = x;
	vector.y = y;
	vector.z = z;

	return vector;
}
