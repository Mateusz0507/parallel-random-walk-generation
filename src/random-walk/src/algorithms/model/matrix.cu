#include "algorithms/model/matrix.cuh"


algorithms::model::matrix::matrix(vector3 v1, vector3 v2, vector3 v3)
{
	m[0][0] = v1.x; m[0][1] = v2.x; m[0][2] = v3.x;
	m[1][0] = v1.y; m[1][1] = v2.y; m[1][2] = v3.y;
	m[2][0] = v1.z; m[2][1] = v2.z; m[2][2] = v3.z;
}

__device__ algorithms::model::matrix::matrix(spherical_coordinates coords)
{
	/*
	* Matrix of transformation to new basis where vector represented by coords is [1, 0, 0] in new basis.
	* If vector v has coordinates alpha, beta in spherical coordinate system and we use physics convention:
	* x = r * sin(alpha) * cos(beta)
	* y = r * sin(alpha) * sin(beta)
	* z = r * cos(alpha)
	* Then the basis in spherical coordinates are:
	* [1, 0, 0] ->  alpha,          beta
	* [0, 1, 0] ->  pi/2 - alpha,	beta + pi
	* [0, 0, 1] ->  pi/2,			beta + pi/2
	*/

	real_t alpha = coords.alpha;
	real_t beta = coords.beta;

	vector3 v1 = spherical_coordinates::get_vector(alpha, beta);
	vector3 v2 = spherical_coordinates::get_vector(PI/2 - alpha, beta + PI);
	vector3 v3 = spherical_coordinates::get_vector(PI/2, beta + PI/2);

	m[0][0] = v1.x; m[0][1] = v2.x; m[0][2] = v3.x;
	m[1][0] = v1.y; m[1][1] = v2.y; m[1][2] = v3.y;
	m[2][0] = v1.z; m[2][1] = v2.z; m[2][2] = v3.z;
}

__device__ vector3 algorithms::model::matrix::multiply(vector3 v)
{
	vector3 result;
	result.x = v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2];
	result.y = v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2];
	result.z = v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2];

	return result;
}
