#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cstdlib";
#include "iostream"

bool cuda_check_error(cudaError_t error, const char* file, const int line, bool terminate);

#define cuda_check(action, kill) cuda_check_error(action, __FILE__, __LINE__, kill)
#define cuda_check_continue(action) cuda_check(action, false)
#define cuda_check_terminate(action) cuda_check(action, true)
