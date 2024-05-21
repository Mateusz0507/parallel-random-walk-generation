#pragma once

#include <set>

class cuda_memory_releaser
{
public:
	cuda_memory_releaser& get();

	bool log(void* dev_ptr);
	bool unregister(void* dev_ptr);

private:
	static cuda_memory_releaser releaser;

	std::set<void*> cuda_pointers;

	cuda_memory_releaser();
	~cuda_memory_releaser();

	
};
