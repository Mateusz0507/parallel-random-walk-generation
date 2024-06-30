# Parallel random walk generator

## Abstract
Parallel random walk generator is a tool for generating self avoiding random walk in 3D continuous space.

## Requirements
- CUDA Toolkit 12.2 or higher
- Visual Studio
  
## Compilation
In order to build the project, you need to set the following flags in VS:
- Configuration Properties -> CUDA C/C++ -> Common -> Generate Relocatable Device Code -> Yes (-rdc=true) 
- Configuration Properties -> CUDA C/C++ -> Device -> Code Generation -> compute_52,sm_52 
- Configuration Properties -> Linker -> Input -> Additional Dependencies -> cudadevrt.lib

These settings enable separate compilation of the .cu files.

## Usage
The program can be executed with the following parameters:
- -m/--method: Specifies the generation method. Possible values: naive (default value), normalization, genetic, genetic2.
- -N/--N: Specifies the length of the path (number of particles). Default value equals 100 (at least 3).
- -d/--directional-level: Directional coefficient for energy-based methods. Higher values result in a more directed distribution (at least 0).
- -s/--segments-number: Number of segments into which the path is divided in directional sampling (at least 1).
- --mutation-ration: Mutation probability for genetic methods (value between 0 and 1 or exactly 0 or 1).
- --generation-size: Generation size in genetic algorithms (at least 8).

### Examples
1. The following example invokes naive energetic method with directional level 2, which is divided into 5 segments.
```bash
>random-walk.exe --method=naive --directional-level=2 --segments-number=5
```
2. Execution of improved genetic method with mutation ration equal to 0.04 and generation size 100.
```bash
>random-walk.exe -m=genetic2 --mutation-ratio=0.04 --generation-size=100
```

