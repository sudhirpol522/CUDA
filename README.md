# CUDA Programming Examples

A collection of CUDA C/C++ programs demonstrating GPU parallel programming concepts.

## Author
**Sudhir Pol**

## Programs

### 1. Vector Addition (`vector_add.cu`)
Basic CUDA program that performs element-wise addition of two vectors on the GPU.

### 2. Simple Vector Addition (`new.cu`)
Simplified vector addition with error checking using custom CUDA error handling macros.

### 3. Matrix Addition (`matrix_addition.cu`)
2D matrix addition using CUDA with 2D thread blocks and grids.
- Uses 2x2 thread blocks
- Demonstrates 2D indexing with `blockIdx` and `threadIdx`

### 4. Matrix Transpose (`transpose.cu`)
Transposes an 8x8 matrix using CUDA.
- Uses 16x16 thread blocks
- Demonstrates memory access patterns for matrix operations

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 13.1)
- Microsoft Visual Studio Build Tools (2019-2022)

## Compilation

### Windows
Use the provided batch script:
```bash
compile.bat
```

Or compile manually:
```bash
nvcc -allow-unsupported-compiler <filename>.cu -o <output>.exe
```

**Note:** On Windows, you need to initialize the Visual Studio environment first:
```bash
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

### Linux
```bash
nvcc <filename>.cu -o <output>
```

## Running

After compilation, simply run the executable:
```bash
.\<program_name>.exe    # Windows
./<program_name>        # Linux
```

## License
MIT License
