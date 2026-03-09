# C++ and Python Embedding Demo

Small example that embeds Python in a C++ program and calls functions from a Python module.

## Files
- `main.cpp`: C++ entry point that initializes Python, imports `embedded_module`, and calls its functions.
- `embedded_module.py`: Python helpers used by the C++ code.

## Build

### With CMake
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
Run with:
```sh
./build/cpp_python_demo
```

### Without CMake
Use your system compiler and `python3-config` to pick up the right headers and linker flags:

```sh
g++ main.cpp -o cpp_python_demo $(python3-config --includes) $(python3-config --ldflags)
```

If you prefer clang, replace `g++` with `clang++`. You need Python development headers installed (e.g., `python3-dev`).

## Run
Execute the compiled binary; it will run the Python code and print results:

```sh
./cpp_python_demo
```
