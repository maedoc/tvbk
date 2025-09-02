#!/bin/bash

cmake -S . -B buildlib
cmake --build buildlib
ctest --test-dir buildlib
cp buildlib/libuops.so uops_py/
