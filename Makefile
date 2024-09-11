# run `bear -- make -B -j` to get compile_commands.json

r123ver = 1.14.0
r123inc = random123-$(r123ver)/include

OPT = -O3 -funroll-loops -fopenmp-simd -ffast-math
# OPT = -g -Og
CFLAGS = $(OPT) -Wall -Wextra -Wimplicit-fallthrough
CXXFLAGS = $(OPT) -I$(r123inc)
CSRC = conn.c ode.c matmul.c ops.c # csr8.c
CXXSRC = r123.cpp
OBJ = $(patsubst %.c,%.o,$(CSRC)) $(patsubst %.cpp,%.o,$(CXXSRC))
SO = libtvbk.so
CC = gcc
CXX = g++
philox = $(r123inc)/Random123/philox.h

all: $(SO) tvbk.py test

$(SO) : $(philox) $(OBJ)
	$(CXX) -shared $(OBJ) -o $@

test: test.c tvbk.h $(OBJ)
	$(CC) $(CFLAGS) -c test.c
	$(CXX) -static test.o $(OBJ) -o $@ -lm

tvbk.py: tvbk.h	
	ctypesgen -l libtvbk.so tvbk.h > tvbk.py

tvbk.wasm: em.cpp tvbk.h $(SRC)
	emcc -lembind -o tvbk.js em.cpp $(SRC)

clean:
	# except wheels
	git clean -xf -e '*.whl'
	
verify_so_path:
	python3 -c "import tvb_kernels as tk; print( tk.tvbk._libs['libtvbk.so'].access['cdecl']._name)" | grep python.*packages

wheel: tvbk.py
	pip wheel .
	# python3 -m wheel tags --platform-tag manylinux2010 tvb_kernels-*-py3-none-any.whl
	# rm tvb_kernels-*-py3-none-any.whl

tag:
	git tag $(shell python3 -c "import tvb_kernels; print(tvb_kernels.__version__)")

cleantest:
	make clean
	make -B -j
	./test
	make wheel
	pip uninstall -y tvb_kernels || true
	pip install tvb_kernels-*.whl
	make clean
	make verify_so_path
	pytest tests.py

philox: $(r123inc)/Random123/philox.h

$(r123inc)/Random123/philox.h:
	curl -LO https://github.com/DEShawResearch/random123/archive/refs/tags/v$(r123ver).tar.gz
	tar xzf v$(r123ver).tar.gz
	rm v$(r123ver).tar.gz
