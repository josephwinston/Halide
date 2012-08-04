ifneq ($(shell env command -v g++-mp-4.7),)
GXX ?= g++-mp-4.7
endif
GXX ?= g++-mp-4.6

PNGFLAGS=$(shell libpng-config --ldflags) $(shell libpng-config --cflags)
ifeq ($(HL_TARGET),ptx)
  SCHEDULE=100
  CUDA_LFLAGS=-L/usr/local/cuda/lib -lcuda
else
  SCHEDULE=0
endif

all: halide_snake

halide_snake: ../../cpp_bindings/ halide_snake.cpp
	$(MAKE) -C ../../cpp_bindings libHalide.a
	$(GXX) -O3 halide_snake.cpp -I ../../cpp_bindings/ ../../cpp_bindings/libHalide.a  -std=c++0x -lpng -o halide_snake -lpthread -ldl $(PNGFLAGS) $(CUDA_LFLAGS)

out.png: halide_snake
	./halide_snake blood_cells.png out.png

clean:
	rm halide_snake out.png
