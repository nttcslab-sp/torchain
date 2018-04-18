# ==========================
#  Compiler option settings
# ==========================

THC_INCLUDE=$(shell python -c "from torch.utils.ffi import _setup_wrapper;[print('-I '+p, end=' ') for p in _setup_wrapper(with_cuda=True)[1]]")
TH_INCLUDE=$(shell python -c "from torch.utils.ffi import _setup_wrapper;[print('-I '+p, end=' ') for p in _setup_wrapper(with_cuda=False)[1]]")
# TODO: extract this options like TH_INCLUDE


CXX_DEBUG_FLAGS		=	-g3 -O0 -DDEBUG -coverage
CXX_RELEASE_FLAGS	=	-s -O3 -DNDEBUG
CUDA_DEBUG_FLAGS	=	-O3 -DDEBUG
CUDA_RELEASE_FLAGS	=   -O3 -DNDEBUG

KALDI_ROOT := kaldi
FST_ROOT := $(KALDI_ROOT)/tools/openfst
KALDI_OPT=  -isystem $(KALDI_ROOT)/src -L$(KALDI_ROOT)/src/matrix -lkaldi-matrix -L$(KALDI_ROOT)/src/lib -lkaldi-base -isystem $(FST_ROOT)/include -L$(FST_ROOT)/lib

MY_LIB_LIBS := libmy_lib.a
USE_CUDA := True
CXX_OPT=-pthread -Wsign-compare -fwrapv -Wall -fPIC -std=c++11 -fopenmp $(KALDI_OPT)
CUDA_OPT=-std=c++11 --default-stream per-thread --expt-extended-lambda --expt-relaxed-constexpr $(KALDI_OPT)


ifeq ($(USE_CUDA),True) # 'nvcc' found
	CXX_OPT += -DWITH_CUDA
	MY_LIB_LIBS += libmy_lib_cuda.a
	KALDI_OPT += -L$(KALDI_ROOT)/src/cudamatrix -lkaldi-cudamatrix -L$(KALDI_ROOT)/src/chain -lkaldi-chain
endif


# ==============================
#  Source files and build rules
# ==============================

objects := $(wildcard src/*.cpp)
objects := $(objects:.cpp=.o)

cuda_objects := $(wildcard src/*.cu)
cuda_objects := $(cuda_objects:.cu=.o)

headers := $(wildcard src/*.hpp)

src/%.o: src/%.cpp $(headers)
	g++ -c $< -o $@ $(CXX_OPT) $(TH_INCLUDE)

src/%.o: src/%.cu $(headers)
	nvcc -c $< -o $@ -Xcompiler "$(CXX_OPT)" $(THC_INCLUDE) -I /usr/local/cuda/samples/common/inc $(CUDA_OPT)

libmy_lib.a: $(objects)
	ar rcs $@ $^

libmy_lib_cuda.a: $(cuda_objects)
	ar rcs $@ $^

# NOTE: about ar rcs: https://stackoverflow.com/questions/29714300/what-does-the-rcs-option-in-ar-do


kaldi:
	git clone --depth 1 https://github.com/kaldi-asr/kaldi.git

kaldi/src/cudamatrix/libkaldi-cudamatrix.so: kaldi
	cd kaldi/tools && make
	cd kaldi && ./configure # TODO add nice configs
	cd kaldi/src && make cudamatrix/libkaldi-cudamatrix.so

# ==========
#  Commands
# ==========


.PHONY: all clean test-gpu test-cpu install release debug test

all: release

# $(KALDI_ROOT)/src/cudamatrix/libkaldi-cudamatrix.so $(KALDI_ROOT)/src/matrix/libkaldi-matrix.so
test: LD_LIBRARY_PATH := $(KALDI_ROOT)/src/cudamatrix:$(KALDI_ROOT)/src/matrix:$(KALDI_ROOT)/src/chain:$(LD_LIBRARY_PATH)
test: debug
	PYTHONPATH=$(PWD):$(PYTHONPATH) python test/test.py

gdb: LD_LIBRARY_PATH := $(KALDI_ROOT)/src/cudamatrix:$(KALDI_ROOT)/src/matrix:$(LD_LIBRARY_PATH)
gdb: debug
	PYTHONPATH=$(PWD):$(PYTHONPATH) gdb -ex r --args python test/test.py

release: CXX_OPT+=$(CXX_RELEASE_FLAGS)
release: CUDA_OPT+=$(CUDA_RELEASE_FLAGS)
release: $(MY_LIB_LIBS)
	python build.py

debug: CXX_OPT+=$(CXX_DEBUG_FLAGS)
debug: CUDA_OPT+=$(CUDA_DEBUG_FLAGS)
debug: $(MY_LIB_LIBS)
	python build.py

clean:
	rm -fv $(MY_LIB_LIBS) $(objects) $(cuda_objects)
	rm -rfv my_lib/_ext htmlcov .coverage
	rm -rfv build *.egg-info *.so .eggs dist
	rm -rfv core.*
	find . -name .cache | xargs rm -rfv
	find . -name "*.gcno" | xargs rm -rfv
	find . -name "*.gcda" | xargs rm -rfv
	find . -name "*.gcov" | xargs rm -rfv
	rm -rfv html-gcov
	rm -fv gcov.info

test-gpu: debug
	CUDA_VISIBLE_DEVICES=0,1 py.test --cov=my_lib --cov-report=term --cov-report=html test

test-cpu: debug
	CUDA_VISIBLE_DEVICES="" python build.py
	CUDA_VISIBLE_DEVICES="" py.test --cov=my_lib --cov-report=term --cov-report=html test

install: clean release
	python setup.py install

gcov-cpu: test-cpu
	gcov -rbf -s src src/*.cpp

gcov-gpu: test-gpu
	gcov -rbf -s src src/*.cpp src/*.cu
	lcov -c --no-external -d . -o gcov.info
	genhtml -o html-gcov gcov.info --ignore-errors source
