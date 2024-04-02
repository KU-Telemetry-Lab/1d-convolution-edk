# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

SRCDIR = ./src
BINDIR = ./bin
OBJDIR = ./obj
INCDIR = ./inc

$(shell   mkdir -p $(BINDIR))
$(shell   mkdir -p $(OBJDIR))

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# GENCODE_FLAGS += -gencode arch=compute_37,code=sm_37
# GENCODE_FLAGS += -Wno-deprecated-gpu-targets

GENCODE_FLAGS += -gencode arch=compute_80,code=sm_80

# -forward-unknown-to-host-compiler

# added this -fPIE, when  using bundleElt
NVCC_FLAGS   :=  -std=c++17 -m64 $(GENCODE_FLAGS) --debug --generate-line-info -O2 --use_fast_math --forward-unknown-to-host-compiler -mavx2 --default-stream per-thread -fPIE
NVCC_INCLUDES := /usr/local/cuda-samples/Common
NVCC_LIBRARIES :=

$(foreach lf,$(LDFLAGS),$(eval LDFLAGS4NVCC +=  -Xcompiler \"$(lf)\"))
NVCC_LDFLAGS := $(LDFLAGS4NVCC)

SOURCES := $(wildcard $(SRCDIR)/*.cu)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

INC_DIRS := $(INCDIR) $(NVCC_INCLUDES)
# Include files add together a prefix, gcc make sense that -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Make Makefiles output Dependency files
# That -MMD and -MP flags together to generate Makefiles
# That generated Makefiles will take .o as .d to the output
# That "-MMD" and "-MP" To generate the dependency files, all you have to do is
# add some flags to the compile command (supported by both Clang and GCC):
CPP_FLAGS ?= $(INC_FLAGS) -MMD -MP

# Dependency files
# To use the .d files, just need to find them all:
#
DEPS := $(OBJECTS:.o=.d)

################################################################################

ALL_EXES := 1d_convolution_tests 1d_convolution_bundle

MAIN_OBJECTS := $(foreach exe, $(ALL_EXES), $(OBJDIR)/$(exe).o)
NOT_MAIN_OBJECTS := $(filter-out $(MAIN_OBJECTS), $(OBJECTS))

# Target rules

.PHONY: all clean
all: $(ALL_EXES)

%.ptx: %.cu
	$(NVCC)  $(CPP_FLAGS) $(NVCC_FLAGS) -o $@ -ptx -src-in-ptx -c $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC)  $(CPP_FLAGS) $(NVCC_FLAGS) -o $@ -c $<

1d_convolution_tests: $(OBJDIR)/1d_convolution_tests.o $(NOT_MAIN_OBJECTS)
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $(BINDIR)/$@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

1d_convolution_bundle: $(OBJDIR)/1d_convolution_bundle.o $(NOT_MAIN_OBJECTS)
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $(BINDIR)/$@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

clean:
	rm -f $(OBJECTS)
	rm -f $(addprefix $(BINDIR)/, $(ALL_EXES))
