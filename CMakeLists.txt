CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(THNETS)
SET(THNETS_VERSION_MAJOR 0)
SET(THNETS_VERSION_MINOR 1)

SET(ASM_OPTIONS "-x assembler-with-cpp")
SET(CMAKE_ASM_FLAGS "${CFLAGS} ${ASM_OPTIONS}" )

ENABLE_LANGUAGE(C ASM)

OPTION(CUDNN_HALF "CUDNN_HALF" ON)
OPTION(THNETS_EXAMPLE "THNETS_EXAMPLE" ON)
OPTION(MEMORYDEBUG "MEMORYDEBUG" OFF)
OPTION(GDBDEBUG "GDBDEBUG" OFF)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -fPIC -L/usr/local/cuda/lib64")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -L/usr/local/cuda/lib64")

IF(OPTIMIZE)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF()

IF(MEMORYDEBUG)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMEMORYDEBUG")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMEMORYDEBUG")
ENDIF()

IF(GDBDEBUG)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb -O0")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
ENDIF()


ADD_LIBRARY(THNets_CPU SHARED
	images.c
	memory.c
	thapi.c
	thbasic.c
	thload.c
)

ADD_LIBRARY(THNets_OpenBLAS_BASE SHARED
	OpenBLAS-stripped/copy.c
	OpenBLAS-stripped/gemm_beta.c
	OpenBLAS-stripped/gemv_t.c
	OpenBLAS-stripped/sgemm.c
	OpenBLAS-stripped/sgemv.c
	OpenBLAS-stripped/sger.c
)

ADD_LIBRARY(THNets_Modules
	modules/Dropout.c
	modules/Linear.c
	modules/Normalize.c
	modules/Reshape.c
	modules/SoftMax.c
	modules/SpatialBatchNormalization.c
	modules/SpatialConvolution.c
	modules/SpatialConvolutionMM.c
	modules/SpatialFullConvolution.c
	modules/SpatialMaxPooling.c
	modules/SpatialMaxUnpooling.c
	modules/SpatialZeroPadding.c
	modules/Threshold.c
	modules/View.c
)

TARGET_LINK_LIBRARIES(THNets_CPU THNets_Modules)

FIND_PACKAGE(OpenCL)
IF(OpenCL_FOUND)
	INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})
	LINK_DIRECTORIES(${OpenCL_LIBRARY})

	ADD_LIBRARY(THNets_OpenCL
		opencl/opencl_basic.c
		opencl/opencl_img.c
		opencl/opencl_SoftMax.c
		opencl/opencl_SpatialConvolution.c
		opencl/opencl_SpatialMaxPooling.c
		opencl/opencl_Threshold.c
	)

	TARGET_LINK_LIBRARIES(THNets_CPU THNets_OpenCL)
ENDIF()

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
	INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
	LINK_DIRECTORIES(${CUDA_LIBRARIES})

	#Compile the main library
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; 
		-Xcompiler -fPIC 
		-I/usr/local/cuda/include 
		-DCUDNN
		-L/usr/local/cuda/lib64
		-lcudart
		-lcudnn
		-lcublas
	)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCUDNN")
	IF(OPTIMIZE)
		SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
	ENDIF()
	IF(CUDNN_HALF)
		SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -DCUDNN_HALF -DHAVEHALF)
	ENDIF()
	CUDA_ADD_LIBRARY(THNets_CUDA 
			cudnn/cudnn_basic.c
			cudnn/cudnn_copy.cu
			cudnn/cudnn_SoftMax.c
			cudnn/cudnn_SpatialBatchNormalization.c
			cudnn/cudnn_SpatialConvolution.c
			cudnn/cudnn_SpatialMaxPooling.c
			cudnn/cudnn_th.h
			cudnn/cudnn_Threshold.c
			cudnn/cunn_SpatialFullConvolution.cu
			cudnn/cunn_SpatialMaxPooling.cu
			cudnn/cunn_SpatialMaxUnpooling.cu
	)
	TARGET_LINK_LIBRARIES(THNets_CPU THNets_CUDA)
ENDIF()

IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DX86_64")
	SET(CAN_USE_ASSEMBLER TRUE)
	SET_SOURCE_FILES_PROPERTIES(OpenBLAS-stripped/x86_64/axpy_sse.S PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
	SET_SOURCE_FILES_PROPERTIES(OpenBLAS-stripped/x86_64/gemm_kernel_8x4_penryn.S PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
	SET_SOURCE_FILES_PROPERTIES(OpenBLAS-stripped/x86_64/gemm_ncopy_4.S PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
	SET_SOURCE_FILES_PROPERTIES(OpenBLAS-stripped/x86_64/gemm_tcopy_4.S PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
	TARGET_SOURCES(THNets_OpenBLAS_BASE PRIVATE
		"OpenBLAS-stripped/x86_64/axpy_sse.S"
		"OpenBLAS-stripped/x86_64/gemm_kernel_8x4_penryn.S"
		"OpenBLAS-stripped/x86_64/gemm_ncopy_4.S"
		"OpenBLAS-stripped/x86_64/gemm_ncopy_8.c"
		"OpenBLAS-stripped/x86_64/gemm_tcopy_4.S"
		"OpenBLAS-stripped/x86_64/gemm_tcopy_8.c"
	)
ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)

TARGET_LINK_LIBRARIES(THNets_CPU THNets_OpenBLAS_BASE openblas cuda rt cudnn cublas)

CUDA_ADD_EXECUTABLE(THNets_test test.c images.c)

TARGET_COMPILE_OPTIONS(THNets_test PRIVATE "-L/usr/local/cuda/lib64")

TARGET_LINK_LIBRARIES(THNets_test THNets_CPU jpeg png)# cuda cudnn rt cublas)
