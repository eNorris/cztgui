#-------------------------------------------------
#
# Project created by QtCreator 2015-02-20T08:53:15
#
#-------------------------------------------------

QT       += core gui

# Manually add printsupport to QCustomPlot packages
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = cuda1
TEMPLATE = app

SOURCES += main.cpp \
    MainWindow.cpp \
    qcustomplot.cpp \
    SimEngine.cpp \
    ClickablePlot.cpp \
    VerticalAutoslider.cpp \
    cudaengine.cpp \
    cudakernel.cu \
    cudalink.cu

SOURCES -= cudakernel.cu \
           cudalink.cu

HEADERS  += MainWindow.h \
    qcustomplot.h \
    utils.h \
    SimEngine.h \
    ClickablePlot.h \
    VerticalAutoslider.h \
    cudaengine.h \
    cudakernel.h \
    cudalink.h

HEADERS -= cudalink.h \
    cudakernel.h

FORMS    += ui_MainWindow.ui

# From: https://cudaspace.wordpress.com/2012/07/05/qt-creator-cuda-linux-review/

# project build directories
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3  # C++ flags

# Cuda sources
CUDA_SOURCES += cudakernel.cu \
                cudalink.cu

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda-7.0
CUDA_SDK =
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bits Operating system
# libs used in your code
LIBS += -lcudart -lcuda
# GPU architecture
CUDA_ARCH     = sm_35                # Yeah! I've a new device. Adjust with your compute capability
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

