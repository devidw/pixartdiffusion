cmake_minimum_required(VERSION 3.18)
project(PixArtDiffusionCPU CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Torch_DIR to your libtorch path (e.g., -DTorch_DIR=~/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(pixart_cpu main.cpp)
target_link_libraries(pixart_cpu PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(pixart_cpu PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

# On macOS, use libc++
if(APPLE)
  target_compile_options(pixart_cpu PRIVATE -stdlib=libc++)
  target_link_options(pixart_cpu PRIVATE -stdlib=libc++)
endif()


