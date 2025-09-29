cmake_minimum_required(VERSION 3.18)
project(PixArtDiffusionVulkan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ncnn REQUIRED)

add_executable(pixart_vulkan main.cpp)
target_link_libraries(pixart_vulkan PRIVATE ncnn::ncnn)

if(APPLE)
  target_compile_options(pixart_vulkan PRIVATE -stdlib=libc++)
  target_link_options(pixart_vulkan PRIVATE -stdlib=libc++)
endif()


