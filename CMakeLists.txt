cmake_minimum_required(VERSION 3.14)
project(heart_rate_estimation)

set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

# matplotlibcppで使うPythonとNumpyを探す
find_package(Python3 COMPONENTS Development NumPy)
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(ooura-fft-cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} fftsg Python3::Python Python3::NumPy ${OpenCV_LIBS})
