cmake_minimum_required(VERSION 3.22)
project(GAT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen (Homebrew: /opt/homebrew/include/eigen3)
find_package(Eigen3 REQUIRED)

# Google Test (auto-fetch on first configure)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

# 加载 gtest_discover_tests 命令
include(GoogleTest)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
