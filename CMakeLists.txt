cmake_minimum_required(VERSION 3.18)

project("catai" LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

### set base library
set(BASE_LIBRARY
        absl::base
        absl::check
        absl::flags
        absl::debugging
        spdlog::spdlog_header_only)

### Import llama.cpp Start
set(LLAMA_ACCELERATE OFF CACHE BOOL "" FORCE)
set(LLAMA_CLBLAST ON CACHE BOOL "" FORCE)
set(LLAMA_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp)
### Import llama.cpp End

add_executable(${PROJECT_NAME} "" src/main.cpp src/LlamaCppAgent.cpp src/LlamaCppAgent.h src/LlamaCppAgentOptions.cpp src/LlamaCppAgentOptions.h)

target_sources(${PROJECT_NAME}
        PRIVATE
        src/main.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${BASE_LIBRARY}
        llama)