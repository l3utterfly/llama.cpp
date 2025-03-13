find_package (Threads REQUIRED)
find_program(GLSLC_EXECUTABLE glslc)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found.")
endif()

add_executable(vulkan-shaders-gen vulkan-shaders-gen.cpp)
target_compile_features(vulkan-shaders-gen PRIVATE cxx_std_17)
target_link_libraries(vulkan-shaders-gen PUBLIC Threads::Threads)
