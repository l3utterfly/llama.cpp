# Normal build for host
find_package(Threads REQUIRED)
find_package(Vulkan COMPONENTS glslc REQUIRED)

add_executable(vulkan-shaders-gen vulkan-shaders-gen.cpp)
target_compile_features(vulkan-shaders-gen PRIVATE cxx_std_17)
target_link_libraries(vulkan-shaders-gen PUBLIC Threads::Threads)
target_link_libraries(vulkan-shaders-gen PRIVATE Vulkan::Vulkan)
