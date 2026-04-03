set(CMAKE_SYSTEM_NAME iOS)

# specify the cross compiler
set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)

# specify which architectures to build for
set(CMAKE_OSX_ARCHITECTURES "$(ARCHS_STANDARD)")

# you can also choose to build for a specific device
# set(CMAKE_OSX_ARCHITECTURES "arm64")
# or for the simulator
# set(CMAKE_OSX_ARCHITECTURES "x86_64")

set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphoneos;-iphonesimulator")

# you might also want to set the deployment target
# set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "10.0")
