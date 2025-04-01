cmake_minimum_required(VERSION 3.21)

project(ooKuwaharaArnoldImager VERSION 1.1.0 LANGUAGES CXX)

# Force 64-bit builds on Windows
if (WIN32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Only 64-bit builds are supported on Windows.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vfx-cmake-modules/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vfx-cmake-modules/cmake/modules")

include(GNUInstallDirs)

# ================================================
# CMake and Compiler Configuration
# ================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Useless on Windows

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # Don't work with MSVC

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/W4 /wd4100)  # Warning level 4 and disable unused parameter warning
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ================================================
# Sources
# ================================================
file(GLOB SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB HEADERS CONFIGURE_DEPENDS "include/*.hpp" "include/*.h")

# ================================================
# Find dependencies
# ================================================
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Arnold REQUIRED)

# ================================================
# Create shared library target
# ================================================
add_library(${PROJECT_NAME} SHARED
    ${SOURCES} ${HEADERS}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
    
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Arnold::Arnold
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        OPENCV_STATIC
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/O2 /arch:AVX>)  # Enable AVX instructions
endif()

# ================================================
# Generate artifacts for Arnold and DCCs
# ================================================
# Generate the module description file for Maya
set(MAYA_MODULE_BINARY_DIR  "${CMAKE_CURRENT_BINARY_DIR}/MayaModule")
set(MAYA_MODULE_NAME        "${PROJECT_NAME}.mod")
set(MAYA_AE_TEMPLATES_DIR   "scripts/mtoa/ui/ae")

include(GenerateMayaModuleFile)
generate_maya_module_file(
    MODULE_NAME ${PROJECT_NAME}
    MODULE_VERSION ${PROJECT_VERSION}
    DST_DIRPATH ${MAYA_MODULE_BINARY_DIR}
    ARNOLD_PLUGIN_PATH ${CMAKE_INSTALL_BINDIR}
    MTOA_TEMPLATES_PATH ${MAYA_AE_TEMPLATES_DIR}
)

# ================================================
# Install
# ================================================
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # for executables and DLLs on Windows
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  # for shared libraries on UNIX
)

# Install MTD file, rename it to ensure it has the same name as the library
# This is important for Arnold to find the plugin
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/mtd/ooKuwaharaArnoldImager.mtd
    DESTINATION ${CMAKE_INSTALL_BINDIR}/
    RENAME ${PROJECT_NAME}.mtd
)

# Install the AE templates for Maya
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/maya/ae/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${MAYA_AE_TEMPLATES_DIR}"
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.mel"
)

# Install the module description file for Maya
install(
    FILES ${MAYA_MODULE_BINARY_DIR}/${MAYA_MODULE_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/
)
