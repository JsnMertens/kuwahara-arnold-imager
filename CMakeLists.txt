cmake_minimum_required(VERSION 3.21)

project(ooKuwaharaArnoldImager LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)

# Force 64-bit builds on Windows
if (WIN32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Only 64-bit builds are supported on Windows.")
endif()

# ================================================
# CMake and Compiler Configuration
# ================================================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Useless on Windows

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ================================================
# Sources
# ================================================
file(GLOB SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB HEADERS CONFIGURE_DEPENDS "include/*.hpp" "include/*.h")

# ================================================
# Find dependencies
# ================================================
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
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Arnold::Arnold
)

# ================================================
# Install
# ================================================
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # for executables and DLLs on Windows
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for shared libraries on UNIX
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    # for static libraries, if any
)

install(
    FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
