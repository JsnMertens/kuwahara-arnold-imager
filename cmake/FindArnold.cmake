# Find Autodesk Arnold
#
# This module locates the Arnold SDK libraries and headers.
# Only Arnold_PATH / Arnold_ROOT are used for searching; no default system paths are checked.
#
# Imported targets
# ----------------
#   Arnold::Arnold        - The Arnold library target, if found
#
# Result variables
# ----------------
#   Arnold_FOUND          - True if the Arnold SDK is found
#   Arnold_INCLUDE_DIR    - Path to the Arnold include directory
#   Arnold_LIBRARY        - Path to the Arnold library
#   Arnold_VERSION        - Arnold version string 
#   Arnold_VERSION_ARCH   - Arnold version architecture
#   Arnold_VERSION_MAJOR  - Arnold version major
#   Arnold_VERSION_MINOR  - Arnold version minor
#   Arnold_VERSION_PATCH  - Arnold version patch
#

# Get hint paths from ARNOLD_PATH or Arnold_ROOT (env or cache variables)
set(_Arnold_HINTS "")
if (DEFINED Arnold_ROOT)
  list(APPEND _Arnold_HINTS "${Arnold_ROOT}")
elseif (DEFINED ENV{Arnold_ROOT})
  list(APPEND _Arnold_HINTS "$ENV{Arnold_ROOT}")
endif ()
if (DEFINED ARNOLD_PATH)
  list(APPEND _Arnold_HINTS "${ARNOLD_PATH}")
elseif (DEFINED ENV{ARNOLD_PATH})
  list(APPEND _Arnold_HINTS "$ENV{ARNOLD_PATH}")
endif ()

# Remove duplicates and empty entries from hints
list(REMOVE_DUPLICATES _Arnold_HINTS)
if (_Arnold_HINTS STREQUAL "")
  message(STATUS "ARNOLD_PATH or Arnold_ROOT are not defined, please install Arnold SDK and set those variables.")
endif ()

# Locate the Arnold include directory (ai.h).
find_path(
  Arnold_INCLUDE_DIR
  NAMES ai.h
  HINTS "${_Arnold_HINTS}"
  PATH_SUFFIXES "include/" "include/arnold/"
  NO_DEFAULT_PATH
  DOC "Directory of Arnold headers (ai.h)"
)


# Locate the Arnold library.
# ai.lib on Windows, libai.so on Linux.
find_library(
  Arnold_LIBRARY
  NAMES ai
  HINTS "${_Arnold_HINTS}"
  PATH_SUFFIXES "lib/" "bin/"
  NO_DEFAULT_PATH
  DOC "Arnold library (ai)"
)

# Extract the Arnold version from ai_version.h
set(Arnold_VERSION "")
if (Arnold_INCLUDE_DIR AND EXISTS "${Arnold_INCLUDE_DIR}/ai_version.h")
  file(
    STRINGS "${Arnold_INCLUDE_DIR}/ai_version.h" _versionLines
    REGEX "^#define[ \t]+AI_VERSION_[A-Z_]+[ \t]+(\"?[0-9]+\"?)"
  )

  foreach(_versionLine IN LISTS _versionLines)
    # Extract the ARCH version
    string(FIND "${_versionLine}" "ARCH" pos_arch)
    if(NOT pos_arch EQUAL -1)
        string(REGEX MATCH "[0-9]+" _arch "${_versionLine}")
        set(Arnold_VERSION_ARCH "${_arch}")
    endif()

    # Extract the MAJOR version
    string(FIND "${_versionLine}" "MAJOR" pos_major)
    if(NOT pos_major EQUAL -1)
        string(REGEX MATCH "[0-9]+" _major "${_versionLine}")
        set(Arnold_VERSION_MAJOR "${_major}")
    endif()

    # Extract the MINOR version
    string(FIND "${_versionLine}" "MINOR" pos_minor)
    if(NOT pos_minor EQUAL -1)
        string(REGEX MATCH "[0-9]+" _minor "${_versionLine}")
        set(Arnold_VERSION_MINOR "${_minor}")
    endif()

    # Extract the FIX version
    string(FIND "${_versionLine}" "FIX" pos_fix)
    if(NOT pos_fix EQUAL -1)
        string(REGEX MATCH "[0-9]+" _fix "${_versionLine}")
        set(Arnold_VERSION_PATCH "${_fix}")
    endif()
  endforeach()

  # Concatenate the version components
  if(Arnold_VERSION_ARCH AND Arnold_VERSION_MAJOR AND Arnold_VERSION_MINOR AND Arnold_VERSION_PATCH)
    set(Arnold_VERSION "${Arnold_VERSION_ARCH}.${Arnold_VERSION_MAJOR}.${Arnold_VERSION_MINOR}.${Arnold_VERSION_PATCH}")
    message(STATUS "Found Arnold SDK version: ${Arnold_VERSION}")
  else()
    message(WARNING "Could not determine all Arnold SDK version components from ai_version.h")
  endif()
endif ()

# Set Arnold_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Arnold
  REQUIRED_VARS Arnold_LIBRARY Arnold_INCLUDE_DIR
  VERSION_VAR Arnold_VERSION
)

# If found, create the imported target
if (Arnold_FOUND)
  # Create imported target for Arnold library
  if (NOT TARGET Arnold::Arnold)
    if (WIN32)
      find_file(
        _Arnold_DLL
        NAMES "ai.dll"
        HINTS ${_Arnold_HINTS}
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
      )
      if (_Arnold_DLL)
        # Use SHARED imported target with separate implib and runtime DLL
        add_library(Arnold::Arnold SHARED IMPORTED)
        set_target_properties(
          Arnold::Arnold PROPERTIES
            IMPORTED_IMPLIB     "${Arnold_LIBRARY}" # Import library (.lib)
            IMPORTED_LOCATION   "${_Arnold_DLL}"    # DLL location
        )
      else ()
        # If DLL not found, treat the .lib as a static library for linking
        add_library(Arnold::Arnold STATIC IMPORTED)
        set_target_properties(
          Arnold::Arnold PROPERTIES
            IMPORTED_LOCATION "${Arnold_LIBRARY}"
        )
      endif ()
    else ()  # Linux (and other UNIX)
      add_library(Arnold::Arnold SHARED IMPORTED)
      set_target_properties(
        Arnold::Arnold PROPERTIES
          IMPORTED_LOCATION "${Arnold_LIBRARY}"
      )
    endif ()

    # Set include directories
    set_target_properties(
      Arnold::Arnold PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Arnold_INCLUDE_DIR}"
    )
  endif ()
endif ()
