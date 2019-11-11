# This must be called before project() since it sets this variable as empty into the cache
set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build to create, Release (optimized) by default")

project("oco")

cmake_minimum_required(VERSION 3.2)

# Version number
set(OCO_VERSION "1.0.0")

# Global setting: Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Conda has a separate set of debug flags defined, which is more than just
# adding a -g. If we are doing a debug rather than release build, use those
# flags if we find them.
set(DEBUG_CXXFLAGS "$ENV{DEBUG_CXXFLAGS}" CACHE STRING "Flags to use in Debug")
set(DEBUG_CFLAGS "$ENV{DEBUG_CFLAGS}" CACHE STRING "Flags to use in Debug")
set(DEBUG_FFLAGS "$ENV{DEBUG_FFLAGS}" CACHE STRING "Flags to use in Debug")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(DEBUG_CXXFLAGS)
    set(CMAKE_CXX_FLAGS "${DEBUG_CXXFLAGS}")
  endif(DEBUG_CXXFLAGS)
  if(DEBUG_CFLAGS)
    set(CMAKE_C_FLAGS "${DEBUG_CFLAGS}")
  endif(DEBUG_CFLAGS)
  if(DEBUG_FFLAGS)
    # Suppress warning about comparing reals. This is often a bad thing
    # to do, but the fortran code we have does this the "right" way, e.g.
    # checking for identical to zero as a flag value.
    set(CMAKE_Fortran_FLAGS "${DEBUG_FFLAGS}  -Wno-compare-reals")
  endif(DEBUG_FFLAGS)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# An annoying thing Anaconda does on the mac is add
# -Wl,-dead_strip_dylibs. Not really clear why this would be imposed on all
# environments, but it is at least currently done. This causes
# "lazy symbol bindings failed" errors, in particular libgsl does lazy
# bindings iwht libgslcblas. It is common to include libraries on the link
# line for reasons other than directly providing symbols, so we really want
# the CMake rule to determine what libraries to include rather than having
# the linker toss libraries out it "thinks" it doesn't need. We strip
# the option out if it gets set by the passed in LDFLAGS from the Anaconda
# environment.
if(CMAKE_EXE_LINKER_FLAGS)
    string(REPLACE "-Wl,-dead_strip_dylibs" "" CMAKE_EXE_LINKER_FLAGS
           ${CMAKE_EXE_LINKER_FLAGS})
endif()

if(CMAKE_MODULE_LINKER_FLAGS)
    string(REPLACE "-Wl,-dead_strip_dylibs" "" CMAKE_MODULE_LINKER_FLAGS
           ${CMAKE_MODULE_LINKER_FLAGS})
endif()

if(CMAKE_SHARED_LINKER_FLAGS)
    string(REPLACE "-Wl,-dead_strip_dylibs" "" CMAKE_SHARED_LINKER_FLAGS
           ${CMAKE_SHARED_LINKER_FLAGS})
endif()

# By default install into the build directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
    message(STATUS "Setting install prefix to: " ${CMAKE_INSTALL_PREFIX})
    message(STATUS "Specify -DCMAKE_INSTALL_PREFIX to cmake to change")
else()
    message(STATUS "Using install prefix: " ${CMAKE_INSTALL_PREFIX})
endif()    

set(THIRDPARTY_LIBRARIES "")
set(THIRDPARTY_INCLUDES "")

# Find ReFRACTOR
find_package(Refractor
             REQUIRED COMPONENTS python
             CONFIG
             HINTS ${REFRACTOR_DIR} ${CONDA_PREFIX})
list(APPEND THIRDPARTY_INCLUDES ${REFRACTOR_INCLUDE_DIRS})
list(APPEND THIRDPARTY_LIBRARIES ${REFRACTOR_LIBRARIES})

# Insert the ReFRACtor cmake modules directory at the beginning of
# our search path so we can override built in modules with updated
# copies and to enable the finding of additional packages
if(DEFINED REFRACTOR_DIR)
  list(INSERT CMAKE_MODULE_PATH 0 "${REFRACTOR_DIR}/cmake/Modules/")
else()
  # If not otherwise pointed to a different directory, use CONDA_PREFIX
  # if defined
  if(DEFINED ENV{CONDA_PREFIX})
    list(INSERT CMAKE_MODULE_PATH 0 "$ENV{CONDA_PREFIX}/cmake/Modules/")
  endif(DEFINED ENV{CONDA_PREFIX})
endif(DEFINED REFRACTOR_DIR)

# Find Blitz++
find_package(Blitz REQUIRED)
list(APPEND THIRDPARTY_INCLUDES ${BLITZ_INCLUDE_DIR})
list(APPEND THIRDPARTY_LIBRARIES ${BLITZ_LIBRARIES})

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX)
# Different versions of FindHDF5 may use one or the other of these variable names, with
# newer versions saying that HDF5_C_INCLUDE_DIRS deprecates the other
if(HDF5_C_INCLUDE_DIRS)
    list(APPEND THIRDPARTY_INCLUDES ${HDF5_C_INCLUDE_DIRS})
else()
    list(APPEND THIRDPARTY_INCLUDES ${HDF5_C_INCLUDE_DIR})
endif()
list(APPEND THIRDPARTY_LIBRARIES ${HDF5_LIBRARIES})
include_directories(${THIRDPARTY_INCLUDES})

# Find Boost
# We want to use the Conda version if it is available, and the user hasn't
# otherwise specified a different version
if(NOT BOOST_ROOT AND DEFINED ENV{CONDA_PREFIX})
  set(BOOST_ROOT $ENV{CONDA_PREFIX})
endif(NOT BOOST_ROOT AND DEFINED ENV{CONDA_PREFIX})
find_package(Boost REQUIRED COMPONENTS system)
list(APPEND THIRDPARTY_INCLUDES ${Boost_INCLUDE_DIRS})
list(APPEND THIRDPARTY_LIBRARIES ${Boost_LIBRARIES})

# When we are using debug mode, turn on the blitz range checking.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DBZ_DEBUG")

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
endif()

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# For some reason on the Mac we don't automatically figure out that CONDA
# should be on the rpath. So add that. This seems like a bug, the reading of
# CMAKE_INSTALL_RPATH_USE_LINK_PATH seems to indicate it should do this 
# automatically. But it doesn't, so go ahead and set this
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
endif(DEFINED ENV{CONDA_PREFIX})

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Expose options that user should supply
set(ABSCO_DIR $ENV{abscodir} CACHE PATH "Base path of Absoprtion Coefficient tables are located")
if(NOT ABSCO_DIR)
    message(WARNING "ABSCO_DIR option not supplied, tests will fail")
endif()

include_directories(${THIRDPARTY_INCLUDES})

add_subdirectory(lib)

add_subdirectory(python)

enable_testing()
add_subdirectory(test)

# Configure setup shell script to have appropriate paths and install
configure_file (
    "${PROJECT_SOURCE_DIR}/script/setup_oco_env.sh.in"
    "${PROJECT_BINARY_DIR}/setup_oco_env.sh"
)
install(FILES "${PROJECT_BINARY_DIR}/setup_oco_env.sh" DESTINATION "${CMAKE_INSTALL_PREFIX}")
