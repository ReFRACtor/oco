# This must be called before project() since it sets this variable as empty into the cache
set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build to create, Release (optimized) by default")

project("oco")

cmake_minimum_required(VERSION 3.2)

# Version number
set(OCO_VERSION "1.0.0")

# Global setting: Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
             HINTS ${REFRACTOR_DIR})
list(APPEND THIRDPARTY_INCLUDES ${REFRACTOR_INCLUDE_DIRS})
list(APPEND THIRDPARTY_LIBRARIES ${REFRACTOR_LIBRARIES})
list(APPEND THIRDPARTY_LIBRARIES ${REFRACTOR_PYTHON_LIBRARIES})

# Insert the ReFRACtor cmake modules directory at the beginning of
# our search path so we can override built in modules with updated
# copies and to enable the finding of additional packages
list(INSERT CMAKE_MODULE_PATH 0 "${REFRACTOR_DIR}/cmake/Modules/")

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
find_package(Boost REQUIRED COMPONENTS system unit_test_framework)
list(APPEND THIRDPARTY_INCLUDES ${Boost_INCLUDE_DIRS})
list(APPEND THIRDPARTY_LIBRARIES ${Boost_LIBRARIES})

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
