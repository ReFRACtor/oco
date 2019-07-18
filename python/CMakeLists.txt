# Package the the module lives under
set(PYTHON_PACKAGE_NAME "oco")

# New policy is to pass SWIG_MODULE_NAME as -module. We don't actually need
# this, but no reason not to turn it on. Without this, you get a warning
# message from CMaker version 3.14 and later
if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif()
if(POLICY CMP0078)
  cmake_policy(SET CMP0078 OLD)
endif()

# Find Python. Note for conda on the Mac you need to find the interpreter
# before the libs (i.e., the order is important). Otherwise this finds
# the system python for libraries but uses conda for the interpreter, which
# of course causes problems. This goes away if you do this in the other order.
find_package(PythonInterp 3.0 REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

# Find NumPy
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

# Find SWIG
find_package(SWIG 3.0 REQUIRED)
include(${SWIG_USE_FILE})
set(CMAKE_SWIG_FLAGS "")

# Output SWIG files to a directory named for the package
set(PACKAGE_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_PACKAGE_NAME})
set(CMAKE_SWIG_OUTDIR ${PACKAGE_OUTDIR})

# Add the C++ library directory to the list of includes
include_directories(${THIRDPARTY_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/lib)

# Which interface files should be used for creating modules
set(MODULE_INTERFACE_FILES 
    hdf_sounding_id.i
    level_1b_hdf.i
    level_1b_oco.i
    oco_met_file.i
    oco_noise_model.i
    oco_sounding_id.i
)

# Use C++ mode for SWIG interface files, turn on all warnings
set_source_files_properties(${MODULE_INTERFACE_FILES} PROPERTIES CPLUSPLUS ON SWIG_FLAGS "-Wall") 

foreach(swig_file ${MODULE_INTERFACE_FILES})
    get_filename_component(module_name ${swig_file} NAME_WE)

    # Figure out which other interface files are included, and hence depended on by the module interface file
    # regenerate SWIG wrappers and recompile if those change
    execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/interface_deps.sh ${CMAKE_CURRENT_SOURCE_DIR}/${swig_file}
        OUTPUT_VARIABLE DEP_INTERFACE_FILES OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE INTERFACE_DEP_RES ERROR_VARIABLE INTERFACE_DEP_ERR)

    if(${INTERFACE_DEP_RES})
        message(FATAL_ERROR "Error running interface_deps.py:\n${INTERFACE_DEP_ERR}\n${DEP_INTERFACE_FILES}")
    endif()

    # Make module recompile if any of the interface files change
    set(SWIG_MODULE_${module_name}_EXTRA_DEPS ${DEP_INTERFACE_FILES})

    if(CMAKE_VERSION VERSION_GREATER 3.7)
        swig_add_library(${module_name} TYPE SHARED LANGUAGE python SOURCES ${swig_file})
    else()
        swig_add_module(${module_name} python ${swig_file})
    endif()
    if(APPLE)
      # Compiling with python causes a problem on the Mac using conda.
      # See https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
      # The symbols get resolved when loaded into python, so this doesn't cause
      # a problem.
      swig_link_libraries(${module_name} ${THIRDPARTY_LIBRARIES} oco)
      set_target_properties(${SWIG_MODULE_${module_name}_REAL_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    else()
      swig_link_libraries(${module_name} ${PYTHON_LIBRARIES} ${THIRDPARTY_LIBRARIES} oco)
    endif(APPLE)
    target_compile_definitions(${SWIG_MODULE_${module_name}_REAL_NAME} PRIVATE SWIG_TYPE_TABLE=refractor)

    LIST(APPEND SWIG_LIBS ${SWIG_MODULE_${module_name}_REAL_NAME})
endforeach()

set_target_properties(${SWIG_LIBS} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PACKAGE_OUTDIR})

# Ensure module has .so extension (required by python) on all platforms including OS X (where dylib would otherwise be the case)
set_target_properties(${SWIG_LIBS} PROPERTIES SUFFIX ".so")

# Make a custom target to allow doing: make python
add_custom_target(python DEPENDS ${SWIG_LIBS})

# Add python files to install target
file(GLOB INSTALL_INTERFACE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.i)
install(FILES ${INSTALL_INTERFACE_FILES} DESTINATION include)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib(True, False, ''))" OUTPUT_VARIABLE PYTHON_INSTALL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# Install python source files from the python/ source directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "${PYTHON_INSTALL_DIR}/${PYTHON_PACKAGE_NAME}"
        FILES_MATCHING
        PATTERN "*.py")

# Install SWIG created files
install(DIRECTORY ${CMAKE_SWIG_OUTDIR} DESTINATION ${PYTHON_INSTALL_DIR}
        FILES_MATCHING
        PATTERN "*.${CMAKE_SHARED_LIBRARY_SUFFIX}"
        PATTERN "*.so"
        PATTERN "*.py")

# Expose Python install path to parent scope
set(PYTHON_INSTALL_DIR ${PYTHON_INSTALL_DIR} PARENT_SCOPE)
      
