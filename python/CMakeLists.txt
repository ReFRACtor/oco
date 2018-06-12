# Package the the module lives under
set(PYTHON_PACKAGE_NAME "oco")

# Find Python
find_package(PythonLibs 3.0 REQUIRED)
find_package(PythonInterp 3.0 REQUIRED)
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
set(MODULE_INTERFACE_FILES "oco.i")

# Use C++ mode for SWIG interface files, turn on all warnings
set_source_files_properties(${MODULE_INTERFACE_FILES} PROPERTIES CPLUSPLUS ON SWIG_FLAGS "-Wall") 

foreach(swig_file ${MODULE_INTERFACE_FILES})
    get_filename_component(module_name ${swig_file} NAME_WE)

    # Figure out which other interface files are included, and hence depended on by the module interface file
    # regenerate SWIG wrappers and recompile if those change
    execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/interface_deps.sh ${CMAKE_CURRENT_SOURCE_DIR}/${swig_file}
        OUTPUT_VARIABLE DEP_INTERFACE_FILES OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(${DEP_INTERFACE_FILES})
    # Make module recompile if any of the interface files change
    set(SWIG_MODULE_${module_name}_EXTRA_DEPS ${DEP_INTERFACE_FILES})

    if(CMAKE_VERSION VERSION_GREATER 3.7)
        swig_add_library(${module_name} TYPE SHARED LANGUAGE python SOURCES ${swig_file})
    else()
        swig_add_module(${module_name} python ${swig_file})
    endif()

    swig_link_libraries(${module_name} ${PYTHON_LIBRARIES} ${THIRDPARTY_LIBRARIES} oco)
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

# Install python source files from the python/ source directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "python/${PYTHON_PACKAGE_NAME}"
        FILES_MATCHING
        PATTERN "*.py")

# Install SWIG created files
install(DIRECTORY ${CMAKE_SWIG_OUTDIR} DESTINATION python
        FILES_MATCHING
        PATTERN "*.${CMAKE_SHARED_LIBRARY_SUFFIX}"
        PATTERN "*.so"
        PATTERN "*.py")
