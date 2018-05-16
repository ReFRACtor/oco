set(TEMPLATE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${TEMPLATE_LIB_DIR})

# Add each source file to LIB_SOURCES
set(LIB_SOURCES
    # file1.cc
    # file2.cc
)

# Create shared library
add_library(template SHARED ${LIB_SOURCES})
target_link_libraries(template ${THIRDPARTY_LIBRARIES})

# Expose lib directory location to top level scope
set(TEMPLATE_LIB_DIR ${TEMPLATE_LIB_DIR} PARENT_SCOPE)

# Add header files and library to install target
file(GLOB TEMPLATE_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
install(FILES ${TEMPLATE_INCLUDE_FILES} DESTINATION include)

install(TARGETS template LIBRARY DESTINATION lib)
