set(OCO_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${OCO_LIB_DIR})

# Add each source file to LIB_SOURCES
file(GLOB LIB_SOURCES "*.cc")

# Create shared library
add_library(oco SHARED ${LIB_SOURCES})
target_link_libraries(oco ${THIRDPARTY_LIBRARIES})

# Expose lib directory location to top level scope
set(OCO_LIB_DIR ${OCO_LIB_DIR} PARENT_SCOPE)

# Add header files and library to install target
file(GLOB OCO_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
install(FILES ${OCO_INCLUDE_FILES} DESTINATION include)

install(TARGETS oco LIBRARY DESTINATION lib)
