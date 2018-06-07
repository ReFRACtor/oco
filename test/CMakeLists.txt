set(OCO_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${OCO_TEST_DIR} ${OCO_LIB_DIR})

# Add each test file to TEST_SOURCES
file(GLOB TEST_SOURCES "*.cc")

# Create test case fixtures default properties file from CMake variables
configure_file (
    global_fixture_default.cc.in 
    "${CMAKE_CURRENT_BINARY_DIR}/global_fixture_default.cc"
)

list(APPEND TEST_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/global_fixture_default.cc")

# Create test executable and link to main library target
add_executable(oco_test ${TEST_SOURCES})
target_link_libraries(oco_test oco ${THIRDPARTY_LIBRARIES})

# Add test to ctest
add_test(NAME oco COMMAND oco_test --log_level=test_suite)

# Expose test directory location to top level scope
set(OCO_TEST_DIR ${OCO_TEST_DIR} PARENT_SCOPE)
