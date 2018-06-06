set(OCO_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${OCO_TEST_DIR} ${OCO_LIB_DIR})

# Add each test file to TEST_SOURCES
set(TEST_SOURCES
    #file1_test.cc
)

# Create test executable and link to main library target
add_executable(oco_test ${TEST_SOURCES})
target_link_libraries(oco_test oco ${THIRDPARTY_LIBRARIES})

# Add test to ctest
add_test(NAME oco COMMAND oco_test)

# Expose test directory location to top level scope
set(OCO_TEST_DIR ${OCO_TEST_DIR} PARENT_SCOPE)
