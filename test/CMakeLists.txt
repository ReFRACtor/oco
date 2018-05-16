set(TEMPLATE_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${TEMPLATE_TEST_DIR} ${TEMPLATE_LIB_DIR})

# Add each test file to TEST_SOURCES
set(TEST_SOURCES
    #file1_test.cc
)

# Create test executable and link to main library target
add_executable(template_test ${TEST_SOURCES})
target_link_libraries(template_test template ${THIRDPARTY_LIBRARIES})

# Add test to ctest
add_test(NAME template COMMAND template_test)

# Expose test directory location to top level scope
set(TEMPLATE_TEST_DIR ${TEMPLATE_TEST_DIR} PARENT_SCOPE)
