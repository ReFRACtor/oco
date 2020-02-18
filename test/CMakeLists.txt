# Enable ctest
enable_testing()

set(OCO_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${OCO_TEST_DIR} ${OCO_LIB_DIR})

# Add each test file to TEST_SOURCES
file(GLOB TEST_SOURCES "*.cc")

# Create test case fixtures default properties file from CMake variables
configure_file (
    global_fixture_default.cc.in 
    "${CMAKE_CURRENT_BINARY_DIR}/global_fixture_default.cc"
)

set(RUN_RF_WRAP ${CMAKE_CURRENT_BINARY_DIR}/run_rf_wrap.sh)
configure_file(run_rf_wrap.sh.in ${RUN_RF_WRAP})

list(APPEND TEST_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/global_fixture_default.cc")

# Create test executable and link to main library target
add_executable(oco_test ${TEST_SOURCES})
target_link_libraries(oco_test oco ${THIRDPARTY_LIBRARIES})
set_target_properties(oco_test PROPERTIES INSTALL_RPATH "${PROJECT_BINARY_DIR}/lib;${CMAKE_INSTALL_PREFIX}/lib")
if(DEFINED ENV{CONDA_PREFIX})
  set_property(TARGET oco_test APPEND PROPERTY INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
endif(DEFINED ENV{CONDA_PREFIX})
set_target_properties(oco_test PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# Add test to ctest
add_test(NAME oco COMMAND oco_test --log_level=test_suite)

# Expose test directory location to top level scope
set(OCO_TEST_DIR ${OCO_TEST_DIR} PARENT_SCOPE)

add_custom_target(python_install
  COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target install
  DEPENDS python)

# Retrieval test. We just check that this runs for now
add_test(NAME run_log_h2o_run_test
        COMMAND ${RUN_RF_WRAP} -v ${PROJECT_SOURCE_DIR}/config/log_h2o_example_retrieval.py)

add_custom_target(run_log_h2o_test
        COMMAND ctest --progress --verbose -R run_log_h2o_run_test
        DEPENDS python_install)

# Compare results to a RtRetrievalFramework (OCO-2 production code) output
add_test(NAME rtr_comparison
        COMMAND ${PYTHON_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/test/python/test_rtr_comparison.py)

add_custom_target(rtr_comparison_test
        COMMAND ctest --progress --verbose -R rtr_comparison
        DEPENDS python_install)
      
# Target to run all tests
add_custom_target(all_test COMMAND ctest --progress DEPENDS oco_test python_install)
