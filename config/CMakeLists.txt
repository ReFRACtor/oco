set(OCO_CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/etc/oco/config")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${OCO_CONFIG_DIR}/..
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.h5"
        PATTERN "*.nc"
        PATTERN "README")

# Expose install path to parent scope
set(OCO_CONFIG_DIR ${OCO_CONFIG_DIR} PARENT_SCOPE)

