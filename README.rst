=================================
ReFRACtor Extension Template Repo
=================================

Introduction
------------
This repo is meant to serve as the template repository for instrument teams and others that will use or extend ReFRACtor.

The structure and content of this template allows users to make use of all existing ReFRACtor functionality
in their own C++ units and python modules while keeping those libraries and modules logically separated from the "core" ReFRACtor code base.

Setup and Use
-------------
To make use of this template, it is envisioned that users will make the following changes and additions while replacing ``<project>`` with the name of their project.

- Rename ``/script/setup_template_env.sh.in`` to ``/script/setup_<project>_env.sh.in``

- Modify ``/script/setup_<project>_env.sh.in``

  * Replace the variable names of ``template_install_path``, ``template_lib_path``, and ``template_python_path`` variables with ``<project>_install_path``, ``<project>_lib_path``, and ``<project>_python_path``

- Modify ``/CMakeLists.txt``

  * Set project() name
  * Set version_number
  * If you don't require ReFRACtor's python module, remove ``"COMPONENTS python"`` from the require line in the ``find_package(Refractor ...)`` command
  * If your project will not have a python component, comment out the ``add_subdirectory(python)`` line.
  * Set the filenames in the configure_file() to ``"${PROJECT_SOURCE_DIR}/script/setup_<project>_env.sh.in"`` and ``"${PROJECT_BINARY_DIR}/setup_<project>_env.sh"``
  * Replace the ``"${PROJECT_BINARY_DIR}/setup_template_env.sh"`` filename in ``install()`` with ``"${PROJECT_BINARY_DIR}/setup_<project>_env.sh"``

- Add C++ source implementation and header files to ``/lib/`` directory

- Modify ``/lib/CMakeLists.txt``

  * Change variables ${TEMPLATE_LIB_DIR} and ${TEMPLATE_INCLUDE_DIR} to ${<project>_LIB_DIR} and ${<project>_INCLUDE_DIR}
  * Add sources to LIB_SOURCES
  * Replace template with <project> in the add_library(), target_link_libraries() and install(TARGETS ...)

- Add Python SWIG interface files to ``/python/`` directory

- Modify ``python/CMakeLists.txt`` (if project has python components)

  * Set PYTHON_PACKAGE_NAME
  * Add interface files to MODULE_INTERFACE_FILES
  * Set the library target
  * Replace template with <project> in swig_link_libraries()

  **Note: It is important that the SWIG_TYPE_TABLE remain set to "refractor" in target_compile_definitions() in order to share type information with the ReFRACtor python modules.**

- Add test implementation and header files to ``/test/``

- Add input data required for testing to ``/test/in/``

- Modify test implementation and/or header files to point to ``/test/in/`` directory for input data

- Modify ``test/CMakeLists.txt``

  * Change variable ``TEMPLATE_TEST_DIR`` to ``<project>_TEST_DIR``
  * Add test sources to ``TEST_SOURCES``
  * Replace ``template_test`` with ``<project>_test`` in ``add_executable()``, ``target_link_libraries()``, and ``add_test()``
  * Replace ``template`` with ``<project>`` in ``target_link_libraries()`` and ``add_test()``

