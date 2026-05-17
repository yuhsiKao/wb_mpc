cmake_minimum_required(VERSION 3.14)
project(casadi_catkin)

find_package(catkin REQUIRED)
include(ExternalProject)

# ── Install prefix: devel space so downstream find_package(casadi) works ──
set(CASADI_INSTALL_PREFIX ${CATKIN_DEVEL_PREFIX})
set(CASADI_SOURCE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/../casadi)
set(CASADI_BUILD_DIR      ${CMAKE_CURRENT_BINARY_DIR}/casadi_build)
set(CASADI_LIB            ${CASADI_INSTALL_PREFIX}/lib/libcasadi.so)

file(MAKE_DIRECTORY ${CASADI_INSTALL_PREFIX}/include)
file(MAKE_DIRECTORY ${CASADI_INSTALL_PREFIX}/lib)

ExternalProject_Add(casadi_external
  SOURCE_DIR ${CASADI_SOURCE_DIR}
  BINARY_DIR ${CASADI_BUILD_DIR}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CASADI_INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_LIBDIR=lib
    # Enable IPOPT (already installed system-wide)
    -DWITH_IPOPT=ON
    -DWITH_BUILD_IPOPT=OFF
    # Disable all language front-ends and tests
    -DWITH_PYTHON=OFF
    -DWITH_MATLAB=OFF
    -DWITH_OCTAVE=OFF
    -DBUILD_TESTING=OFF
  BUILD_BYPRODUCTS ${CASADI_LIB}
  # Do not re-run cmake on every build (source does not change)
  BUILD_ALWAYS OFF
)

# Create an imported target so downstream targets can link against casadi
# even within the same catkin build invocation.
add_library(casadi SHARED IMPORTED GLOBAL)
add_dependencies(casadi casadi_external)
set_target_properties(casadi PROPERTIES
  IMPORTED_LOCATION ${CASADI_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${CASADI_INSTALL_PREFIX}/include
)

catkin_package(
  INCLUDE_DIRS ${CASADI_INSTALL_PREFIX}/include
  LIBRARIES    casadi
)
