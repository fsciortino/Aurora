cmake_minimum_required(VERSION 3.17.2...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)


find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)


# F2PY headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)


add_custom_command(
  OUTPUT _auroramodule.c example-f2pywrappers.f
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/aurora/main.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/impden.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/math.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/grids.f90"
  VERBATIM
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/main.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/impden.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/math.f90"
          "${CMAKE_CURRENT_SOURCE_DIR}/aurora/grids.f90"
          -m _aurora --lower)

python_add_library(_aurora MODULE "${CMAKE_CURRENT_BINARY_DIR}/_auroramodule.c"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/main.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/impden.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/math.f90"
                   "${CMAKE_CURRENT_SOURCE_DIR}/aurora/grids.f90"
                   WITH_SOABI)
target_link_libraries(_aurora PRIVATE fortranobject)

install(TARGETS _aurora DESTINATION aurora/)
