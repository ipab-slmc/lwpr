cmake_minimum_required(VERSION 2.8.5)
project(lwpr)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_EXPAT "Use libexpat for XML support" ON)
set(NUM_THREADS 1 CACHE STRING "Number of execution threads")

set(LWPR_AUTHOR sethu.vijayakumar@ed.ac.uk)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Threads REQUIRED)
if (${USE_EXPAT})
  find_package(EXPAT)
  set(HAVE_LIBEXPAT ${EXPAT_FOUND})
endif()

find_package(Eigen3)
add_definitions(-DHAVE_CONFIG_H)

include_directories(
  include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${EXPAT_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
) 

set(LWPR_VERSION_STRING 1.2.6)
set(LWPR_SOVERSION_STRING 1)

if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/include/lwpr_config.h)
  message(FATAL_ERROR "'include/lwpr_config.h' exists in the source path. Have you tried to build LWPR using autotools? Please delete the file.")
endif()

CONFIGURE_FILE(include/lwpr_config.h.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/lwpr_config.h)

set(LWPR_SOURCES src/lwpr.c src/lwpr_aux.c src/lwpr_binio.c src/lwpr_math.c src/lwpr_mem.c src/lwpr_xml.c)

if (${BUILD_SHARED_LIBS})
  add_library(lwpr SHARED ${LWPR_SOURCES})
  target_link_libraries(lwpr ${CMAKE_THREAD_LIBS_INIT} -lm ${EXPAT_LIBRARIES})
  set_target_properties(lwpr PROPERTIES VERSION ${LWPR_VERSION_STRING} SOVERSION ${LWPR_SOVERSION_STRING})
  target_compile_definitions(lwpr PRIVATE LIBRARY_EXPORTS=1)
endif()

if (${BUILD_STATIC_LIBS})
  add_library(lwpr_static STATIC ${LWPR_SOURCES})
  if(MSVC)
    set_target_properties(lwpr_static PROPERTIES OUTPUT_NAME "lwpr_static")
  else()
    set_target_properties(lwpr_static PROPERTIES OUTPUT_NAME "lwpr")
  endif()
  set_target_properties(lwpr_static PROPERTIES VERSION ${LWPR_VERSION_STRING})
endif()

if (${BUILD_EXAMPLES})
  add_executable(example_c example_c/cross.c)
  target_link_libraries(example_c lwpr)
  add_executable(example_cpp example_cpp/cross.cc)
  target_link_libraries(example_cpp lwpr)
  if (${EIGEN3_FOUND})
    add_executable(example_eigen_cpp example_cpp/cross_eigen.cc)
    target_link_libraries(example_eigen_cpp lwpr)
  endif()
endif()

if (${BUILD_SHARED_LIBS})
	install(TARGETS lwpr
	    DESTINATION lib
	    LIBRARY DESTINATION lib
	    ARCHIVE DESTINATION lib)
endif()

if (${BUILD_STATIC_LIBS})
	install(TARGETS lwpr_static
	    DESTINATION lib
	    LIBRARY DESTINATION lib
	    ARCHIVE DESTINATION lib)
endif()

if (${BUILD_EXAMPLES})
	install(TARGETS example_c example_cpp RUNTIME DESTINATION bin)
    if (${EIGEN3_FOUND})
    	install(TARGETS example_eigen_cpp RUNTIME DESTINATION bin)
    endif()
endif()
