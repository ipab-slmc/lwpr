cmake_minimum_required(VERSION 3.3.0)
project(lwpr)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)
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
find_package(Matlab COMPONENTS MX_LIBRARY MEX_COMPILER)

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

if (${BUILD_PYTHON})
  find_package(PythonLibs REQUIRED 3)
  include_directories(${PYTHON_INCLUDE_DIRS})
  add_library(pylwpr MODULE python/lwprmodule.c)
  target_link_libraries(pylwpr m lwpr ${PythonLibs_LIBRARIES})
  set_target_properties(pylwpr PROPERTIES PREFIX "")
  set_target_properties(pylwpr PROPERTIES OUTPUT_NAME "lwpr")
endif()

if (${MATLAB_FOUND})

    set(MATLAB_NATIVE matlab/lwpr_init.m matlab/lwpr_num_data.m matlab/lwpr_num_rfs.m matlab/lwpr_pls_stats.m matlab/lwpr_predict.m matlab/lwpr_predict_J.m matlab/lwpr_predict_JcJ.m matlab/lwpr_predict_JH.m matlab/lwpr_read_binary.m matlab/lwpr_read_xml.m matlab/lwpr_set.m matlab/lwpr_storage.m matlab/lwpr_update.m matlab/lwpr_write_binary.m matlab/lwpr_write_xml.m matlab/lwpr_x_check_add_projection.m matlab/lwpr_x_compute_projection.m matlab/lwpr_x_compute_projection_d.m matlab/lwpr_x_compute_projection_r.m matlab/lwpr_x_dist_derivatives.m matlab/lwpr_x_init_rf.m matlab/lwpr_x_kernel.m matlab/lwpr_x_predict_one.m matlab/lwpr_x_predict_one_gH.m matlab/lwpr_x_predict_one_J.m matlab/lwpr_x_predict_one_JcJ.m matlab/lwpr_x_update_distance_metric.m matlab/lwpr_x_update_means.m matlab/lwpr_x_update_one.m matlab/lwpr_x_update_regression.m matlab/test_lwpr_1D.m matlab/test_lwpr_2D.m matlab/test_lwpr_nD.m)

	set(MATLAB_SOURCES lwpr_x_dist_derivatives;lwpr_x_check_add_projection;lwpr_x_compute_projection;lwpr_x_compute_projection_r;lwpr_x_compute_projection_d;lwpr_x_update_distance_metric;lwpr_x_update_means;lwpr_x_update_regression;lwpr_x_init_rf;lwpr_x_update_one;lwpr_x_predict_one;lwpr_num_data;lwpr_num_rfs;lwpr_predict;lwpr_predict_J;lwpr_predict_JcJ;lwpr_predict_JH;lwpr_update;lwpr_storage;lwpr_write_binary;lwpr_read_binary)

	set(MEX_DEFINES MATLAB)
    if("${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    	set(MEX_DEFINES ${MEX_DEFINES} MX_COMPAT_32)
    endif()

	FOREACH(MEXFN ${MATLAB_SOURCES})  
	    matlab_add_mex(NAME ${MEXFN} SRC mexsrc/${MEXFN}.c src/lwpr_matlab.c ${LWPR_SOURCES} LINK_TO ${CMAKE_THREAD_LIBS_INIT} -lm)
	    target_compile_definitions(${MEXFN} PUBLIC ${MEX_DEFINES})
	ENDFOREACH()

	if (${EXPAT_FOUND})
	    matlab_add_mex(NAME lwpr_write_xml SRC mexsrc/lwpr_write_xml.c src/lwpr_matlab.c ${LWPR_SOURCES} LINK_TO ${CMAKE_THREAD_LIBS_INIT} -lm ${EXPAT_LIBRARIES})
	    matlab_add_mex(NAME lwpr_read_xml SRC mexsrc/lwpr_read_xml.c src/lwpr_matlab.c ${LWPR_SOURCES} LINK_TO ${CMAKE_THREAD_LIBS_INIT} -lm ${EXPAT_LIBRARIES})
	    target_compile_definitions(lwpr_write_xml PUBLIC ${MEX_DEFINES})
	    target_compile_definitions(lwpr_read_xml PUBLIC ${MEX_DEFINES})
	else()
	    matlab_add_mex(NAME lwpr_write_xml SRC mexsrc/lwpr_write_xml.c src/lwpr_matlab.c ${LWPR_SOURCES} LINK_TO ${CMAKE_THREAD_LIBS_INIT} -lm)
	    target_compile_definitions(lwpr_write_xml PUBLIC ${MEX_DEFINES})
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

if (${MATLAB_FOUND})
	FOREACH(MEXFN ${MATLAB_SOURCES})  
	    	install(TARGETS ${MEXFN}
	    DESTINATION share/Matlab
	    LIBRARY DESTINATION share/Matlab
	    ARCHIVE DESTINATION share/Matlab)
	ENDFOREACH()
	install(FILES ${MATLAB_NATIVE} DESTINATION share/Matlab)
endif()