AC_INIT([lwpr], [1.2.6], [sethu.vijayakumar@ed.ac.uk])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADERS([include/lwpr_config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_PROG_AR
LT_INIT

AC_PROG_CC
CFLAGS=-O3 -g
AC_PROG_CXX
AC_PROG_LIBTOOL

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

AC_CHECK_LIB([expat], [XML_ParserCreate])
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_LIB([pthread], [pthread_create])

AC_CHECK_FUNCS([memset])
AC_CHECK_FUNCS([strcasecmp])
AC_CHECK_FUNCS([strtol])

AC_C_CONST
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_STRTOD
AC_TYPE_SIZE_T

AC_HEADER_STDBOOL
AC_HEADER_STDC
AC_STRUCT_TM

AC_DEFUN([SET_NUM_THREADS],
[echo You selected --enable-threads=$enable_threads
numthr=0
if test "X$enable_threads" = "Xno"; then
   numthr=1
elif test $enable_threads -ge 1; then
  if test $enable_threads -le 32; then
     numthr=$enable_threads
  fi
fi
if test "$numthr" = "0"; then  
   echo Number of threads must be between 1 and 32
   exit 1
fi
AC_DEFINE_UNQUOTED([NUM_THREADS],$numthr,[Number of threads to use])
])


MEXSUB=
OCTSUB=

AC_DEFUN([SET_MATLAB],
[echo ==================================================================================
echo You selected --with-matlab=$with_matlab
if test "X$with_matlab" = "Xno"; then
   echo MEX files will not be built
else
   if test -e $with_matlab/extern/include/mex.h -a -e $with_matlab/bin/mex; then
      MATLAB_PATH=$with_matlab
      MEXSUB=mexsrc
      echo Include path for Matlab MEX files is $with_matlab/extern/include
      AC_CONFIG_FILES([mexsrc/Makefile])
   else
      echo ERROR: At least one of the following files 
      echo *  $with_matlab/extern/include/mex.h 
      echo *  $with_matlab/bin/mex 
      echo was not found. Please make sure that echo you have selected the right 
      echo path with the --with-matlab option.
      exit 1
   fi
fi
echo ==================================================================================
])


AC_DEFUN([SET_OCTAVE],
[echo ==================================================================================
echo You selected --with-octave=$with_octave
if test "X$with_octave" = "Xno"; then
   echo MEX files \(for Octave\) will not be built
else
   if test -e $with_octave/bin/mkoctfile -a -e $with_octave/bin/octave-config; then
      OCTAVE_PATH=$with_octave
      OCTSUB=mexoct
      OCTINCLUDEPATH=`$with_octave/bin/octave-config -p OCTINCLUDEDIR`
      echo Include path for Octave MEX files is $OCTINCLUDEPATH
      AC_CONFIG_FILES([mexoct/Makefile])
   else
      echo ERROR: At least one of the following files 
      echo *  $with_octave/bin/mkoctfile
      echo *  $with_matlab/bin/octave-config 
      echo was not found. Please make sure that echo you have selected the right 
      echo path with the --with-octave option.
      exit 1
   fi
fi
echo ==================================================================================
])


AC_ARG_ENABLE([threads], 
   [AC_HELP_STRING([--enable-threads=K],
                  [use K threads (1..32) for predictions and updates (default is 1, no multi-threading)])],
   [SET_NUM_THREADS], 
   [AC_DEFINE([NUM_THREADS],[1],[Number of threads to use])]
)

AC_ARG_WITH([matlab], 
   [AC_HELP_STRING([--with-matlab=<path to Matlab installation>],
                  [Enables building MEX files and a special shared library liblwprmex.so])],
   [SET_MATLAB]
)

AC_ARG_WITH([octave], 
   [AC_HELP_STRING([--with-octave=<path to Octave installation (must be >= 2.9.12) >],
                  [Enables building Octave MEX files and a special shared library liblwproctave.so])],
   [SET_OCTAVE]
)

AC_SUBST([MATLAB_PATH])
AC_SUBST([OCTAVE_PATH])
AC_SUBST([MEXSUB])
AC_SUBST([OCTSUB])
AC_SUBST([OCTINCLUDEPATH])

AC_CONFIG_FILES([Makefile src/Makefile include/Makefile example_c/Makefile example_cpp/Makefile tests/Makefile])
AC_OUTPUT
