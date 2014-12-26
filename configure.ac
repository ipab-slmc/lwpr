#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([FULL-PACKAGE-NAME], [VERSION], [BUG-REPORT-ADDRESS])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([example_cpp/cross.cc])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
LT_INIT

# Checks for libraries.
# FIXME: Replace `main' with a function in `-llwpr':
AC_CHECK_LIB([lwpr], [main])
# FIXME: Replace `main' with a function in `-llwprmex':
AC_CHECK_LIB([lwprmex], [main])
# FIXME: Replace `main' with a function in `-llwproctave':
AC_CHECK_LIB([lwproctave], [main])
# FIXME: Replace `main' with a function in `-lm':
AC_CHECK_LIB([m], [main])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([memset sqrt strcasecmp strtol])

AC_CONFIG_FILES([Makefile
                 example_c/Makefile
                 example_cpp/Makefile
                 include/Makefile
                 mexoct/Makefile
                 mexsrc/Makefile
                 src/Makefile
                 tests/Makefile])
AC_OUTPUT
