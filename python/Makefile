# This Makefile should only be used in case  "setup.py"  fails,
# for example, because Python's "distutils" is not available
# 
# Modify CFLAGS and LDFLAGS so that they point to the Python
# inlude and extension directories.
#
# If you installed the LWPR library in a non-standard location,
# you also need to include this as well

CFLAGS+=-I/usr/include/python2.4 -I/usr/lib/python2.4/site-packages/numpy/core/include
LDFLAGS+=-lpython2.4 -lm -llwpr

default:
	gcc -shared -o lwpr.so lwprmodule.c $(CFLAGS) $(LDFLAGS)

