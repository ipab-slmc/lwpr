CFLAGS+=-I../include -O3

SRCS=../src/lwpr.c ../src/lwpr_aux.c ../src/lwpr_mem.c ../src/lwpr_math.c ../src/lwpr_xml.c ../src/lwpr_binio.c
OBJS=lwpr.o lwpr_aux.o lwpr_mem.o lwpr_math.o lwpr_xml.o lwpr_binio.o

default: dynamic example_c example_cpp

dynamic:
	gcc -shared -o liblwpr.dll $(SRCS) $(CFLAGS) -Wall -Wl,--out-implib,liblwpr.a

static:	
	gcc -c $(SRCS) $(CFLAGS)
	ar cru liblwpr.a $(OBJS)
	ranlib liblwpr.a
	
clean:
	-rm *.o
	
example_c:
	gcc -o cross.exe ../example_c/cross.c -llwpr -L. $(CFLAGS)
	
example_cpp:
	g++ -o cross++.exe ../example_cpp/cross.cc -llwpr -L. $(CFLAGS)