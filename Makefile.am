ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src include example_c example_cpp tests @MEXSUB@ @OCTSUB@
EXTRA_DIST = Doxyfile README.TXT INSTALL.TXT
TESTS = tests/crosscheck

dist-hook:
	mkdir $(distdir)/doc
	cp -p $(top_srcdir)/doc/lwpr_doc.tex $(distdir)/doc
	cp -p $(top_srcdir)/doc/lwpr_doc.pdf $(distdir)/doc
	cp -p $(top_srcdir)/doc/lwpr_xml.dtd $(distdir)/doc
	
	mkdir -p $(distdir)/matlab
	cp -p $(top_srcdir)/matlab/*.m $(distdir)/matlab

	mkdir -p $(distdir)/mexsrc
	cp -p $(top_srcdir)/mexsrc/*.c $(distdir)/mexsrc      
   
	mkdir -p $(distdir)/python
	cp -p $(top_srcdir)/python/setup.py $(distdir)/python
	cp -p $(top_srcdir)/python/setup_nolib.py $(distdir)/python
	cp -p $(top_srcdir)/python/test.py $(distdir)/python
	cp -p $(top_srcdir)/python/lwprmodule.c $(distdir)/python
	cp -p $(top_srcdir)/python/Makefile $(distdir)/python
	cp -p $(top_srcdir)/src/lwpr_matlab.c $(distdir)/src/lwpr_matlab.c
	cp -p $(top_srcdir)/include/lwpr_config_def.h $(distdir)/include/lwpr_config.h

	mkdir -p $(distdir)/VisualC/
	cp -p $(top_srcdir)/VisualC/lwpr_mex.def     $(distdir)/VisualC
	cp -p $(top_srcdir)/VisualC/buildmex.bat     $(distdir)/VisualC
   
	mkdir -p $(distdir)/VisualC/Express2008/
	cp -p $(top_srcdir)/VisualC/Express2008/LWPR_VC2008.sln $(distdir)/VisualC/Express2008   

	mkdir -p $(distdir)/VisualC/Express2008/dynLWPR
	cp -p $(top_srcdir)/VisualC/Express2008/dynLWPR/dynLWPR.vcproj $(distdir)/VisualC/Express2008/dynLWPR
	cp -p $(top_srcdir)/VisualC/Express2008/dynLWPR/lwpr.def       $(distdir)/VisualC/Express2008/dynLWPR

	mkdir -p $(distdir)/VisualC/Express2008/libLWPR
	cp -p $(top_srcdir)/VisualC/Express2008/libLWPR/libLWPR.vcproj $(distdir)/VisualC/Express2008/libLWPR
   
	mkdir -p $(distdir)/VisualC/Express2008/Example_C
	cp -p $(top_srcdir)/VisualC/Express2008/Example_C/Example_C.vcproj $(distdir)/VisualC/Express2008/Example_C      
	
	mkdir -p $(distdir)/VisualC/Express2008/Example_C++
	cp -p $(top_srcdir)/VisualC/Express2008/Example_C++/Example_C++.vcproj $(distdir)/VisualC/Express2008/Example_C++   

	mkdir -p $(distdir)/MinGW
	cp -p $(top_srcdir)/MinGW/Makefile  $(distdir)/MinGW
