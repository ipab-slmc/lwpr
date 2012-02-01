lib_LTLIBRARIES = liblwproctave.la 

liblwproctave_la_SOURCES = $(top_srcdir)/src/lwpr.c \
                     $(top_srcdir)/src/lwpr_aux.c \
                     $(top_srcdir)/src/lwpr_math.c \
                     $(top_srcdir)/src/lwpr_binio.c \
                     $(top_srcdir)/src/lwpr_mem.c \
                     $(top_srcdir)/src/lwpr_xml.c \
                     $(top_srcdir)/src/lwpr_matlab.c
             
liblwproctave_la_CFLAGS = -I$(top_srcdir)/include -I$(OCTINCLUDEPATH)/octave -DMATLAB
liblwproctave_la_LDFLAGS = -version-info 1:1:0

mexsources=lwpr_predict lwpr_predict_J lwpr_predict_JcJ lwpr_predict_JH lwpr_update lwpr_num_rfs lwpr_num_data lwpr_storage lwpr_write_xml lwpr_read_xml lwpr_write_binary lwpr_read_binary
mexsources+=lwpr_x_check_add_projection lwpr_x_compute_projection lwpr_x_compute_projection_d lwpr_x_compute_projection_r lwpr_x_dist_derivatives lwpr_x_init_rf lwpr_x_predict_one lwpr_x_update_distance_metric lwpr_x_update_means lwpr_x_update_one lwpr_x_update_regression 
mexflags=-DMATLAB -I$(top_srcdir)/include -L$(libdir) -llwproctave

install-exec-hook:
	@echo ===========================================================================
	@echo Now building MEX files with the following flags
	@echo $(mexflags)
	@echo ===========================================================================   
	@for x in $(mexsources); do echo Building Octave MEX file $$x; $(OCTAVE_PATH)/bin/mkoctfile --mex -o ../matlab/$$x.mex $(mexflags) ../mexsrc/$$x.c; done
	@echo ===========================================================================
	

