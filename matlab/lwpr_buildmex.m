function lwpr_buildmex
% lwpr_buildmex
%
% This routine builds MEX-equivalents for most lwpr_* MATLAB functions,
% yielding a quite dramatic speed-up.
%

% LWPR: A MATLAB library for incremental online learning
% Copyright (C) 2007  Stefan Klanke, Sethu Vijayakumar, Stefan Schaal
% Contact: sethu.vijayakumar@ed.ac.uk
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Library General Public
% License along with this library; if not, write to the Free
% Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

if nargin<1
   use_expat = 0;
end

mypath=which('lwpr_buildmex');
pos=strfind(mypath,'lwpr_buildmex.m');
olddir = pwd;
basedir = mypath(1:pos-1);
cd(basedir);

fprintf(1,'Checking for EXPAT dependency ... ');

mex -DMATLAB -I../include ../mexsrc/lwpr_has_expat.c
use_expat = lwpr_has_expat;
if use_expat
   fprintf(1,'yes\n');
else
   fprintf(1,'no\n');
end

fprintf(1,'\nC library:\n');

srcs = { '../src/lwpr.c', ...
         '../src/lwpr_aux.c', ...
         '../src/lwpr_mem.c', ...
         '../src/lwpr_math.c', ...
         '../src/lwpr_xml.c', ...
         '../src/lwpr_binio.c', ...         
         '../src/lwpr_matlab.c'};

for i=1:length(srcs)
   fprintf(1,'Compiling module "%s" ...\n',srcs{i});
   command = sprintf('mex -c -DMATLAB -I../include %s',srcs{i});
   eval(command);
end  

if ispc
   objs = ['lwpr.obj ' ...
           'lwpr_aux.obj ' ...   
           'lwpr_mem.obj ' ...
           'lwpr_math.obj ' ...           
           'lwpr_matlab.obj'];
   bobj =  'lwpr_binio.obj';
   xobj =  'lwpr_xml.obj';
else
   objs = ['lwpr.o ' ...
           'lwpr_aux.o ' ...   
           'lwpr_mem.o ' ...
           'lwpr_math.o ' ...
           'lwpr_matlab.o'];
   bobj =  'lwpr_binio.o';           
   xobj =  'lwpr_xml.o';
end

if exist('OCTAVE_VERSION')
   % Octave versions >= ~2.9.10 can handle MATLAB-style MEX files,
   % but Octave's mex script apparently puts the object files where
   % the source files are, and not into the current directory
   objs = ['../src/lwpr.o ' ...
           '../src/lwpr_aux.o ' ...      
           '../src/lwpr_mem.o ' ...
           '../src/lwpr_math.o ' ...
           '../src/lwpr_matlab.o'];
   bobj =  '../src/lwpr_binio.o';
   xobj =  '../src/lwpr_xml.o';
end

xfuncs = {'lwpr_x_dist_derivatives', ...
         'lwpr_x_check_add_projection', ...
         'lwpr_x_compute_projection', ...
         'lwpr_x_compute_projection_r', ...
         'lwpr_x_compute_projection_d', ...         
         'lwpr_x_update_distance_metric', ...
         'lwpr_x_update_means', ...
         'lwpr_x_update_regression', ...
         'lwpr_x_check_add_projection',...
         'lwpr_x_init_rf', ...
         'lwpr_x_update_one', ...
         'lwpr_x_predict_one'};
         
funcs = {'lwpr_num_data', ...
         'lwpr_num_rfs', ...
         'lwpr_predict', ...
         'lwpr_predict_J', ...         
         'lwpr_predict_JcJ', ...                  
         'lwpr_predict_JH', ...                  
         'lwpr_update', ...
         'lwpr_storage'};

bfuncs = {'lwpr_write_binary', ...
          'lwpr_read_binary'};
         
funcs = [xfuncs funcs];
                  
fprintf(1,'\nMEX functions:\n');
         
for i=1:length(funcs)
   fprintf(1,'Building "%s" ...\n',funcs{i});
   command = sprintf('mex ../mexsrc/%s.c %s -I../include',funcs{i},objs);   
   eval(command);
end  

for i=1:length(bfuncs)
   fprintf(1,'Building "%s" ...\n',bfuncs{i});
   command = sprintf('mex ../mexsrc/%s.c %s %s -I../include',bfuncs{i},objs,bobj);   
   eval(command);
end 


if use_expat
   fprintf(1,'Building "lwpr_write_xml" ...\n');
   command = sprintf('mex ../mexsrc/lwpr_write_xml.c %s %s -I../include -lexpat',objs,xobj);
   eval(command);

   fprintf(1,'Building "lwpr_read_xml" ...\n');
   command = sprintf('mex ../mexsrc/lwpr_read_xml.c %s %s -I../include -lexpat',objs,xobj);
   eval(command);
else
   fprintf(1,'Building "lwpr_write_xml" ...\n');
   command = sprintf('mex ../mexsrc/lwpr_write_xml.c %s %s -I../include',objs,xobj);
   eval(command);
end


if ispc
   delete('*.obj');
else
   delete('*.o');
end

cd(olddir);
