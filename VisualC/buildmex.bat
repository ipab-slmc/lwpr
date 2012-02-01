set INCDIR=/I "C:\Program Files\Matlab71\extern\include"
set LIBDIR=/LIBPATH:"C:\Program Files\Matlab71\extern\lib\win32\microsoft" 

set CFLAGS=/DMATLAB /O2 /Oy- /I "..\include" /DWIN32 /GR /Zp8 /MT /W3 /nologo /c /D_CRT_SECURE_NO_DEPRECATE %INCDIR%

set LFLAGS=/NOLOGO /DLL /MANIFEST:NO /SUBSYSTEM:WINDOWS /MACHINE:X86  
set LFLAGS=%LFLAGS% %LIBDIR% libmex.lib libmat.lib libmx.lib

set MEXFL=%LFLAGS% /EXPORT:mexFunction lwpr_mex.lib
set MEXFC=%CFLAGS% /DMATLAB_MEX_FILE
set MEXT=mexw32

set srcs=lwpr lwpr_aux lwpr_mem lwpr_binio lwpr_xml lwpr_math lwpr_matlab
set mexfiles=lwpr_storage lwpr_predict lwpr_update lwpr_num_data lwpr_num_rfs

rem First, compile the C library functions. 
rem Because CFLAGS contains /DMATLAB, the library will use mxMalloc instead of malloc etc.
for %%s in (%srcs%) do cl  %CFLAGS% ..\src\%%s.c

rem Now, link the C library functions as a DLL
rem This also creates the import library lwpr_mex.lib
link %LFLAGS% /DEF:"lwpr_mex.def" /OUT:lwpr_mex.dll lwpr.obj lwpr_aux.obj lwpr_mem.obj lwpr_binio.obj lwpr_xml.obj lwpr_math.obj lwpr_matlab.obj
del lwpr.obj lwpr_aux.obj lwpr_mem.obj lwpr_binio.obj lwpr_xml.obj lwpr_math.obj lwpr_matlab.obj

rem Compile all the MEX sources	
for %%s in (%mexfiles%) do cl %MEXFC% ..\mexsrc\%%s.c

rem Link them to DLLs  (or rather, .mexw32 files)
for %%s in (%mexfiles%) do link %MEXFL% /OUT:%%s.%MEXT% %%s.obj	

rem And finally, move resulting MEX-files into the ../matlab directory (together with lwpr_mex.dll), clean up
move /Y *.%MEXT% ..\matlab
move lwpr_mex.dll ..\matlab
del lwpr_mex.lib lwpr_mex.exp
for %%s in (%mexfiles%) do del %%s.exp %%s.lib %%s.obj
