Contents of the LWPR library  
(C) 2007 Stefan Klanke and Sethu Vijayakumar
sethu.vijayakumar@ed.ac.uk

The library is freely available under the terms of the LGPL 
with an exception that allows for static linking, see the
file COPYING.

Please see the file INSTALL.TXT  for installation instructions.

Inside the top-level directory created after unzipping the archive, 
there are several subdirectories that contain sources, include files 
and documentation for programming languages other than Matlab:

doc
    contains supplementary documentation and hints how to tune 
    learning parameters etc. 

matlab
    contains the Matlab functions (.m files). To use the LWPR library 
    from Matlab, all you have to do is to add this directory to your 
    Matlab path, and to run "lwpr_buildmex" within Matlab in order to 
    build the MEX wrappers. Recent versions of Octave (2.9.12 or later) 
    are compatible with Matlab's MEX-interface, and thus the build 
    script we provide works in that environment as well. 

include
    C header files of the LWPR library. 
    C++ header (lwpr.hh) file for wrapping the C library as a C++ class

src
    C sources.

mexsrc / mexoct
    C sources of Matlab/Octave MEX-wrappers, as well as directives 
    for building them using GNU autotools. On Windows, building 
    the MEX files is handled by the script lwpr_buildmex, so 
    probably you will not have to look into these.
    
example_c
    contains a simple demo that shows how to use the library from 
    a C program. 
    
example_cpp
    contains a demo how to use the C++ wrapper to call the LWPR library 
    in C++ style. 
    
python
    contains a Python extension module for LWPR, written in C, and 
    also a Python script demonstrating its usage. If you have Python's 
    distutils installed, you can build the extension using setup.py, 
    otherwise try the included Makefile on a Linux/Unix system.
    Please note that you need to have "numpy" already installed on
    your system.
    
html
    contains documentation for the C and C++ modules as generated 
    by Doxygen.
        
VisualC
    Visual Studio "solutions" and project files. Only tested with
    Visual Studio Express 2008.

MingW
    Contains a simple Makefile for building the C library and
    examples on Windows using the MinGW compiler. Probably also
    works with Cygwin.

tests
    Contains a simple test program (written in C) that checks
    some aspects of the library during a "make check" call
    on UNIX-like systems.




THIS SOURCE CODE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND, 
AND ITS AUTHOR AND THE JOURNAL OF MACHINE LEARNING RESEARCH (JMLR) 
AND JMLR'S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL 
WARRANTIES, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND 
ANY WARRANTIES OR NON INFRINGEMENT. THE USER ASSUMES ALL LIABILITY 
AND RESPONSIBILITY FOR USE OF THIS SOURCE CODE, AND NEITHER THE 
AUTHOR NOR JMLR, NOR JMLR'S PUBLISHERS AND DISTRIBUTORS, WILL BE 
LIABLE FOR DAMAGES OF ANY KIND RESULTING FROM ITS USE. 
Without limiting the generality of the foregoing, neither the 
author, nor JMLR, nor JMLR's publishers and distributors, warrant 
that the Source Code will be error-free, will operate without 
interruption, or will meet the needs of the user.
