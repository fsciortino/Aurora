# Usage:
# make           # generate aurora shared-object library
# make clean     # delete previous versions of the aurora shared-object library

.PHONY: all aurora clean

flags=
fcompiler=gnu95
#flags="-fast"
#fcompiler=intelem


############

all: aurora

aurora :
	@echo "Generating aurora shared-object library"
	@echo "compiler " $(fcompiler) " flags: " ${flags}
	f2py3 -c --fcompiler=${fcompiler} -m _aurora flib/main.f90 flib/impden.f90 flib/math.f90 flib/grids.f90 --opt=${flags}
	mv _aurora.cpython*.so flib/
	julia -e 'import Pkg; Pkg.develop(path="jlib/"); Pkg.add("PackageCompiler")'
	python -m julia.sysimage jlib/aurora.so 



clean : 
	@echo "Eliminating aurora shared-object library"
	rm flib/_aurora*.so
