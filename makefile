# Usage:
# make aurora           # generate aurora shared-object library
# make clean     # delete previous versions of the aurora shared-object library
#
# When using "make" or "make jlib", attempt to create Julia library as well (requires Julia to be available)

.PHONY: all aurora clean

#flags=
#fcompiler=gnu95
flags="-fast"
fcompiler=intelem


############

all: aurora jlib

aurora :
	@echo "Generating aurora shared-object library"
	@echo "compiler " $(fcompiler) " flags: " ${flags}
	f2py3 -c --fcompiler=${fcompiler} -m _aurora aurora/main.f90 aurora/impden.f90 aurora/math.f90 aurora/grids.f90 --opt=${flags}
	mv _aurora.cpython*.so aurora/

jlib :
	julia -e 'import Pkg; Pkg.develop(path="aurora/jlib/"); Pkg.add("PackageCompiler"); Pkg.add("PyCall"); Pkg.build("PyCall")'
	python3 -m pip install --user julia
	python3 -c "import julia; julia.install()"
	python3 -m julia.sysimage aurora/jlib/aurora.so


clean :
	@echo "Eliminating aurora shared-object library"
	rm aurora/_aurora*.so
