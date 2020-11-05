# Usage:
# make aurora           # generate aurora shared-object library
# make clean             # delete previous versions of the aurora shared-object library
# make julia              # create Julia sysimage -- requires Julia to be available on system!
# make clean_julia     # delete Julia sysimage
#
# To create Fortran libraries for Aurora, make sure to select the correct compiler option at the top of this makefile.

.PHONY: all aurora clean

# option #1: basic gnu95 compiler (no flags is fine)
flags=
fcompiler=gnu95

# option #2: if Intel compilers are available, use this option:
#flags="-fast"
#fcompiler=intelem


############

all: aurora

aurora :
	@echo "Generating aurora shared-object library"
	@echo "compiler " $(fcompiler) " flags: " ${flags}
	f2py3 -c --fcompiler=${fcompiler} -m _aurora aurora/main.f90 aurora/impden.f90 aurora/math.f90 aurora/grids.f90 --opt=${flags}
	mv _aurora.cpython*.so aurora/

julia :
	julia -e 'import Pkg; Pkg.develop(path="aurora.jl/"); Pkg.add("PackageCompiler"); Pkg.add("PyCall"); Pkg.build("PyCall")'
	python3 -m pip install --user julia
	python3 -c "import julia; julia.install()"
	python3 -m julia.sysimage aurora.jl/sysimage.so


clean :
	@echo "Eliminating Aurora shared-object library"
	rm aurora/_aurora*.so

clean_julia :
	@echo "Eliminating Aurora Julia sysimage"
	rm aurora.jl/sysimage.so
