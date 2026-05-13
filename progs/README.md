# About

Included here are all external binaries/programs we typically use in CSPy

The format for directory names is:

`{hostname}_{compiler}_{compiler_version}`

and executables will have their version appended to the executable filename.

The compilers used were:

Iridis4: Intel 17.0.0 20160721, gfortran 6.1.0
Iridis5: Intel 18.0.1 20171018, gfortran 6.4.0

And the following compile flags were used:

Intel: `-xHost -Ofast -ipo`
Intel (prof): `-xHost -Ofast -ipo -pg`

This means the iridis4 binaries are generated for the sandybridge architecture
(circa 2011) whereas iridis5 binaries are generated for the skylake architecture
(circa 2015). Older CPUs than these are not supported.

