# Description
Difftools is a set of differentiation tools to be applied on 1D or 2D on regular grids that represent some sort of data. It applies a finite differences approach, where the order depends on the desired precision and the resolution of the grid.

# Roadmap
For the moment, the actual code is a prototype of the final idea, which will have:
- Written in a faster language than Python (Fortran for example)
- Add parallelization (both on thread-based or GPU-based)
- Add capabilities in 3D
- Add the ability to work on non-regular data
- Add other functionalities such as the computation of Jacobians, Hessians and Taylor expansions.

# Bilbiography
This excellent [paper](https://www.geometrictools.com/Documentation/FiniteDifferences.pdf)
