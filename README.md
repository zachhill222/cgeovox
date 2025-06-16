# cgeovox
A C++ library for voxel-based geometry and finite elements.


# TODO
- Finish CHARMS refinement
	* basis function activation
	* basis function refinement
	* stiffness/mass matrix assembly

- Data structure refinement
	* ensure data in octrees can only be edited when needed
		- store data as const? seems excessive.
	* tweak gv::util::Box
		- add support for other data types than double
		- move semantics?