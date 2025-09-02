#pragma once


#include "util/point.hpp"
#include "util/octree.hpp"
#include "util/box.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"

#include "util/octree_util.hpp" //for viewing octree structure
#include "mesh/Q1.hpp" //unstructured voxel mesh

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

namespace gv::geometry{

	//class for storing octrees efficiently
	template <typename Particle_t, size_t n_data=8>
	class ParticleOctree : public gv::util::BasicOctree<Particle_t, 3, n_data> {
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t = gv::util::Box<3,double>;

		ParticleOctree() : gv::util::BasicOctree<Particle_t, 3, n_data>() {}
		ParticleOctree(const Box_t &bbox) : gv::util::BasicOctree<Particle_t, 3, n_data>(bbox, 64) {}

		//check if a point is in any particle
		bool is_in_particle(const Point_t &point) const
		{
			std::vector<size_t> d_idx = this->get_data_indices(point);
			for (size_t i=0; i<d_idx.size(); i++) {if ((*this)[d_idx[i]].contains(point)) {return true;}}
			return false;
		}

	private:
		bool is_data_valid(Box_t const &box, Particle_t const &P) const override {return gv::geometry::collides_GJK(box,P);}
	};



	//struct for meshing options
	struct AssemblyMeshOptions
	{
		bool include_void = true;
		bool include_solid = true;
		bool include_interface = true;
		//the void region is the set of voxels with no vertex contained in a particle
		//the solid region is the set of voxels with all 8 vertices contained in some particle
		//the interface region is the set of voxels with at least one (but not all) vertex contained in some particle


		bool check_centroid = false;
		//if check_centroid=true, then the interface region is split into two sub-regions
		//interface_void: the set of all voxels that are in the interface but whos centroid is in the void region
		//interface_solid: the set of all voxels that are in the interface but whos centroid is in the solid region
		//do not set this to "true" if you are only meshing the interface

		int void_marker = 0;
		int solid_marker = 1;
		int interface_marker = 2;

		gv::util::Point<3,size_t> N {32, 32, 32};
		double scale = 1.0; //TODO: implement functionality to scale/change length units.
	};
	

	//class for interacting with ParticleOctree
	template <typename Particle_t, size_t n_data=8>
	class Assembly
	{
	public:
		using ParticleList_t = ParticleOctree<Particle_t, n_data>;
		using Point_t = gv::util::Point<3,double>;
		using Box_t = gv::util::Box<3,double>;


		Assembly() : _particles() {};
		Assembly(const std::string filename, const std::string columns) : _particles() {readfile(filename, columns);}

		//check if a point is in any particle
		bool is_in_particle(const Point_t &point) const
		{	
			if (not _particles.bbox().contains(point)) {return false;}
			return _particles.is_in_particle(point);
		}

		//read particles from file with specified format. TODO: read format from start of file.
		void readfile(const std::string filename, const std::string columns);

		//get bounding box
		Box_t bbox() const {return _particles.bbox();}

		//save geometry to a text file as a rectangular prism of sampled points with the. voidspace=0, solidspace=1.
		void save_geometry(const std::string filename, const Box_t &box, const size_t N[3]) const;
		void save_geometry(const std::string filename, const size_t  N[3]) const {save_geometry(filename, this->_particles.bbox(), N);}
		void save_solid(const std::string filename, const Box_t &box, const size_t N[3]) const;
		void save_solid(const std::string filename, const size_t  N[3]) const {save_solid(filename, this->_particles.bbox(), N);}

		//view octree structure of _particles octree
		void view_octree_vtk(const std::string filename="octree_structure.vtk") const {gv::util::view_octree_vtk(_particles, filename);}

		//create unstructured voxel mesh of the entire region. mark elements with 0 for void, 1 for solid, and 2 for interface
		void create_voxel_mesh_Q1(gv::mesh::VoxelMeshQ1 &out_mesh, const Box_t &box, const AssemblyMeshOptions &opts) const;
		void create_voxel_mesh_Q1(gv::mesh::VoxelMeshQ1 &out_mesh, const AssemblyMeshOptions &opts) const {create_voxel_mesh_Q1(out_mesh, this->_particles.bbox(), opts);}

		//check if a voxel should be meshed and what its marker should be (given mesh options)
		void check_voxel(const Box_t &voxel, const AssemblyMeshOptions &opts, int &marker, bool &include) const;

	private:
		ParticleList_t _particles;
	};


	template <typename Particle_t, size_t n_data>
	void Assembly<Particle_t, n_data>::check_voxel(const Box_t &voxel, const AssemblyMeshOptions &opts, int &marker, bool &include_element) const
	{
		//check if the element should be marked as void, solid, or interface
		int n_vert = 0; //number of vertices in the solid phase
		for (int i=0; i<8; i++) {if (is_in_particle(voxel.voxelvertex(i))) {n_vert += 1;}}

		if (n_vert==0) {marker = opts.void_marker;}
		else if (n_vert==8) {marker = opts.solid_marker;}
		else {marker = opts.interface_marker;}

		//check if the element should be added to the mesh
		include_element = false;
		if (marker==opts.solid_marker)
		{
			if (opts.include_solid) {include_element=true;}
		}
		else if (marker==opts.void_marker)
		{
			if (opts.include_void) {include_element=true;}
		}
		else if (opts.include_interface) //marker is interface, only continue here if we are allowed to include it in the mesh
		{
			if (opts.check_centroid) //check interface more closely by checking the center
			{
				bool center_in_solid = is_in_particle(voxel.center());
				if (center_in_solid and opts.include_solid) {include_element=true;} //center is in the solid, include element as interface (mostly solid)
				else if (!center_in_solid and opts.include_void) {include_element=true;} //center is in the void, include element as interface (mostly void)
			}
			else {include_element = true;} //include the interface without checking the center
		}
	}




	//add particles from a file to the current assembly. TODO: split into smaller functions.
	template <typename Particle_t, size_t n_data>
	void Assembly<Particle_t, n_data>::readfile(const std::string filename, const std::string columns)
	{
		// COLUMN OPTIONS:
		// -id (IDENTIFIER, int)
		// -rrr (TRIPLE RADIUS, double[3])
		// -xyz (CENTER, double[3])
		// -eps (SHAPE PARAMETERS, double[2])
		// -v (VOLUME, double)
		// -q (QUATERNION, double[4])
		// -l (BOUNDING BOX LENGTH, double[3])

		// INITIALIZE TEMPORARY STORAGE
		std::vector<Particle_t> temp_particles;

		int id;
		double rx, ry, rz;
		double x, y, z;
		double eps[2];
		double vol;
		double qw, qx, qy, qz;
		double lx, ly, lz;

		enum col_string_to_int {
			ID, RRR, XYZ, EPS, VOL, QUAT, LEN
		};

		//SET USE/SKIP COLUMNS
		std::vector<int> col_param;
		std::stringstream ss(columns);
		std::string col_opt;
		char del = '-';
		while (getline(ss, col_opt, del)){
			col_opt.erase(remove_if(col_opt.begin(), col_opt.end(), isspace), col_opt.end()); //strip whitespace
			if (col_opt.size()>0){
				if (col_opt.compare("id")==0){
				col_param.push_back(ID);
				}else if (col_opt.compare("rrr")==0){
					col_param.push_back(RRR);
				}else if (col_opt.compare("eps")==0){
					col_param.push_back(EPS);
				}else if (col_opt.compare("v")==0){
					col_param.push_back(VOL);
				}else if (col_opt.compare("xyz")==0){
					col_param.push_back(XYZ);
				}else if (col_opt.compare("q")==0){
					col_param.push_back(QUAT);
				}else if (col_opt.compare("l")==0){
					col_param.push_back(LEN);
				}else{
					std::string error_message = "Unkonwn parameter: " + col_opt;
					throw std::runtime_error(error_message);
				}
			}
			
		}


		//OPEN FILE AS INPUT FILE STREAM
		std::ifstream _file(filename);
		std::string line;

		if( not _file.is_open() )
		{
			std::cout << "Could not open " << filename << std::endl;
			return;
		}

		//READ FILE
		while (getline(_file, line)){
			if (!line.empty() && line[0] != '#'){
				std::istringstream iss(line);
				for (size_t param_idx=0; param_idx<col_param.size(); param_idx++){
					switch (col_param[param_idx]){
					case RRR:
						iss >> rx; //x-radius
						iss >> ry; //y-radius
						iss >> rz; //z-radius
						break;
					case XYZ:
						iss >> x; //x-coordinate of center
						iss >> y; //y-coordinate of center
						iss >> z; //z-coordinate of center
						break;
					case EPS:
						iss >> eps[0]; //super-ellipsoid shape parameter (local xy-plane)
						iss >> eps[1]; //super-ellipsoid shape parameter (local z-axis)
						break;
					case QUAT:
						iss >> qw; //rotation angle from global coordinates to particle local coordinates
						iss >> qx; //rotation axis x-component
						iss >> qy; //rotation axis y-component
						iss >> qz; //rotation axis z-component
						break;
					case VOL:
						iss >> vol; //volume of particle
						break;
					case LEN:
						iss >> lx; //assembly bounding box length
						iss >> ly; //assembly bounding box width
						iss >> lz; //assembly bounding box height
						break;
					case ID:
						iss >> id; //particle unique ID in assembly
						break;
					}
				}

				Particle_t P(Point_t {rx,ry,rz}, Point_t {x,y,z}, gv::util::Quaternion<double>(qw,-qx,-qy,-qz), eps[0], eps[1]);
				temp_particles.push_back(P);
			}
		}

		//ADD EXISTING PARTICLES TO LIST
		for (size_t i=0; i<_particles.size(); i++) {temp_particles.push_back(_particles[i]);}

		//GET BOUNDING BOX SIZE
		Box_t bbox = temp_particles[0].bbox();
		for (size_t i=1; i<temp_particles.size(); i++) {bbox.combine(temp_particles[i].bbox());}

		//MAKE ParticleOctree
		_particles.set_bbox(bbox);
		_particles.reserve(temp_particles.size()+_particles.size());

		for (size_t i=0; i<temp_particles.size(); i++)
		{
			_particles.push_back(temp_particles[i]);
		}
	}


	//save geometry to a text file as a rectangular prism of sampled points with the. voidspace=0, solidspace=1.
	template <typename Particle_t, size_t n_data>
	void Assembly<Particle_t, n_data>::save_geometry(const std::string filename, const Box_t &box, const size_t N[3]) const
	{
		//////////////// OPEN FILE ////////////////
		std::ofstream geofile(filename);

		if (not geofile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			geofile.close();
			return;
		}


		//////////////// WRITE FILE ////////////////
		std::stringstream buffer;

		//HEADER
		buffer << "nx= " << N[0] << std::endl;
		buffer << "ny= " << N[1] << std::endl;
		buffer << "nz= " << N[2] << std::endl;


		//DATA
		Point_t centroid {0,0,0};
		Point_t ijk {0,0,0};

		Point_t H = box.sidelength() / Point_t {N[0], N[1], N[2]};

		for (long unsigned int  k=0; k<N[2]; k++){
			ijk[2] = 0.5 + (double) k;
			for (long unsigned int  j=0; j<N[1]; j++){
				ijk[1] = 0.5 + (double) j;
				for (long unsigned int  i=0; i<N[0]; i++){
					ijk[0] = 0.5 + (double) i;
					centroid = box.low() + H*ijk;
					buffer << is_in_particle(centroid) << " "; //HYBGE NOTATION: FLUID=0, SOLID=1
				}
				buffer << std::endl;
			}
			buffer << std::endl;

			geofile << buffer.rdbuf();
			buffer.str("");
		}


		//////////////// CLOSE FILE ////////////////
		geofile.close();
	}


	//save geometry to a .vtk file as a rectangular prism of sampled points with the. voidspace=0, solidspace=1.
	template <typename Particle_t, size_t n_data>
	void Assembly<Particle_t, n_data>::save_solid(const std::string filename, const Box_t &box, const size_t N[3]) const
	{
		//////////////// OPEN FILE ////////////////
		std::ofstream meshfile(filename);

		if (not meshfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			meshfile.close();
			return;
		}


		//COMPUTE SPACING
		Point_t sample_point {0,0,0};
		Point_t ijk {0,0,0};

		Point_t H = box.sidelength() / Point_t{N[0],N[1],N[2]};


		//////////////// WRITE TO FILE ////////////////
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";

		//POINTS (CENTROIDS)
		buffer << "DATASET STRUCTURED_POINTS\n";
		buffer << "DIMENSIONS " << N[0]+1 << " " << N[1]+1 << " " << N[2]+1 << "\n";
		buffer << "ORIGIN " << box.low() << "\n";
		buffer << "SPACING " << H << "\n\n";

		meshfile << buffer.rdbuf();
		buffer.str("");


		//CELL_MARKERS (CENTROIDS OF CELLS)
		buffer << "CELL_DATA " << N[0]*N[1]*N[2] << "\n";
		buffer << "SCALARS cellMarkers integer\n";
		buffer << "LOOKUP_TABLE solid\n";
		for (long unsigned int  k=0; k<N[2]; k++)
		{
			ijk[2] = 0.5 + (double) k;
			for (long unsigned int  j=0; j<N[1]; j++)
			{
				ijk[1] = 0.5 + (double) j;
				for (long unsigned int  i=0; i<N[0]; i++)
				{
					ijk[0] = 0.5 + (double) i;
					sample_point = box.low() + H*ijk;
					buffer << is_in_particle(sample_point) << " ";
				}
			}
			buffer << "\n";
		}
		buffer << "\n";
		meshfile << buffer.rdbuf();
		buffer.str("");


		//POINT_MARKERS (VERTICES OF CELLS)
		buffer << "POINT_DATA " << (N[0]+1)*(N[1]+1)*(N[2]+1) << "\n";
		buffer << "SCALARS pointMarkers integer\n";
		buffer << "LOOKUP_TABLE solid\n";
		for (long unsigned int  k=0; k<=N[2]; k++)
		{
			ijk[2] = (double) k;
			for (long unsigned int  j=0; j<=N[1]; j++)
			{
				ijk[1] = (double) j;
				for (long unsigned int  i=0; i<=N[0]; i++)
				{
					ijk[0] = (double) i;
					sample_point = box.low() + H*ijk;
					buffer << is_in_particle(sample_point) << " ";
				}
			}
			buffer << "\n";
		}
		buffer << "\n";
		meshfile << buffer.rdbuf();
		buffer.str("");


		//LOOKUP TABLE
		buffer << "LOOKUP_TABLE solid 2\n";
		buffer << "0.5 0.5 0.5 0\n"; //solid
		buffer << "0.5 0.5 0.5 1\n"; //void
		buffer << "\n";
		meshfile << buffer.rdbuf();
		buffer.str("");


		//////////////// CLOSE FILE ////////////////
		meshfile.close();
	}


	//construct an unstructured voxel mesh of the region
	template <typename Particle_t, size_t n_data>
	void Assembly<Particle_t, n_data>::create_voxel_mesh_Q1(gv::mesh::VoxelMeshQ1 &out_mesh, const Box_t &box, const AssemblyMeshOptions &opts) const
	{
		out_mesh.set_bbox(Box_t{opts.scale*box.low(), opts.scale*box.high()});
		out_mesh.reserve((opts.N[0]+1)*(opts.N[1]+1)*(opts.N[2]+1));

		//COMPUTE SPACING
		Point_t H = box.sidelength()/Point_t(opts.N);

		//CONSTRUCT ELEMENTS
		for (size_t  k=0; k<opts.N[2]; k++)
		{
			for (size_t  j=0; j<opts.N[1]; j++)
			{
				for (size_t  i=0; i<opts.N[0]; i++)
				{
					//create element
					const Point_t low = box.low() + H*Point_t{i,j,k};
					const Point_t high = box.low() + H*Point_t{i+1,j+1,k+1};
					const Box_t element_box(low,high);
					bool add_element = false;
					int elem_marker;

					//get number of vertices contained in particles
					int n_vert = 0;
					for (int n=0; n<8; n++)
					{
						if (this->is_in_particle(element_box[n])) {n_vert += 1;}
					}

					//determine if the element should be added to the mesh and what marker it gets
					switch (n_vert)
					{
					case 0:
						if (opts.include_void)
						{
							// out_mesh.add_element(&new_elem);
							// out_mesh.elem_marker.push_back(opts.void_marker);
							add_element = true;
							elem_marker = opts.void_marker;
						}
						break;

					case 8:
						if (opts.include_solid)
						{
							// out_mesh.add_element(&new_elem);
							// out_mesh.elem_marker.push_back(opts.solid_marker);
							add_element = true;
							elem_marker = opts.solid_marker;
						}
						break;

					default:
						if (opts.include_interface)
						{
							//check centroid if needed
							if (opts.check_centroid)
							{	
							const Point_t centroid = element_box.center();
								bool is_solid = this->is_in_particle(centroid);
								if (is_solid and opts.include_solid)
								{
									// out_mesh.add_element(&new_elem);
									// out_mesh.elem_marker.push_back(opts.interface_marker);
									add_element = true;
									elem_marker = opts.interface_marker;
								}
								else if ((!is_solid) and opts.include_void)
								{
									// out_mesh.add_element(&new_elem);
									// out_mesh.elem_marker.push_back(opts.interface_marker);
									add_element = true;
									elem_marker = opts.interface_marker;
								}
							}
							else
							{
								// out_mesh.add_element(&new_elem);
								// out_mesh.elem_marker.push_back(opts.interface_marker);
								add_element = true;
								elem_marker = opts.interface_marker;
							}
						}
						break;
					}

					//add element to the mesh
					if (add_element)
					{
						Point_t elem[8];
						for (int i=0; i<8; i++) {elem[i] = opts.scale * element_box.voxelvertex(i);}
						out_mesh.add_element(elem);
						out_mesh.elem_marker.push_back(elem_marker);
					}
				}
			}
		}

		//print vertices
		// std::cout << "MESH NODES:\n";
		// for (size_t i=0; i<out_mesh.nNodes(); i++)
		// {
		// 	std::cout << i << ": " << out_mesh.node(i) << std::endl;
		// }
	}
}
