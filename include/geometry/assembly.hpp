#pragma once


#include "util/point.hpp"
#include "util/octree.hpp"
#include "util/box.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"

#include "util/view_octree.hpp" //for viewing octree structure
#include "mesh/Q1.hpp" //unstructured voxel mesh

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

namespace gv::geometry{

	//class for storing octrees efficiently
	template <typename Particle_t, size_t n_data=8>
	class ParticleOctree : public gv::util::BasicOctree<Particle_t, 3, true, n_data> {
	public:
		ParticleOctree() : gv::util::BasicOctree<Particle_t, 3, true, n_data>() {}
		ParticleOctree(const gv::util::Box<3> &bbox) : gv::util::BasicOctree<Particle_t, 3, true, n_data>(bbox) {}

		//check if a point is in any particle
		bool is_in_particle(const gv::util::Point<3,double> &point) const
		{
			const auto* node = this->getnode(point);
			for (size_t i=0; i<node->cursor; i++)
			{
				if (this->_data[node->data_idx[i]].contains(point)) {return true;}
			}
			return false;
		}

	private:
		bool is_data_valid(gv::util::Box<3> const &box, Particle_t const &P) const override {return gv::geometry::collides_GJK(box,P);}
	};


	

	//class for interacting with ParticleOctree
	template <typename Particle_t, size_t n_data=8>
	class Assembly
	{
	public:
		Assembly() {};
		Assembly(const std::string filename, const std::string columns) {readfile(filename, columns);}

		//check if a point is in any particle
		bool is_in_particle(const gv::util::Point<3,double> &point) const
		{	
			if (not _particles.bbox().contains(point)) {return false;}
			return _particles.is_in_particle(point);
		}

		//read particles from file with specified format. TODO: read format from start of file.
		void readfile(const std::string filename, const std::string columns);

		//get bounding box
		gv::util::Box<3> bbox() const {return _particles.bbox();}

		//save geometry to a text file as a rectangular prism of sampled points with the. voidspace=0, solidspace=1.
		void save_geometry(const std::string filename, const gv::util::Box<3> &box, const size_t N[3]) const;
		void save_geometry(const std::string filename, const size_t  N[3]) const {save_geometry(filename, this->_particles.bbox(), N);}
		void save_solid(const std::string filename, const gv::util::Box<3> &box, const size_t N[3]) const;
		void save_solid(const std::string filename, const size_t  N[3]) const {save_solid(filename, this->_particles.bbox(), N);}

		//view octree structure of _particles octree
		void view_octree_vtk(const std::string filename="octree_structure.vtk") const {gv::util::view_octree_vtk(_particles, filename);}

		//create unstructured voxel mesh of the entire region. mark elements with 0 for void, 1 for solid, and 2 for interface
		void create_voxel_mesh_Q1(gv::mesh::VoxelMeshQ1 &out_mesh, const gv::util::Box<3> &box, const size_t N[3]) const;
		void create_voxel_mesh_Q1(gv::mesh::VoxelMeshQ1 &out_mesh, const size_t N[3]) const {create_voxel_mesh_Q1(out_mesh, this->_particles.bbox(), N);}

	private:
		ParticleOctree<Particle_t, n_data> _particles;
	};





	//add particles from a file to the current assembly. TODO: split into smaller functions.
	template <typename Particle_t, size_t n_data>
	void Assembly<Particle_t, n_data>::readfile(const std::string filename, const std::string columns)
	{
		std::cout << "reading " << filename << std::endl;
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

				Particle_t P(gv::util::Point<3,double> {rx,ry,rz}, eps, gv::util::Point<3,double> {x,y,z}, gv::util::Quaternion<double>(qw,-qx,-qy,-qz));
				temp_particles.push_back(P);
			}
		}

		//ADD EXISTING PARTICLES TO LIST
		for (size_t i=0; i<_particles.size(); i++) {temp_particles.push_back(_particles[i]);}

		//GET BOUNDING BOX SIZE
		gv::util::Box<3> bbox = temp_particles[0].bbox();
		for (size_t i=0; i<temp_particles.size(); i++) {bbox.combine(temp_particles[i].bbox());}

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
	void Assembly<Particle_t, n_data>::save_geometry(const std::string filename, const gv::util::Box<3> &box, const size_t N[3]) const
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
		gv::util::Point<3,double> centroid {0,0,0};
		gv::util::Point<3,double> ijk {0,0,0};

		gv::util::Point<3,double> H = box.high()-box.low();
		H[0]/= (double) N[0];
		H[1]/= (double) N[1];
		H[2]/= (double) N[2];

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
	void Assembly<Particle_t, n_data>::save_solid(const std::string filename, const gv::util::Box<3> &box, const size_t N[3]) const
	{
		//////////////// OPEN FILE ////////////////
		std::ofstream meshfile(filename);

		if (not meshfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			meshfile.close();
			return;
		}


		//COMPUTE SPACING
		gv::util::Point<3,double> sample_point {0,0,0};
		gv::util::Point<3,double> ijk {0,0,0};

		gv::util::Point<3,double> H = box.high()-box.low();
		H[0]/= (double) N[0];
		H[1]/= (double) N[1];
		H[2]/= (double) N[2];


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
	void Assembly<Particle_t, n_data>::create_voxel_mesh_Q1(gv::mesh::VoxelMeshQ1 &out_mesh, const gv::util::Box<3> &box, const size_t N[3]) const
	{
		out_mesh.set_bbox(1.0625*box);
		out_mesh.reserve(N[0]*N[1]*N[2]);

		//COMPUTE SPACING
		gv::util::Point<3,double> sample_point {0,0,0};
		gv::util::Point<3,double> ijk {0,0,0};
		gv::util::Point<3,double> origin = box.low();
		gv::util::Point<3,double> H = box.high()-box.low();
		H[0]/= (double) N[0];
		H[1]/= (double) N[1];
		H[2]/= (double) N[2];


		//CONSTRUCT POINTS
		for (long unsigned int  k=0; k<=N[2]; k++)
		{
			ijk[2] = (double) k;
			for (long unsigned int  j=0; j<=N[1]; j++)
			{
				ijk[1] = (double) j;
				for (long unsigned int  i=0; i<=N[0]; i++)
				{
					ijk[0] = (double) i;
					out_mesh.add_node(origin + H*ijk);
				}
			}
		}


		//CONSTRUCT ELEMENTS
		for (long unsigned int  k=0; k<N[2]; k++)
		{
			ijk[2] = (double) k;
			for (long unsigned int  j=0; j<N[1]; j++)
			{
				ijk[1] = (double) j;
				for (long unsigned int  i=0; i<N[0]; i++)
				{
					ijk[0] = (double) i;
					// std::cout << "start element" << std::endl;

					//create element
					size_t new_elem[8];
					gv::util::Box<3> element_box(origin+H*ijk,origin+H*ijk+H);
					for (int n=0; n<8; n++)
						{
							new_elem[n] = out_mesh.node_idx(element_box.voxelvertex(n));
							if (new_elem[n] == (size_t) -1)
							{
								std::cout << "element (" << i << ", " << j << ", " << k << "): " << n << ": " << element_box.voxelvertex(n) << std::endl;
								
							}
						}



					//count number of verticies contained in particles
					// std::cout << "count vertices" << std::endl;
					int n_vert = 0;
					for (int n=0; n<8; n++)
					{
						// std::cout << n << ": " << new_elem[n] << std::endl;
						if (this->is_in_particle(element_box[n])) {n_vert += 1;}
					}

					//add element to mesh
					// std::cout << "add element" << std::endl;
					out_mesh.add_element(new_elem);

					// std::cout << "mark element" << std::endl;
					if (n_vert==0) {out_mesh.elem_marker.push_back(0);}
					else if (n_vert==8) {out_mesh.elem_marker.push_back(1);}
					else {out_mesh.elem_marker.push_back(2);}

					// std::cout << "element (" << i << ", " << j << ", " << k << "): " << n_vert << std::endl; 
				}
			}
		}
	}
}
