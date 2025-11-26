#pragma once

#include "mesh/mesh_util.hpp"

#include <iostream>
#include <iomanip>
#include <string>

#include <vector>
#include <array>
#include <cassert>

#include <atomic>
#include <mutex>

namespace gv::mesh
{

	/////////////////////////////////////////////////
	/// Coloring methods
	/////////////////////////////////////////////////
	enum ColorMethod {
		GREEDY,   //first available color
		BALANCED, //available color with minimal count
	};

	/////////////////////////////////////////////////
	/// Function to print coloring methods
	/////////////////////////////////////////////////
	std::ostream& operator<<(std::ostream &os, const ColorMethod method) {
		switch (method) {
			case ColorMethod::GREEDY:   return os << "GREEDY";
			case ColorMethod::BALANCED: return os << "BALANCED";
			default:                    return os << "UNKNOWN";
		}
	}


	/////////////////////////////////////////////////
	/// This class is to be used to color a TopologicalMesh. There may be paralel mesh operations and the read/writes
	/// should be carefully managed.
	///
	/// @tparam ColorMethod The method that will be used to color the elements
	/// @tparam Element_t The type of element that will be used
	/////////////////////////////////////////////////
	template <ColorMethod COLOR_METHOD, ColorableMeshElement Element_t, size_t MAX_COLORS>
	class MeshColorManager {
	public:
		MeshColorManager() {
			for (size_t c=0; c<MAX_COLORS; c++) {_counts[c]=0;}
		}
		MeshColorManager(std::vector<Element_t> &_elements) : _elements(_elements) {
			for (size_t c=0; c<MAX_COLORS; c++) {_counts[c]=0;}
		}
		
	private:
		mutable std::mutex                          _mutex;
		std::vector<Element_t>                     &_elements;
		std::array<std::atomic<size_t>, MAX_COLORS> _counts;

		static constexpr ColorMethod color_method = COLOR_METHOD;
		template<ColorMethod C, ColorableMeshElement E, size_t M>
		friend std::ostream& operator<<(std::ostream &os, const MeshColorManager<C,E,M> &manager);
	public:
		size_t colorCount(const size_t c) const {return _counts[c];}
		size_t nColors() const {
			for (size_t c=0; c<MAX_COLORS; c++) {
				if (_counts[c]==0) {return c;}
			}
			return MAX_COLORS;
		}

		void decrementCount(const size_t color, const size_t count=1) {
			assert(_counts[color]>=count);
			_counts[color]-=count;
		}


		void incrementCount(const size_t color, const size_t count=1) {
			_counts[color]+=count;
		}

		/////////////////////////////////////////////////
		/// Get a valid color for the specified element with the specified neighbors.
		/// This method has no thread protection.
		///
		/// @param elem_idx The index in _elements of the element to be colored
		/// @param neighbors The neighbors that cannot share a color with the element. It is assumed that the neighbors are already colored.
		/////////////////////////////////////////////////
		size_t getColor_Unlocked(const size_t elem_idx, const std::vector<size_t> &neighbors) const {
			//The method that calls this has the responsibility of ensuring that _elements[elem_idx] and _elements[neighbors[i]] are readable.
			//The method that calls this has the responsibility of ensuring that _colors[] does not change (if that is necessary)
			//Note that _counts[] is atomic so reading and incrementing will not fail, but there may be read race conditions.

			//no colors
			if (_counts[0]==0) {return 0;}

			//decide which colors are allowed
			std::array<bool, MAX_COLORS> color_allowed;
			color_allowed.fill(true);
			for (size_t e_idx : neighbors) {
				size_t neighbor_color = _elements[e_idx].color;
				if (neighbor_color < MAX_COLORS) {
					color_allowed[neighbor_color] = false;
				} //note that is is sometimes possible that neighbors are uncolored. in this case neighbor.color=(size_t) -1.
			}

			//get the color
			if constexpr (COLOR_METHOD==ColorMethod::GREEDY) {
				for (size_t c=0; c<MAX_COLORS; c++) {
					if (color_allowed[c]) {return c;}
				}
				throw std::runtime_error("Ran out of colors (MAX_COLORS= " + std::to_string(MAX_COLORS) + ")");
			} else if constexpr (COLOR_METHOD==ColorMethod::BALANCED) {
				//get the color with minimal count (approximate due to race conditions)
				size_t color_count = (size_t) -1;
				size_t color = MAX_COLORS;

				for (size_t c=0; c<MAX_COLORS; c++) {
					if (_counts[c]==0 and color_count < (size_t) -1) {return color;} //end of used colors
					if (color_allowed[c] and _counts[c]<color_count) {
						color  = c;
						color_count = _counts[c];
					}
				}
				return color;
			} else {
				throw std::runtime_error("Unknown COLOR_METHOD: " + std::to_string(COLOR_METHOD) +
						" (valid: " + std::to_string(ColorMethod::GREEDY) + ", " + std::to_string(ColorMethod::BALANCED) + ")");
			}
			
			return 0;
		}


		/////////////////////////////////////////////////
		/// Color for the specified element with the specified neighbors.
		/// This method is locked to run on a single thread at a time.
		///
		/// @param elem_idx The index in _elements of the element to be colored
		/// @param neighbors The neighbors that cannot share a color with the element. It is assumed that the neighbors are already colored.
		/////////////////////////////////////////////////
		void setColor_Locked(const size_t elem_idx, const std::vector<size_t> &neighbors) {
			std::lock_guard<std::mutex> lock(_mutex);

			size_t this_color = getColor_Unlocked(elem_idx, neighbors);
			_elements[elem_idx].color = this_color;
			_counts[this_color]++;
		}


		/////////////////////////////////////////////////
		/// Color for the specified element with the specified neighbors.
		/// This method can be called in many threads so long as reading _elements[neighbors[]] is safe 
		///     and writing to _elements[elem_idx] is safe.
		///
		/// Reading and writing from _count is atomic, so there may unoptimal coloring sometimes.
		///
		/// @param elem_idx The index in _elements of the element to be colored
		/// @param neighbors The neighbors that cannot share a color with the element. It is assumed that the neighbors are already colored.
		/////////////////////////////////////////////////
		void setColor_Unlocked(const size_t elem_idx, const std::vector<size_t> &neighbors) {
			const size_t old_color = _elements[elem_idx].color;
			if (old_color<MAX_COLORS and _counts[old_color]>0) {_counts[old_color]--;}

			size_t this_color = getColor_Unlocked(elem_idx, neighbors);
			_elements[elem_idx].color = this_color;
			_counts[this_color]++;

			//check if the color is valid
			bool is_valid = true;
			for (size_t n : neighbors) {
				if (this_color == _elements[n].color) {
					is_valid = false;
					break;
				}
			}

			//re-run until the color is valid
			if (!is_valid) {
				setColor_Unlocked(elem_idx, neighbors);
				std::cout << "Race condition when coloring element " << elem_idx << " (neighbor colors changed). Re-coloring.\n";
			}
		}
	};

	///////////////////////////////
	/// Function to print color information.
	///////////////////////////////
	template<ColorMethod C, ColorableMeshElement E, size_t M>
	std::ostream& operator<<(std::ostream &os, const MeshColorManager<C,E,M> &manager) {
		os << "\n" << std::string(50, '=') << "\n" << "Color Summary\n" << std::string(50, '-') << "\n";

		os << std::setw(16) << std::left << "Method" << std::setw(10) << std::right << manager.color_method << "\n";
		
		const size_t nColors = manager.nColors();
		os << std::setw(16) << std::left << "nColors " << std::setw(10) << std::right << nColors << "\n";



		size_t nColoredElements = 0;
		size_t maxColor = 0;
		for (size_t c = 0; c < nColors; c++) {
			nColoredElements+=manager._counts[c];
			maxColor = std::max(maxColor, manager._counts[c].load());
		}
		os << std::setw(16) << std::left << "nElements" << std::setw(10) << std::right << nColoredElements << "\n";

		os << std::string(50, '-') << "\n";

		os << std::right 
		   << std::setw(6)  << "Color"
		   << std::setw(10) << "Count"
		   << std::setw(10) << "Percent"
		   << "\n";

		double max_percent = 100.0 * (double) maxColor / (double) nColoredElements;
		for (size_t c = 0; c < nColors; c++) {
			double percent = 100.0 * (double) manager._counts[c] / (double) nColoredElements;
			os << std::right << std::setw(6) << c
			   << std::setw(10) << manager._counts[c] << std::setw(2) << ""
			   << std::setw(8) << std::fixed << std::setprecision(2) << percent;
			os << std::right << std::setw(2) << "";
			
			int bar_characters = 20.0 * percent/max_percent;
			os << std::left  << std::string(bar_characters, '|') << "\n";
		}

		os << std::string(50, '-') << "\n";
		return os;
	}
}