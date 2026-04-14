#pragma once

#include<vector>
#include<algorithm>

namespace gv::vmesh
{

	//each bilinear form will be represented by a matrix (TODO: add matrix free vector operations)
	//the matrix can be incrementally updated between mesh refinements (TODO: add this)
	//because of this, the bilinear form holds data for the matrix
	struct SparseRowEntry
	{
		uint64_t col;
		double   val;
		SparseRowEntry(uint64_t col, double val) : col(col), val(val) {}

		//comparisons for sorting the row by index
		inline constexpr bool operator==(const SparseRowEntry& other) const {
			return col==other.col;
		}

		inline constexpr auto operator<=>(const SparseRowEntry& other) const {
			return col<=>other.col;
		}

		inline SparseRowEntry& operator+=(const SparseRowEntry& other) {
			assert(col==other.col);
			val += other.val;
			return *this;
		}

		inline SparseRowEntry& operator+=(const double inc) {
			val += inc;
			return *this;
		}

		inline SparseRowEntry& operator*=(const double inc) {
			val *= inc;
			return *this;
		}
	};

	struct SparseRow
	{
		using Iter_t = std::vector<SparseRowEntry>::iterator;
		using CIter_t = std::vector<SparseRowEntry>::const_iterator;

		std::vector<SparseRowEntry> entries; 
		uint64_t row;
		SparseRow() : row(0) {}
		explicit SparseRow(uint64_t r) : row(r) {}
		
		//for accessing, the entries must be sorted
		inline void sort() {std::sort(entries.begin(), entries.end());}

		inline CIter_t lower(const SparseRowEntry& entry) const {
			return std::lower_bound(entries.begin(), entries.end(), entry);
		}

		inline Iter_t lower(const SparseRowEntry& entry) {
			return std::lower_bound(entries.begin(), entries.end(), entry);
		}

		//set to the identity row
		void set_eye() {entries.clear(); entries.emplace_back(row,1.0);}

		//read the matrix value
		double operator()(const uint64_t j) const {
			//assume sorted and accumulated
			Iter_t it = lower(SparseRowEntry{j,0.0});
			return (it->col == j) ? it->val : 0.0;
		}

		//insert and preserve sorting
		void insert(SparseRowEntry&& entry) {
			Iter_t it = lower(entry);
			if (it==entries.end()) {
				//new entry at end
				entries.push_back(std::move(entry));
			}
			else if (it->col == entry.col) {
				//accumulate existing entry
				it->val += entry.val;
			}
			else {
				//new interior entry (try to avoid this)
				entries.insert(it, std::move(entry));
			}
		}

		inline void append(SparseRowEntry&& entry) {entries.push_back(std::move(entry));}

		//we can add entries to the end and then accumulate all at once
		//TODO: write binary version (accumulate left/right and then join)
		void accumulate() {
			if (entries.empty()) {return;}

			std::sort(entries.start(), entries.end());
			Iter_t it = start; //write iterator
			for (Iter_t cur = start+1; cur != entries.end(); ++cur) {
				if (cur->col == it->col) {
					it->val  += cur->val;
					cur->val = 0.0;
				}
				else {
					++it; //done writing to this spot
					*it = *cur; //copy the read iterator into the next contiguous memory location
				}
			}
		}

		//reduce storage space
		inline void compress() {
			accumulate();
			entries.shrink_to_fit();
		}
	};


	//CSR structure
	struct CSR_storage
	{
		std::vector<SparseRow> rows;
		uint64_t n_rows, n_cols;
	};
}