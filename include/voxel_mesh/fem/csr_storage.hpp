#pragma once

#include<vector>
#include<algorithm>
#include<Eigen/SparseCore>

namespace gv::vmesh
{

	//each bilinear form will be represented by a matrix (TODO: add matrix free vector operations)
	//the matrix can be incrementally updated between mesh refinements (TODO: add this)
	//because of this, the bilinear form holds data for the matrix
	template<typename ColKey_t=uint64_t>
	struct SparseRowEntry
	{
		ColKey_t col;
		double   val;
		SparseRowEntry(ColKey_t col, double val) : col(col), val(val) {}

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


	//struct for storing rows of the Pseudo CSC matrix.
	template<typename RowKey_t = uint64_t, typename ColKey_t=uint64_t>
	struct SparseRow
	{
		using Entry_t = SparseRowEntry<ColKey_t>;
		using Iter_t  = std::vector<Entry_t>::iterator;
		using CIter_t = std::vector<Entry_t>::const_iterator;

		RowKey_t row_id;
		std::vector<Entry_t> entries; 
		SparseRow() : row_id{} {}
		explicit SparseRow(const RowKey_t r_id, uint64_t nz=0) : row_id{r_id} {entries.reserve(nz);}
		
		//for accessing, the entries must be sorted
		inline void sort() {std::sort(entries.begin(), entries.end());}

		inline CIter_t lower_bound(const Entry_t& entry) const {
			return std::lower_bound(entries.begin(), entries.end(), entry);
		}

		inline Iter_t lower_bound(const Entry_t& entry) {
			return std::lower_bound(entries.begin(), entries.end(), entry);
		}

		//insert and preserve sorting
		void insert(Entry_t&& entry) {
			Iter_t it = lower_bound(entry);
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

		//forward common vector operations
		inline void push_back(Entry_t&& entry) {entries.push_back(std::move(entry));}
		inline void push_back(Entry_t entry) {entries.push_back(entry);}
		inline void emplace_back(ColKey_t col, double val) {entries.emplace_back(col,val);}
		inline uint64_t size() const {return entries.size();}
		inline void shrink_to_fit() {entries.shrink_to_fit();}
		inline reserve(const uint64_t sz) {entries.reserve(sz);}


		//we can add entries to the end and then accumulate all at once
		//TODO: write binary version (accumulate left/right and then join)
		void accumulate() {
			if (entries.empty()) {return;}

			std::sort(entries.begin(), entries.end());
			Iter_t it = entries.begin(); //write iterator
			for (Iter_t cur = it+1; cur != entries.end(); ++cur) {
				if (cur->col == it->col) {
					it->val  += cur->val;
					cur->val = 0.0;
				}
				else {
					++it; //done writing to this spot
					*it = *cur; //copy the read iterator into the next contiguous memory location
				}
			}

			entries.erase(it+1, entries.end());
		}

		//comparison for sorting rows by their id
		constexpr bool operator==(const SparseRow& other) const {return row_id == other.row_id;}
		constexpr auto operator<=>(const SparseRow& other) const {return row_id <=> other.row_id;}
		constexpr bool operator==(const RowKey_t& other_r_id) const {return row_id == other_r_id;}
		constexpr auto operator<=>(const RowKey_t& other_r_id) const {return row_id <=> other_r_id;}

		//add two rows with the same id
		//this can be used as either a row sum (two different row ids) or as an accumulation
		//of a temporary row with the primary row in the csc matrix
		//the row will need to be accumulated/compressed after this
		inline void append(SparseRow<RowKey_t,ColKey_t>&& other) {
			entries.insert(entries.end(),
				std::make_move_iterator(other.begin()),
				std::make_move_iterator(other.end()));
		}
	};


	//Pseudo CSR structure - essentially COO in disguise
	//use as a starting point to build an Eigen CSR matrix
	//rows can be sorted by their RowKey_t (e.g, a uint64_t or similar such as a DOF or mesh feature key)
	//
	//The intended use is for each row i (test) dof, we have a SparseRow that records the non-zero values
	//of integral_D a(test_dof_i, trial_dof_j) over the entire domain.
	//During refinement and coarsening, we may not wish to lose this information. Additionally, which dofs
	//are active will change frequently. Rather than storing row information by a compressed dof number
	//(which is required for the linear solve), we index rows by the dof key (still effectively an integer).
	//Then as new dofs are activated, their row can be added and their contribution to the other rows more easily
	//incremented. Similarly, the column (trial) dofs can act as the index into each sparse row.
	template<typename RowKey_t=uint64_t, typename ColKey_t=uint64_t>
	struct CSR_COO
	{
		std::vector<SparseRow<RowKey_t>> rows;
		using Row_t      = SparseRow<RowKey_t,ColKey_t>;
		using Entry_t    = typename RowKey_t::Entry_t;
		using CRowIter_t = std::vector<Row_t>::const_iterator;
		using RowIter_t  = std::vector<Row_t>::iterator;

		//accumulate and compress the rows
		void accumulate_each_row() {
			for_each_row_omp([this](uint64_t r){rows[r].accumulate();})
		}

		//perform actions on the rows either in serial or parallel
		template<typename Action>
		void for_each_row_omp(Action&& action) {
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (uint64_t r=0; r<rows.size(); ++r) {
				action(rows[r]);
			}
		}

		template<typename Action>
		void for_each_row(Action&& action) {
			for (uint64_t r=0; r<rows.size(); ++r) {
				action(rows[r]);
			}
		}

		//sort the rows by their index
		inline void sort_rows() {
			std::sort(rows.begin(), rows.end());
		}

		//search for a row by its row ID
		inline CRowIter_t lower_bound(const RowKey_t r_id) const {
			return std::lower_bound(rows.begin(), rows.end(), r_id);
		}

		inline RowIter_t lower_bound(const RowKey_t r_id) {
			return std::lower_bound(rows.begin(), rows.end(), r_id);
		}

		//access a current row or create a new row and return a reference
		//the rows must be sorted for this to work. This is not always thread safe.
		//we assume the rows are sorted and preserve this.
		Row_t& row(const RowKey_t r_id) {
			RowIter_t it = lower_bound(r_id);
			if (it==rows.end()) {
				entries.emplace_back(r_id);
				return entries.back();
			}
			return *it;
		}

		//if we know that a new row needs to be created or are ok with
		//accumulating two rows with the same id later, we can quickly append a new row
		//and return a reference to it
		Row_t& new_row(const RowKey_t r_id) {
			entries.emplace_back(r_id);
			return entries.back();
		}


		//when many new rows have been added, we will need to combine them with any existing
		//rows that have the same id and ensure that they are sorted for lookups
		void join_rows() {
			if (rows.empty()) {return;}

			sort_rows();
			RowIter_t it = rows.begin(); //write iterator
			for (RowIter_t cur = start+1; cur != rows.end(); ++cur) {
				if (cur->row_id == it->row_id) {
					it->append(*cur);
				}
				else {
					++it; //done writing to this spot
					std::swap(*it,*cur); //copy the read iterator into the next contiguous memory location
				}
			}

			rows.erase(it+1, rows.end());
		}

		//accumulate all rows together to put the CSC into a valid state
		void accumulate() {
			join_rows();
			accumulate_each_row();
		}

		//remove extra reserved space
		void shrink_to_fit() {
			rows.shrink_to_fit();
			for_each_row_omp([this](Row_t& row) {row.shrink_to_fit();})
		}


		//given a selection of row and column dofs, build the Eigen CSR matrix
		//the structure must be in a valid state. call accumulate() first.
		//any re-numbering of the DOFs occur in the ordering of row_select and col_select.
		//row_select and col_select do not need to be sorted, but they must not contain
		//duplicate entries.
		Eigen::SparseMatrix<double,Eigen::RowMajor,int> to_eigen_csr(
			const std::vector<RowKey_t>& row_select, 
			const std::vector<ColKey_t>& col_select) const;
	};


	template<typename RowKey_t, typename ColKey_t>
	Eigen::SparseMatrix<double,Eigen::RowMajor,int> CSR_COO<RowKey_t,ColKey_t>::to_eigen_csr(
		const std::vector<RowKey_t>& row_select, 
		const std::vector<ColKey_t>& col_select) const
	{
		const int n_rows = static_cast<int>(row_select.size());
		const int n_cols = static_cast<int>(col_select.size());

		//map the compressed row (the index into row_select) into an integer index into the global rows
		auto global_row = [this](RowKey_t c_r) {
			CRowIter_t it = lower_bound(c_r);
			if (it == rows.end() || *it != c_r) {return -1;} //there are no entries for this row
			return static_cast<int>(std::distance(rows.begin(), it));
		}

		//initialize the matrix
		Eigen::SparseMatrix<double,Eigen::RowMajor,int> mat(n_rows,n_cols);
	}
}