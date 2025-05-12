#pragma once

#include <vector>
#include <string>
// #include <qt5/QtGui/QImage>
// #include <qt5/QtCore/QString>

#include <omp.h>
#include <Eigen/SparseCore>


namespace gv::fem
{
	///set dirichlet BC in an integrating matrix (set specified rows to 0 with a 1 on the main diagonal)
	void set_dirichlet_bc(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat, const std::vector<size_t> &nodes)
	{
		//prepare matrix to shorten loops. note for an integrating matrix the diagonals will never be 0 and will never be cleared here.
		mat.makeCompressed();

		//loop through rows and zero them out
		#pragma omp parallel
		for (size_t n_idx=0; n_idx<nodes.size(); n_idx++)
		{
			//get start of row storage
			size_t row_idx   = nodes[n_idx];
			size_t idx_start = mat.outerIndexPtr()[row_idx];
			size_t idx_stop  = mat.outerIndexPtr()[row_idx+1]; //this stop index only works when mat is compressed

			//zero out row
			for (size_t j=idx_start; j<idx_stop; j++)
			{
				mat.valuePtr()[j] = 0.0;
			}
		}

		//loop through rows and set main diagonal entry to 1
		// #pragma omp parallel
		for (size_t n_idx=0; n_idx<nodes.size(); n_idx++)
		{
			size_t row_idx = nodes[n_idx];
			mat.coeffRef(row_idx, row_idx) = 1.0;
		}

		//compress matrix to free storage
		mat.prune(0.0);
		mat.makeCompressed();
	}


	///save matrix to bmp
	// void save_as_bmp(const Eigen::SparseMatrix<double>& mat, const QString& filename)
	// {
	// 	const int rows = mat.rows();
	//     const int cols = mat.cols();

	//     //create a QImage with white background
	//     QImage image(cols, rows, QImage::Format_RGB32);
	//     image.fill(Qt::white);

	//     //loop over non-zeros and set pixels to black
	//     for (Eigen::Index k = 0; k < mat.outerSize(); ++k)
	//     {
	//         for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
	//         {
	//             int row = it.row();   // row index
	//             int col = it.col();   // column index
	//             image.setPixel(col, row, qRgb(0, 0, 0));
	//         }
	//     }

	//     //save the image
	//     image.save(filename);
	// }


}
