#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* matrix dimension type:
used to informate about data set
dimensions */
typedef struct {
	int nrow;
	int ncol;
	double** matrix;
} matrx;

inline matrx getDim(const char* fileDir) {
	int nrow = 1, ncol = 1;

	/*________________________________open_file_and_check_it________________________________*/
	FILE *file;
	errno_t err = fopen_s(&file, fileDir, "r");
	if (err != 0) printf("Error in getDim function: unable to open the file.\n");

	/*________get_data_dimensions________*/
	char c = fgetc(file);

	matrx Dim = { 1, 1 };
	while (c != '\n') {
		if (c == ',') Dim.ncol++;
		c = fgetc(file);
	}
	while (c != EOF) {
		if (c == '\n') Dim.nrow++;
		c = fgetc(file);
	}

	/*____final_formalities_and_result___*/
	fclose(file);
	return Dim;
}

inline void displayMatrix(matrx M) {

	for (int a = 0; a < M.nrow; a++) {
		for (int b = 0; b < M.ncol; b++) {
			//printf("%i ", M[a][b]); // without pointers
			printf("%.2f ", *(*(M.matrix + a) + b)); // with pointers
		}
		printf("\n");
	}
	printf("\n");
}

inline matrx readCSV(const char* fileDir) {

	/*___________________get_data_dimensions____________________*/
	matrx dataDim = getDim(fileDir);
	printf("Data dimensions: (%i, %i)\n\n", dataDim.nrow, dataDim.ncol);


	/*___________________variables_declaration__________________*/
	double** M;
	M = (double**)malloc(sizeof(double*) * dataDim.nrow);
	for (int row = 0; row < dataDim.nrow; row++) {
		M[row] = (double*)malloc(sizeof(double) * dataDim.ncol);
	}


	/*____________________open_file_and_check_it____________________*/
	FILE *fi;
	errno_t err = fopen_s(&fi, fileDir, "r");
	if (err != 0) printf("Error in readCSV function: unable to open file\n");


	/*________________csv_to_matrix_data_translation________________*/
	int i = 0, j = 0; //matrix positions
	int val_size = 2; // dynamic memory needed by val
	char* val = (char*)malloc(sizeof(char) * val_size);; // value to be passed to matrix
	char c; // character readed from data
	while (true) {
		c = fgetc(fi);
		switch (c) {
		case ',':
			*(*(M + i) + j) = (double)atof(val);
			j++;
			val_size = 2;
			val = (char*)realloc(val, sizeof(char) * val_size);
			break;
		case '\n':
			*(*(M + i) + j) = (double)atof(val);
			i++;
			j = 0;
			val_size = 2;
			val = (char*)realloc(val, sizeof(char) * val_size);
			break;
		case EOF:
			*(*(M + i) + j) = (double)atof(val);
			free(val);
			break;
		default:
			if (val_size == 2) {
				*val = c;
				*(val + 1) = '\0';
				val_size++;
			}
			else {
				val = (char*)realloc(val, sizeof(char) * val_size);
				*(val + val_size - 2) = c;
				*(val + val_size - 1) = '\0';
				val_size++;
			}
		}
		if (c == EOF) break;
	};

	fclose(fi);


	dataDim.matrix = M;
	return(dataDim);
}

__device__ inline double sum(double* vector, int length) {
	double result = 0;
	for (int i = 0; i < length; i++) {
		result = result + vector[i];
	}
	return result;
}

__device__ inline double sq(double value) {
	return value * value;
}

//inline matrx corr(matrx M) {
//
//	matrx result;
//	result.ncol = M.ncol;
//	result.nrow = M.ncol;
//	result.matrix = (double**)malloc(sizeof(double*) * M.ncol);
//
//	double* xy = (double*)malloc(sizeof(double) * M.nrow);
//	double*x = (double*)malloc(sizeof(double) * M.nrow);
//	double*y = (double*)malloc(sizeof(double) * M.nrow);
//	double*x_square = (double*)malloc(sizeof(double) * M.nrow);
//	double*y_square = (double*)malloc(sizeof(double) * M.nrow);
//
//
//	for (int i = 0; i < M.ncol; i++) {
//		result.matrix[i] = (double*)malloc(sizeof(double) * M.ncol);
//	}
//
//	for (int i = 0; i < M.ncol; i++) {
//		for (int j = i + 1; j < M.ncol; j++) {
//
//			// calculus of necessary elements for correlation computation
//			for (int element = 0; element < M.nrow; element++) {
//				x[element] = M.matrix[element][i];
//				x_square[element] = sq(M.matrix[element][i]);
//				y[element] = M.matrix[element][j];
//				y_square[element] = sq(M.matrix[element][j]);
//				xy[element] = M.matrix[element][i] * M.matrix[element][j];
//			}
//
//			int n = M.nrow;
//			//printf("corr(%i, %i) = %.4f\n", i, j, (n*sum(xy, n) - sum(x, n)*sum(y, n)) / (sqrt(n*sum(x_square, n) - sum(x, n)*sum(x, n))*sqrt(n*sum(y_square, n) - sum(y, n)*sum(x, n))));
//			//printf("(%i, %i) = %.4f\n", i, j, (sqrt(n*sum(x_square, n) - sum(x, n)*sum(x, n))*sqrt(n*sum(y_square, n) - sum(y, n)*sum(x, n))));
//			result.matrix[i][j] = (n*sum(xy, n) - sum(x, n)*sum(y, n)) / (sqrt(n*sum(x_square, n) - sq(sum(x, n)))*sqrt(n*sum(y_square, n) - sq(sum(y, n))));
//			result.matrix[j][i] = result.matrix[i][j];
//		}
//		result.matrix[i][i] = 1;
//	}
//
//	return result;
//}


inline void writeCSV(matrx M, const char* filedir) {
	FILE* f;
	errno_t err = fopen_s(&f, filedir, "w");
	if (err != 0) printf("Error in writeDim function: unable to open the file.\n");

	for (int i = 0; i < M.nrow; i++) {
		for (int j = 0; j < M.ncol; j++) {
			fprintf(f, "%f", M.matrix[i][j]);
			if (j != M.nrow - 1) fprintf(f, ",");
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

//__global__ void addKernel(int *c, const int *a, const int *b)
__global__ void corrKernel(double** Result, double** M, int nrow)
{
	int k = threadIdx.x;
	//c[i] = a[i] + b[i];

	double* xy = (double*)malloc(sizeof(double) * nrow);
	double*x = (double*)malloc(sizeof(double) * nrow);
	double*y = (double*)malloc(sizeof(double) * nrow);
	double*x_square = (double*)malloc(sizeof(double) * nrow);
	double*y_square = (double*)malloc(sizeof(double) * nrow);

	// calculus of necessary elements for correlation computation
	int n = nrow, i = 0, j = i + 1, addLength = n - 1;
	for (int element = 0; element < nrow; element++) {
		x[element] = M[element][i];
		x_square[element] = sq(M[element][i]);
		y[element] = M[element][j];
		y_square[element] = sq(M[element][j]);
		xy[element] = M[element][i] * M[element][j];
	}
	
	while (k <= n * (n - 1) / 2) {
		if (k < n - 1) {
			Result[i][j] = (n*sum(xy, n) - sum(x, n)*sum(y, n)) / (sqrt(n*sum(x_square, n) - sq(sum(x, n)))*sqrt(n*sum(y_square, n) - sq(sum(y, n))));
			Result[j][i] = Result[i][j];
			j++;
		}
		else {
			addLength--;
			n = n + addLength;
			i++;
			j = i + 1;
		}
	}

	for (int i = 0; i < n; i++) {
		Result[i][i] = 1;
	}
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t GPUcorr(matrx M, double **Result)
{

	double** dev_M;
	double** dev_Result;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((double***)&dev_M, M.nrow * sizeof(double*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for dev_M failed!");
		goto Error;
	}

	for (int i = 0; i < M.nrow; i++) {
		cudaMalloc((double**)&dev_M[i], M.ncol * sizeof(double));
	}

	cudaStatus = cudaMalloc((double***)&dev_Result, M.ncol * sizeof(double*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	for (int i = 0; i < M.ncol; i++) {
		cudaMalloc((double**)&dev_Result[i], M.ncol * sizeof(double));
	}

	/*cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}*/

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_M, M.matrix, M.nrow * sizeof(double*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	for (int i = 0; i < M.ncol; i++) {
		cudaMemcpy(dev_M[i], M.matrix[i], M.ncol * sizeof(double), cudaMemcpyHostToDevice);
	}

	cudaStatus = cudaMemcpy(dev_Result, Result, M.ncol * sizeof(double*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	for (int i = 0; i < M.ncol; i++) {
		cudaMemcpy(dev_Result[i], Result[i], M.ncol * sizeof(double), cudaMemcpyHostToDevice);
	}

	// Launch a kernel on the GPU with one thread for each element.
	//addKernel << <1, size >> >(dev_c, dev_a, dev_b);
	corrKernel << < 1, (M.ncol * (M.ncol - 1)) / 2 >> >(dev_Result, dev_M, M.nrow);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "corrKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Result, dev_Result, M.ncol * sizeof(double*), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	for (int i = 0; i < M.ncol; i++) {
		cudaMemcpy(Result[i], dev_Result[i], M.ncol * sizeof(double), cudaMemcpyDeviceToHost);
	}

Error:
	cudaFree(dev_M);
	cudaFree(dev_Result);

	return cudaStatus;
}