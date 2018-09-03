
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <stdlib.h>
#include "GPUcorrelationsFunctions.h"

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



int main()
{
	/*_______reading_data_____________*/
	matrx M = readCSV("C:\\Users\\AVSM2\\Documents\\datos.txt");
	displayMatrix(readCSV("C:\\Users\\AVSM2\\Documents\\datos.txt"));

	///*_______correlation_matrix_______*/
	//matrx cors = corr(M);
	//printf("\n");
	//printf("Correlation Matrix:\n\n");
	//displayMatrix(cors);

	/*_______save_correlation_matrix__*/
	//writeCSV(cors, "C:\\Users\\AVSM2\\Documents\\datos_correlation_matrix_GPU.txt");

	double** Result;
	Result = (double**)malloc(sizeof(double*) * M.ncol);
	for (int row = 0; row < M.nrow; row++) {
		Result[row] = (double*)malloc(sizeof(double) * M.ncol);
	}


    // Correlations in parallel.
    //cudaError_t cudaStatus = GPUcorr(c, a, b, arraySize);
	cudaError_t cudaStatus = GPUcorr(M, Result);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPUcorr failed!");
        return 1;
    }
	
	matrx matrx_Result = { M.ncol, M.ncol, Result };
	printf("lalalala:\n\n");
	displayMatrix(matrx_Result);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


