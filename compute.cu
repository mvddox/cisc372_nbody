#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

// Thread block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// each thread will compute one element of acceleration matrix
__global__ void computePairwiseAccelsKernel(vector3* accels, double *d_mass, vector3 *d_hPos) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ vector3 shared_d_hPos_cols[BLOCK_SIZE_X];
	__shared__ vector3 shared_d_hPos_rows[BLOCK_SIZE_Y];
	__shared__ double shared_d_mass[BLOCK_SIZE_X];

	if (threadIdx.y == 0 && col < NUMENTITIES) {
		int k;
		for (k=0;k<3;k++) shared_d_hPos_cols[threadIdx.x][k] = d_hPos[col][k];
		shared_d_mass[threadIdx.x] = d_mass[col];
	}
	if (threadIdx.x == 0 && row < NUMENTITIES) {
		int k;
		for (k=0;k<3;k++) shared_d_hPos_rows[threadIdx.y][k] = d_hPos[row][k];
	}

	// all threads must wait until all arrive here. Otherwise there would be a race condition on the shared memory.
	__syncthreads();

	int accels_idx = (row * NUMENTITIES) + col;

	// thread will only do work if it's within bounds of the acceleration matrix
	if (row < NUMENTITIES && col < NUMENTITIES) {
		if (row==col) {
			FILL_VECTOR(accels[accels_idx],0,0,0);
		} else{
			vector3 distance;
			int k;
			for (k=0;k<3;k++) distance[k]=shared_d_hPos_rows[threadIdx.y][k]-shared_d_hPos_cols[threadIdx.x][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*shared_d_mass[threadIdx.x]/magnitude_sq;
			FILL_VECTOR(accels[accels_idx],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
		}
	}
}

// each thread will update the row's acceleration, velocity, and position values
__global__ void updateVelAndPosKernel(vector3* accels, vector3 *d_hPos, vector3 *d_hVel) {
	vector3 accel_sum={0,0,0};
	int row = blockIdx.x;

	int chunkSize = (NUMENTITIES + blockDim.x - 1)/blockDim.x;

	__shared__ vector3 accel_sums[BLOCK_SIZE_X];
	// thread will only do work if this is a valid row in the acceleration matrix
	if (row < NUMENTITIES){
		int j,k;
		int startPoint = threadIdx.x * chunkSize;
		for (j=startPoint;j< startPoint + chunkSize && j < NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[(row * NUMENTITIES) + j][k];
		}
		for (k=0;k<3;k++) accel_sums[threadIdx.x][k] = accel_sum[k];
	} else {
		return;
	}
	__syncthreads();
	vector3 final_sum={0,0,0};
	if (threadIdx.x == 0){
		int i,k;
		for (i = 0; i < blockDim.x; i++) {
			for (k=0;k<3;k++)
				final_sum[k] += accel_sums[i][k];
		}

		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			d_hVel[row][k]+=final_sum[k]*INTERVAL;
			d_hPos[row][k]+=d_hVel[row][k]*INTERVAL;
		}
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	// set up blocks and threads
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((NUMENTITIES + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (NUMENTITIES + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

	//first compute the pairwise accelerations.  Effect is on the first argument.
	computePairwiseAccelsKernel<<<dimGrid, dimBlock>>>(accels, d_mass, d_hPos);

	// set up blocks and threads for second kernel call
	dim3 dimBlock2(BLOCK_SIZE_X);
	dim3 dimGrid2(NUMENTITIES);

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	updateVelAndPosKernel<<<dimGrid2, dimBlock2>>>(accels, d_hPos, d_hVel);
}
