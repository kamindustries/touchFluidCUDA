/* 
	CUDA kernels and functions
	Kurt Kaminski 2016
*/


#ifndef __FLUID_SIM__
#define __FLUID_SIM__

#include <iostream>
#include <map>
#include <string>

#include "FluidKernels.cu"

// Data
float *chemA, *chemA_prev, *laplacian;
float *vel[2], *vel_prev[2];
float *pressure, *pressure_prev;
float *temperature, *temperature_prev;
float *divergence;
float *newObstacle; //incoming obstacles and velocities
float *mouse, *mouse_old;
float *res, *globals, *advectionConstants;

// Global constants
dim3 grid, threads;
int dimX, dimY;
float dt;
int nDiff;
int nReact;
int nJacobi;

// Advection constants
float velDiff;
float tempDiff;
float densDiff;
float curlAmt;
float buoy;
float weight;
float diff;
float visc;
float force;
float source_density;
float source_temp;

///////////////////////////////////////////////////////////////////////////////
// Set constants from UI
///////////////////////////////////////////////////////////////////////////////
void setFluidConstants()
{
	// globals[] = {dt, nDiff, nReact, nJacobi}
	dt = globals[0];
	nDiff = (int)globals[1];
	nReact = (int)globals[2];
	nJacobi = (int)globals[3];

	// advectionConstants[] = {velDiff, tempDiff, densDiff, curl, buoyancy, weight}
	velDiff = advectionConstants[0];
	tempDiff = advectionConstants[1];
	densDiff = advectionConstants[2];
	curlAmt = advectionConstants[3];
	buoy = advectionConstants[4];
	weight = advectionConstants[5];

}

///////////////////////////////////////////////////////////////////////////////
// Clear arrays
///////////////////////////////////////////////////////////////////////////////
void clearArrays()
{
	cudaSetDevice(0);

	for (int i=0; i<2; i++){
	  ClearArray<<<grid,threads>>>(vel[i], 0.0, dimX, dimY);
	  ClearArray<<<grid,threads>>>(vel_prev[i], 0.0, dimX, dimY);
	}

	ClearArray<<<grid,threads>>>(chemA, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(chemA_prev, 0.0, dimX, dimY);

	ClearArray<<<grid,threads>>>(pressure, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(pressure_prev, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(temperature, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(temperature_prev, 0.0, dimX, dimY);
	ClearArray<<<grid,threads>>>(divergence, 0.0, dimX, dimY);

	printf("Fluid::clearArrays(): Cleared GPU arrays.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Set up global variables
///////////////////////////////////////////////////////////////////////////////
void initGlobals(std::map<std::string, const TCUDA_ParamInfo*> &nodes)
{
	// Resolution
	res = (float*)malloc(sizeof(float)*nodes["fluidRes"]->chop.numChannels);
	res = (float*)nodes["fluidRes"]->data;
	
	dimX = res[0];
	dimY = res[1];

	threads = dim3(8,8);
	grid.x = (dimX + threads.x - 1) / threads.x;
	grid.y = (dimY + threads.y - 1) / threads.y;

	printf("Threads dim: %d x %d\n", threads.x, threads.y);
	printf("Grid dim: %d x %d\n", grid.x, grid.y);

	// Allocate mouse array
	mouse = (float*)malloc(sizeof(float)*nodes["mouse"]->chop.numChannels);
	mouse_old = (float*)malloc(sizeof(float)*nodes["mouse"]->chop.numChannels);

	// Local mouse pointer points to CHOP node
	mouse = (float*)nodes["mouse"]->data;
	for (int i = 0; i < nodes["mouse"]->chop.numChannels; i++){
		mouse_old[0]=mouse[1];
	}

	// Allocate arrays for local constants
	globals = (float*)malloc(sizeof(float)*nodes["globals"]->chop.numChannels);
	advectionConstants = (float*)malloc(sizeof(float)*nodes["advection"]->chop.numChannels);

	// Local constants pointers points to CHOP nodes
	globals = (float*)nodes["globals"]->data;
	advectionConstants = (float*)nodes["advection"]->data;

	// Set fluid constants
	setFluidConstants();

	printf("initParameters(): done.\n");
}

///////////////////////////////////////////////////////////////////////////////
// Initialize memory
///////////////////////////////////////////////////////////////////////////////
void initMemory(float* inObstacle)
{
	int size = dimX * dimY;

	// Allocate GPU memory
	cudaMalloc((void**)&chemA, sizeof(float)*size);
	cudaMalloc((void**)&chemA_prev, sizeof(float)*size);
	cudaMalloc((void**)&laplacian, sizeof(float)*size);

	for (int i=0; i<2; i++){
		cudaMalloc((void**)&vel[i], sizeof(float)*size);
		cudaMalloc((void**)&vel_prev[i], sizeof(float)*size);
	}

	cudaMalloc((void**)&pressure, sizeof(float)*size );
	cudaMalloc((void**)&pressure_prev, sizeof(float)*size );
	cudaMalloc((void**)&temperature, sizeof(float)*size );
	cudaMalloc((void**)&temperature_prev, sizeof(float)*size );
	cudaMalloc((void**)&divergence, sizeof(float)*size );


	clearArrays();
}

///////////////////////////////////////////////////////////////////////////////
// Initialize
///////////////////////////////////////////////////////////////////////////////
void init(const TCUDA_ParamInfo **_params, const TCUDA_ParamInfo *_output, std::map<std::string, const TCUDA_ParamInfo*> &nodes)
{
	initGlobals(nodes);
	initMemory((float*)nodes["boundary"]->data);

	printf("Fluid::init(): Dimensions = %d x %d --\n", dimX, dimY);
	printf("Fluid::init(): Allocated GPU memory.\n");
}


void getFromUI(float* inDensity, float* inObstacle)
{
	// Apply incoming density and temperature
	// bgra == 0,1,2,3
	AddFromUI<<<grid,threads>>>(temperature_prev, inDensity, 1, dt, dimX, dimY);
	AddFromUI<<<grid,threads>>>(chemA_prev, inDensity, 2, dt, dimX, dimY);
	
	// Apply obstacle velocity
	AddObstacleVelocity<<<grid,threads>>>(vel_prev[0], vel_prev[1], inObstacle, dt, dimX, dimY);

}

///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
void step(float* inDensity, float* inObstacle)
{
	setFluidConstants();

	// Velocity advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], vel_prev[0], vel_prev[1],
								inObstacle, vel[0], vel[1], 
								dt, velDiff, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	// Temperature advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, inObstacle, temperature,
							dt, tempDiff, dimX, dimY);
	SWAP(temperature_prev, temperature);


	//Vorticity Confinement
	vorticityConfinement<<<grid,threads>>>( vel[0], vel[1], vel_prev[0], vel_prev[1], inObstacle, 
											curlAmt, dt, dimX, dimY);
		
	float Tamb = 0.0;
	ApplyBuoyancy<<<grid,threads>>>(vel_prev[0], vel_prev[1], temperature_prev, chemA,
									vel[0], vel[1], Tamb, buoy, weight, dt, dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

	//// Apply impulses
	getFromUI(inDensity, inObstacle);

	// Density advection
	Advect<<<grid,threads>>>(vel_prev[0], vel_prev[1], chemA_prev, inObstacle, chemA, 
							dt, densDiff, dimX, dimY);
	SWAP ( chemA_prev, chemA );

	// Compute divergence
	ComputeDivergence<<<grid,threads>>>( vel_prev[0], vel_prev[1], inObstacle, divergence, dimX, dimY );

	// Pressure solve
	ClearArray<<<grid,threads>>>(pressure_prev, 0.0, dimX, dimY);
	for (int i=0; i<nJacobi; i++){
		
		Jacobi<<<grid,threads>>>(pressure_prev, divergence, inObstacle, pressure, dimX, dimY);
		SWAP(pressure_prev, pressure);

	}

	// Subtract pressure gradient from velocity
	SubtractGradient<<<grid,threads>>>( vel_prev[0], vel_prev[1], pressure_prev, inObstacle, 
										vel[0], vel[1], dimX, dimY);
	SWAP(vel_prev[0], vel[0]);
	SWAP(vel_prev[1], vel[1]);

}

void makeColor(float* output)
{
	MakeColor<<<grid,threads>>>(vel[0], vel[1], chemA, temperature, 
								output, dimX, dimY);
}

#endif