#include <iostream>
#ifndef SPACETIMEFD
	#include "SpaceTimeFD.hpp"
// #ifndef FD_TEMP
	// #include "FD_temp.hpp"
#endif

double IC_u(double x)
{
    double temp;
    if ((x + 0.25) >= 0) {
        temp += 1;
    }
    if ((x - 0.25) >= 0) {
        temp += 1;
    }
    return temp;
}

double IC_v(double x)
{
    return 0.0;
}

int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    // AMG_parameters AMG = {"", "FFC", 3, 100, 0.01, 6, 1, 0.1, 1e-6};
    AMG_parameters AMG = {1, "A", "FFC", 100, 10, 10, 0.25, 0.1, 0.0, 0.0, 1};
    const char* temp_prerelax = "A";
    const char* temp_postrelax = "A";

    double c = 1.0;
    int nt = 10;
    int nx = 10;
    int Pt = 1;
    int Px = 1;
    SpaceTimeFD matrix(MPI_COMM_WORLD, nt, nx, Pt, Px);
	// FD_temp matrix(MPI_COMM_WORLD, nt, nx, Pt, Px);
    matrix.Wave1D(IC_u, IC_v, c);


    MPI_Finalize();
    return 0;
}