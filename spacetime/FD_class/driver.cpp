#include <iostream>
#ifndef SPACETIMEFD
	#include "SpaceTimeFD.hpp"
#endif

double IC_u(double x)
{
    double temp = 0;
    if ((x + 0.25) >= 0) {
        temp += 1;
    }
    if ((x - 0.25) >= 0) {
        temp -= 1;
    }
    // std::cout << "x = " << x << ", IC_u(x) = " << temp << "\n";
    return temp;
}

double IC_v(double x)
{
    return 0.0;
}


// AMG_parameters {
//     double distance_R;
//     std::string prerelax;
//     std::string postrelax;
//     int interp_type;
//     int relax_type;
//     int coarsen_type;
//     double strength_tolC;
//     double strength_tolR;
//     double filter_tolR;
//     double filter_tolA;
//     int cycle_type;
// };


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    // AMG_parameters AMG = {"", "FFC", 3, 100, 0.01, 6, 1, 0.1, 1e-6};
    AMG_parameters AMG = {1.5, "", "FA", 100, 10, 6, 0.25, 0.01, 0.0, 0.0, 1};
    const char* temp_prerelax = "A";
    const char* temp_postrelax = "A";

   /* Parse command line */
    double hx = -1;
    double dt = -1;
    double x0 = -1;
    double x1 = 1;
    double t0 = 0;
    double t1 = 1;
    double c = 1;
    int arg_index = 0;
    int nt_loc = 6;
    int nx_loc = 6;
    int Pt = numProcess;
    int Px = 1;

    while (arg_index < argc) {
        if ( strcmp(argv[arg_index], "-n") == 0 ) {
            arg_index++;
            nt_loc = atoi(argv[arg_index++]);
            nx_loc = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-P") == 0 ) {
            arg_index++;
            Pt = atoi(argv[arg_index++]);
            Px = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-dt") == 0 ) {
            arg_index++;
            dt = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-hx") == 0 ) {
            arg_index++;
            hx = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-CFL") == 0 ) {
            arg_index++;
            hx = atof(argv[arg_index++]);
        }
        else {
            arg_index++;
        }
    }

    // If user specifies dt, override default time interval [0,1]
    // int nt;
    // if (dt <= 0) {
    //     nt = nt_loc * Pt;
    //     dt = (t1 - t0) / float(nt);
    // }
    // else {
    //     nt = nt_loc * Pt;
    //     t1 = t0 + int(nt*dt);
    // }

    // //
    // int nx = nt;
    // if (hx <= 0) {
    //     nt = nt_loc * Pt;
    //     dt = (t1 - t0) / float(nt);
    // }
    // else {
    //     nt = nt_loc * Pt;
    //     t1 = t0 + int(nt*dt);
    // }


    SpaceTimeFD matrix(MPI_COMM_WORLD, nt_loc, nx_loc, Pt, Px, -1, 1);
    matrix.Wave1D(IC_u, IC_v, c);
    matrix.SetAMGParameters(AMG);
    // matrix.SolveAMG();
    matrix.SaveRHS("test_b");

		matrix.SaveMatrix("A_hypre");

    MPI_Finalize();
    return 0;
}