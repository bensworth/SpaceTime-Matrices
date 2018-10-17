#include <iostream>
#include <fstream>
#include "CGdiffusion.hpp"
#include "DGadvection.hpp"
#include "mfem.hpp"
using namespace mfem;


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    // Parameters
    bool isTimeDependent = true;
    int numTimeSteps = 2;
    int refLevels = 1;
    int order = 1;
   	int dim = 2;
   	int use_gmres = 1;
    bool save_mat = false;
    double solve_tol = 1e-8;
    int print_level = 3;
    int timeDisc = 11;
    int spatialDisc = 1;
    int max_iter = 100;

    // AMG_parameters AMG = {"", "FFC", 3, 100, 0.01, 6, 1, 0.1, 1e-6};
    // const char* temp_prerelax = "";
    // const char* temp_postrelax = "FFC";
    AMG_parameters AMG = {"A", "A", 6, 6, 0.01, 6, -1, 0.1, 0};
    const char* temp_prerelax = "A";
    const char* temp_postrelax = "A";

    OptionsParser args(argc, argv);
    args.AddOption(&spatialDisc, "-s", "--spatial-disc",
                   "Spatial discretization (1=CG diffusion, 2=DG advection");
    args.AddOption(&timeDisc, "-t", "--time-disc",
                  "Time discretization (11=BDF1; 12=BDF2; 13=BDF3; 21=AM1; 22=AM2; 31=AB1; 32=AB2).");
    args.AddOption(&order, "-o", "--order",
                  "Finite element order.");
    args.AddOption(&refLevels, "-l", "--level",
                  "Number levels mesh refinement.");
    args.AddOption(&print_level, "-p", "--print-level",
                  "Hypre print level.");
    args.AddOption(&solve_tol, "-tol", "--solve-tol",
                  "Tolerance to solve linear system.");
    args.AddOption(&dim, "-d", "--dim",
                  "Problem dimension.");
    args.AddOption(&numTimeSteps, "-nt", "--num-time-steps",
                  "Number of time steps.");
    args.AddOption(&use_gmres, "-gmres", "--use-gmres",
                  "Boolean to use GMRES as solver (default with AMG preconditioning).");
    args.AddOption(&(AMG.distance_R), "-AIR", "--AIR-distance",
                  "Distance restriction neighborhood if using AIR (D<=0 implies P^TAP).");
    args.AddOption(&(AMG.interp_type), "-Ai", "--AMG-interpolation",
                  "Index for hypre interpolation routine.");
    args.AddOption(&(AMG.coarsen_type), "-Ac", "--AMG-coarsening",
                  "Index for hypre coarsening routine.");
    args.AddOption(&(AMG.strength_tolC), "-AsC", "--AMG-strengthC",
                   "Theta value determining strong connections for AMG (coarsening).");
    args.AddOption(&(AMG.strength_tolR), "-AsR", "--AMG-strengthR",
                   "Theta value determining strong connections for AMG (restriction).");
    args.AddOption(&(AMG.filterA_tol), "-Af", "--AMG-filter",
                  "Theta value to eliminate small connections in AMG hierarchy. Use -1 to specify O(h).");
    args.AddOption(&(AMG.relax_type), "-Ar", "--AMG-relaxation",
                  "Index for hypre relaxation routine.");
    args.AddOption(&temp_prerelax, "-Ar1", "--AMG-prerelax",
                  "String denoting prerelaxation scheme, e.g., A for all points.");
    args.AddOption(&temp_postrelax, "-Ar2", "--AMG-postrelax",
                  "String denoting postrelaxation scheme, e.g., FC for F relaxation followed by C relaxation.");
    args.Parse();
    AMG.prerelax = std::string(temp_prerelax);
    AMG.postrelax = std::string(temp_postrelax);
    if (AMG.prerelax.compare("N") == 0) AMG.prerelax = "";
    if (AMG.postrelax.compare("N") == 0) AMG.postrelax = "";
    if (!args.Good()) {
        if (rank == 0) {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        args.PrintOptions(std::cout);
    }

    // For now have to keep SpaceTime object in scope so that MPI Communicators
    // get destroyed before MPI_Finalize() is called. 
    if (spatialDisc == 1) {
        CGdiffusion STmatrix(MPI_COMM_WORLD, timeDisc, numTimeSteps, refLevels, order);
        // DGadvection STmatrix(MPI_COMM_WORLD, timeDisc, numTimeSteps, refLevels, order);
        STmatrix.BuildMatrix();
        STmatrix.SetAMGParameters(AMG);
        if (use_gmres) STmatrix.SolveGMRES(solve_tol, max_iter, print_level);
        else STmatrix.SolveAMG(solve_tol, max_iter, print_level);
    }
    else {
        DGadvection STmatrix(MPI_COMM_WORLD, timeDisc, numTimeSteps, refLevels, order);
        STmatrix.BuildMatrix();
        STmatrix.SetAMGParameters(AMG);
        if (use_gmres) STmatrix.SolveGMRES(solve_tol, max_iter, print_level);
        else STmatrix.SolveAMG(solve_tol, max_iter, print_level);
    }

    MPI_Finalize();
    return 0;
}