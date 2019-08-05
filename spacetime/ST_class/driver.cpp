#include <iostream>
#include <fstream>
#include "CGdiffusion.hpp"
#include "DGadvection.hpp"
#include "mfem.hpp"
using namespace mfem;


// Sample command line:
//  srun -n 240 ./driver -s 2 -l 6 -nt 40 -t 31 -Ar 10 -AsR 0.2 -AsC 0.25 -AIR 1 -dt 0.0025 -lump 0
// mpirun -np 4 ./driver -s 1 -l 2 -nt 12 -t 11 -Ar 10 -AsR 0.2 -AsC 0.25 -AIR 1 -dt 0.01 -lump 0

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
    int numTimeSteps = 2; // TODO: I think this should be removed...
    int nt           = 2;
    int refLevels    = 1;
    int order        = 1;
    int dim          = 2;
    int use_gmres    = 0;
    int save_mat     = 0;
    double solve_tol = 1e-8;
    int print_level  = 3;
    int timeDisc     = 11;
    int spatialDisc  = 1;
    int max_iter     = 250;
    double dt        = -1;
    int lump_mass    = true;
    int AMGiters = 1;

    // AMG_parameters AMG = {"", "FFC", 3, 100, 0.01, 6, 1, 0.1, 1e-6};
    AMG_parameters AMG = {1.5, "", "FA", 100, 10, 10, 0.1, 0.05, 0.0, 1e-5, 1};
    const char* temp_prerelax = "A";
    const char* temp_postrelax = "FA";

    OptionsParser args(argc, argv);
    args.AddOption(&spatialDisc, "-s", "--spatial-disc",
                   "Spatial discretization (1=CG diffusion, 2=DG advection");
    args.AddOption(&timeDisc, "-t", "--time-disc",
                  "Time discretization (11=BDF1; 31=AB1).");
    args.AddOption(&order, "-o", "--order",
                  "Finite element order.");
    args.AddOption(&dt, "-dt", "--dt",
                  "Time step size.");
    args.AddOption(&refLevels, "-l", "--level",
                  "Number levels mesh refinement.");
    args.AddOption(&lump_mass, "-lump", "--lump-mass",
                  "Lump mass matrix to be diagonal.");
    args.AddOption(&print_level, "-p", "--print-level",
                  "Hypre print level.");
    args.AddOption(&solve_tol, "-tol", "--solve-tol",
                  "Tolerance to solve linear system.");
    args.AddOption(&dim, "-d", "--dim",
                  "Problem dimension.");
    //args.AddOption(&numTimeSteps, "-nt", "--num-time-steps", // TOOD: I think this should be removed...
    //              "Number of time steps.");
    args.AddOption(&nt, "-nt", "--num-time-steps", "Number of time steps.");
    args.AddOption(&use_gmres, "-gmres", "--use-gmres",
                  "Boolean to use GMRES as solver (default with AMG preconditioning).");
    args.AddOption(&AMGiters, "-amgi", "--amg-iters",
                  "Number of BoomerAMG iterations to precondition one GMRES step.");
    args.AddOption(&save_mat, "-save", "--save-mat",
                  "Boolean to save matrix to file.");
    args.AddOption(&(AMG.cycle_type), "-c", "--cycle-type",
                  "Cycle type; 0=F, 1=V, 2=W.");
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
    args.AddOption(&(AMG.filter_tolR), "-AfR", "--AIR-filterR",
                   "Theta value eliminating small entries in restriction (after building).");
    args.AddOption(&(AMG.filter_tolA), "-Af", "--AMG-filter",
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

    if (dt < 0) dt = 1.0/numTimeSteps;

    // For now have to keep SpaceTime object in scope so that MPI Communicators
    // get destroyed before MPI_Finalize() is called. 
    if (spatialDisc == 1) {
        CGdiffusion STmatrix(MPI_COMM_WORLD, timeDisc, nt,
                             dt, refLevels, order, lump_mass);
        STmatrix.BuildMatrix();
        // if (save_mat) {
        //     STmatrix.SaveMatrix("test.mm");
        // }
        // STmatrix.SetAMGParameters(AMG);
        // if (use_gmres) {
        //     STmatrix.SolveGMRES(solve_tol, max_iter, print_level, false,
        //                         use_gmres, AMGiters);
        // }
        // else {
        //     STmatrix.SolveAMG(solve_tol, max_iter, print_level, false);
        // }
        // STmatrix.PrintMeshData();
    }
    else {
        DGadvection STmatrix(MPI_COMM_WORLD, timeDisc, nt,
                             dt, refLevels, order, lump_mass);
        STmatrix.BuildMatrix();
        if (save_mat) {
            STmatrix.SaveMatrix("test.mm");
        }
        STmatrix.SetAMGParameters(AMG);
        if (use_gmres) {
            STmatrix.SolveGMRES(solve_tol, max_iter, print_level,
                                true, use_gmres, AMGiters);
        }
        else {
            STmatrix.SolveAMG(solve_tol, max_iter, print_level);
        }
        STmatrix.PrintMeshData();
    }

    MPI_Finalize();
    return 0;
}