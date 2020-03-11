#include <iostream>
#include <fstream>
#include "DGadvection.hpp"
#include "mfem.hpp"
using namespace mfem;


// Observations
// ------------
//  + Filtering is *really* important. Falgout coarsening led to OC ~100 w/o filtering.
//  + 2nd order implicit seems to struggle w/ larger time steps, 1st does not. 
//  + Does not do well w/ implicit on small time steps, dt << h, but this is okay.


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);


    /* ------------------------------------------------------ */
    /* --- Set default values for command-line parameters --- */
    /* ------------------------------------------------------ */
    int pit          = 1; 
    bool mass_exists = true;
    int nt           = 2;
    int order        = 1;
    int dim          = 2;
    
    int save_mat     = 0;
    int saveLevel    = 0;  // Don't save anything by default
    const char * out = ""; // Filename of data to be saved...
    
    int timeDisc     = 211;
    
    double dt        = -1;
    
    /* Solver parameters */
    double tol       = 1e-8;
    int maxiter      = 250;
    int printLevel   = 3;
    
    // Parameters if using GMRES as solver rather than AMG
    int use_gmres    = 0;
    int AMGiters     = 1;
    int gmres_preconditioner = 1;
    int precon_printLevel = 1;
    
    int rebuildRate  = 0; 
    
    int binv_scale   = 1;
    int lump_mass    = 1;
    
    int multi_init   = 0; 

    /* --- Spatial discretization parameters --- */
    int refLevels    = 3;

    // Initialize solver options struct with default parameters */
    Solver_parameters solver = {tol, maxiter, printLevel, bool(use_gmres), gmres_preconditioner, 
                                    AMGiters, precon_printLevel, rebuildRate, bool(binv_scale), bool(lump_mass), 
                                    multi_init};

    
    // double distance_R;
    // std::string prerelax;
    // std::string postrelax;
    // int interp_type;
    // int relax_type;
    // int coarsen_type;
    // double strength_tolC;
    // double strength_tolR;
    // double filter_tolR;
    // double filter_tolA;
    // int cycle_type;
    //AMG_parameters AMG = {"", "FFC", 3, 100, 0.01, 6, 1, 0.1, 1e-6};
    // AMG_parameters AMG = {1.5, "", "FA", 100, 10, 10, 0.1, 0.05, 0.0, 1e-5, 1};
    // const char* temp_prerelax = "A";
    // const char* temp_postrelax = "FA";

    // The "standard" AIR parameters used in SetAIR()
    AMG_parameters AMG = {15, "", "FFC", 100, 0, 6, 0.005, 0.0, 0.0, 0.0, 1};
    const char* temp_prerelax = "";
    const char* temp_postrelax = "FFC";
    

    OptionsParser args(argc, argv);
    
    args.AddOption(&pit, "-pit", "--parallel-in-time",
                   "1=Parallel in time, 0=Sequential in time");               
    args.AddOption(&timeDisc, "-t", "--time-disc",
                  "Time discretization (see RK IDs).");
    args.AddOption(&dt, "-dt", "--dt",
                  "Time step size.");
    args.AddOption(&nt, "-nt", "--num-time-steps", "Number of time steps.");    
                  
    /* Linear solver options */              
    args.AddOption(&(solver.tol), "-tol", "-solver-tolerance",
                  "Tolerance to solve linear system.");
    args.AddOption(&(solver.maxiter), "-maxit", "--max-iterations",
                  "Maximum number of linear solver iterations."); 
    args.AddOption(&(solver.printLevel), "-p", "--print-level",
                  "Hypre print level.");
    args.AddOption(&use_gmres, "-gmres", "--use-gmres",
                  "Boolean to use GMRES as solver (default with AMG preconditioning).");
    args.AddOption(&(solver.gmres_preconditioner), "-pre", "gmres-preconditioner",
                  "Type of preconditioning for GMRES.");                     
    args.AddOption(&AMGiters, "-amgi", "--amg-iters",
                  "Number of BoomerAMG iterations to precondition one GMRES step.");       
    args.AddOption(&(solver.precon_printLevel), "-ppre", "--preconditioner-print-level",
                  "Print level of preconditioner when using one.");
    args.AddOption(&(solver.rebuildRate), "-rebuild", "--rebuild-rate",
                   "Frequency at which AMG solver is rebuilt during time stepping (-1=never rebuild, 0=rebuild every opportunity, x>0=after x time steps");              
    args.AddOption(&lump_mass, "-lump", "--lump-mass",
                  "Lump mass matrix to be diagonal.");  
    args.AddOption(&binv_scale, "-binv", "--scale-binv",
                  "Scale linear system by inverse of mass diagonal blocks."); 
    args.AddOption(&(solver.multi_init), "-minit", "--multi-init",
                  "Technique for initializing multistep starting values.");                
                  
    args.AddOption(&order, "-o", "--order",
                  "Spatial discretization order."); 
    args.AddOption(&refLevels, "-l", "--level",
                  "Number levels mesh refinement; FD uses 2^refLevels DOFs.");
    args.AddOption(&dim, "-d", "--dim",
                  "Problem dimension.");

    /* --- Text output of solution etc --- */              
    args.AddOption(&out, "-out", "--out",
                  "Name of output file."); 
    args.AddOption(&saveLevel, "-save", "--save-sol-data",
                  "Level of information to save.");
                  
    /* --- AMG parameters --- */              
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
    
    
    // Fix-up boolean options that cannot be passed by the options parser
    solver.use_gmres  = bool(use_gmres);
    solver.binv_scale = bool(binv_scale);
    solver.lump_mass  = bool(lump_mass);
    
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
        //args.PrintOptions(std::cout); TODO : uncomment  me...
    }

    pit = bool(pit); // Cast to boolean

    if (dt < 0) dt = 1.0/nt;


    // For now have to keep SpaceTime object in scope so that MPI Communicators
    // get destroyed before MPI_Finalize() is called. 
    /* ----------------------------------------------------------------- */
    /* ------ Discontinuous-Galerkin discretizations of advection ------ */
    /* ----------------------------------------------------------------- */
    mass_exists = true; // Have a mass matrix
    DGadvection STmatrix(MPI_COMM_WORLD, pit, mass_exists, timeDisc, nt, dt, refLevels, order, lump_mass);
    // DGadvection STmatrix(MPI_COMM_WORLD, pit, mass_exists, timeDisc, nt, dt);
    STmatrix.SetAMGParameters(AMG);
    STmatrix.SetSolverParameters(solver);
    STmatrix.Solve();
    STmatrix.PrintMeshData();
                         
    MPI_Finalize();
    return 0;
}

