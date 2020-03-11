#include <iostream>
#include <fstream>
// #include "CGdiffusion.hpp"
#include "FDadvection.hpp"
#include "mfem.hpp"
using namespace mfem;

// Sample command line:
//  srun -n 240 ./driver -s 2 -l 6 -nt 40 -t 31 -Ar 10 -AsR 0.2 -AsC 0.25 -AIR 1 -dt 0.0025 -lump 0

/* Some examples for solving 2nd-order, 2D finite difference problems
---IMPLICIT TIME-STEPPING:
mpirun -np 4 ./driver -pit 0 -s 3 -d 2 -nt 60 -t 222 -o 2 -l 6 -p 2 -gmres 1 -rebuild -1 -ppre -1 -tol 1e-5 -FD 2 -saveX 1
---IMPLICIT SPACE-TIME:
mpirun -np 4 ./driver -pit 1 -s 3 -d 2 -nt 60 -t 222 -o 2 -l 6 -p 2 -gmres 1 -rebuild -1 -ppre 1 -tol 1e-5 -FD 2 -saveX 1
---EXPLICIT TIME-STEPPING:
mpirun -np 4 ./driver -pit 0 -s 3 -d 2 -nt 60 -t 122 -o 2 -l 6 -p 2 -FD 2 -saveX 1
---EXPLICIT SPACE-TIME:
mpirun -np 4 ./driver -pit 1 -s 3 -d 2 -nt 60 -t 122 -o 2 -l 6 -p 2 -gmres 1 -ppre 1 -tol 1e-5 -FD 2 -saveX 1
*/


/* --- Ben, here is an example of AIR doing poorly --- */
/* Solve constant coefficient advection in 1D, 2nd-order BDF+2nd-order FD (space-time matrix is lower trinagular)

-Refine mesh in space-time by a factor of 2 each time
-AIR is diverging on 1st iteration (if GMRES is turned off, that is). 
-The blow up at the first iteration increases as the mesh is refined
-Convergence rate is influenced by number of procs (but I'm not using the on proc trinagular solve, am I?)

#---PERIODIC BCs
    mpirun -np 4 ./driver -pit 1 -s 3 -d 1 -t 32 -nt 61 -o 2 -l 6 -p 2 -gmres 0 -tol 1e-5 -FD 1 -save 2 
    mpirun -np 4 ./driver -pit 1 -s 3 -d 1 -t 32 -nt 121 -o 2 -l 7 -p 2 -gmres 0 -tol 1e-5 -FD 1 -save 2 
    mpirun -np 4 ./driver -pit 1 -s 3 -d 1 -t 32 -nt 241 -o 2 -l 8 -p 2 -gmres 0 -tol 1e-5 -FD 1 -save 2 
    
#---INFLOW BCs    
    mpirun -np 4 ./driver -pit 1 -s 3 -d 1 -t 32 -nt 61 -o 2 -l 6 -p 2 -gmres 0 -tol 1e-5 -FD 101 -save 2 
    mpirun -np 4 ./driver -pit 1 -s 3 -d 1 -t 32 -nt 121 -o 2 -l 7 -p 2 -gmres 0 -tol 1e-5 -FD 101 -save 2 
    mpirun -np 4 ./driver -pit 1 -s 3 -d 1 -t 32 -nt 241 -o 2 -l 8 -p 2 -gmres 0 -tol 1e-5 -FD 101 -save 2 
*/

// Pretty acceptable results:
srun -n128 ./FDdriver -s 3 -d 1 -t 32 -nt 1281 -o 2 -l 9 -p 2 -gmres 1 -tol 1e-5 -FD 1 -AsC 0.1 -AsR 0.1 -Ac 10 -Af 0.005 -Ar 3 -Ar1 N -Ar2 FA




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
    bool mass_exists = false;
    int numTimeSteps = 2; // TODO: I think this should be removed...
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
    
    int binv_scale   = 0;
    int lump_mass    = 1;
    
    int multi_init   = 0; 

    /* --- Spatial discretization parameters --- */
    int spatialDisc  = 2;
    int refLevels    = 3;

    // Finite-difference specific parameters
    int FD_ProblemID = 1;
    int px = -1;
    int py = -1;


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
    // AMG_parameters AMG = {15, "", "FA", 100, 10, 10, 0.1, 0.05, 0.0, 1e-5, 1};
    // const char* temp_prerelax = "A";
    // const char* temp_postrelax = "FA";

    // The "standard" AIR parameters used in SetAIR()
    AMG_parameters AMG = {15, "A", "FFC", 100, 0, 6, 0.005, 0.0, 0.0, 0.0, 1};
    const char* temp_prerelax = "A";
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
                  
    /* Spatial discretization */
    args.AddOption(&spatialDisc, "-s", "--spatial-disc",
                   "Spatial discretization (1=CG diffusion, 2=DG advection, 3=FD advection");
    args.AddOption(&order, "-o", "--order",
                  "Spatial discretization order."); 
    args.AddOption(&refLevels, "-l", "--level",
                  "Number levels mesh refinement; FD uses 2^refLevels DOFs.");
    args.AddOption(&dim, "-d", "--dim",
                  "Problem dimension.");
    args.AddOption(&FD_ProblemID, "-FD", "--FD-prob-ID",
                  "FD: Problem ID.");  
    args.AddOption(&px, "-px", "--procx",
                  "FD: Number of procs in x-direction.");
    args.AddOption(&py, "-py", "--procy",
                  "FD: Number of procs in y-direction.");                          

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

    if (dt < 0) dt = 1.0/numTimeSteps;

    // For now have to keep SpaceTime object in scope so that MPI Communicators
    // get destroyed before MPI_Finalize() is called. 
    /* ------------------------------------------------------------ */
    /* ------ Finite-difference discretizations of advection ------ */
    /* ------------------------------------------------------------ */
    bool usingRK        = false;
    bool usingMultistep = false;
    int  smulti         = 0;
    
    double CFL_fraction;
    double CFLlim;
    // Set CFL number 
    
    // Explicit Runge-Kutta
    if (timeDisc > 100 && timeDisc < 200) {
        usingRK = true;
        
        // These limits apply for ERKp+Up schemes
        CFL_fraction = 0.85;
        if (order == 1)  {
            CFLlim = 1.0;
        } else if (order == 2) {
            CFLlim = 0.5;
        } else if (order == 3) {
            CFLlim = 1.62589;
        } else if (order == 4) {
            CFLlim = 1.04449;
        } else if  (order == 5) {
            CFLlim = 1.96583;
        }
    // Implicit Runge-Kutta
    }
    else if (timeDisc > 200 && timeDisc < 300) {
        usingRK = true;
        
        CFLlim = 1.0;
        CFL_fraction = 1.0; // Use a CFL number of ...
    
    // TODO: 
    //--- Work out what sensible CFLs look like for multistep schemes!  ------
    
    // BDF (implicit)
    }
    else if  (timeDisc >= 30 && timeDisc < 40) {
        CFLlim = 0.8;
        //CFLlim = 1.0;
        CFL_fraction = 1.0; // Use a CFL number of ...
        
        usingMultistep = true;
        smulti = timeDisc % 10;
    }
        
    double dx, dy = -1.0;
    
    // Time step so that we run at CFL_fraction of the CFL limit 
    if (dim == 1) {
        dx = 2.0 / pow(2.0, refLevels); // Assumes nx = 2^refLevels, and x \in [-1,1] 
        dt = dx * CFLlim;
    }
    else if (dim == 2) {
        dx = 2.0 / pow(2.0, refLevels); // Assumes nx = 2^refLevels, and x \in [-1,1] 
        dy = dx;
        dt = CFLlim/(1/dx + 1/dy);
    }
    
    CFL_fraction = 5.0;
    dt *= CFL_fraction;
    
    
    // // Manually set time to integrate to (just comment or uncomment this...)
    // double T = 2.0; // For 1D in space...
    // //double T = 2.0  * dt; // For 1D in space...
    // if (dim == 2) T = 0.5; // For 2D in space...
    // 
    // // Time step so that we run at approximately CFL_fraction of CFL limit, but integrate exactly up to T
    // nt = ceil(T / dt);
    // //nt = 6;
    // //dt = T / (nt - 1);
    // 
    // // Round up nt so that numProcess evenly divides number of unknowns. Assuming time-only parallelism...
    // // NOTE: this will slightly change T... But actually, enforce this always so that 
    // // tests are consistent accross time-stepping and space-time system
    // if (usingRK) {
    //     int rem = nt % numProcess; // There are s*nt DOFs for integer s
    //     if (rem != 0) nt += (numProcess-rem); 
    //     //nt = 4; 
    // } else if (usingMultistep) {
    //     int rem = (nt + 1 - smulti) % numProcess; // There are nt+1-s unknowns
    //     if (rem != 0) nt += (numProcess-rem); 
    // }
    // 
    // // TODO : I get inconsistent results if I set this before I set nt... But it shouldn't really matter.... :/ 
    // dt = T / (nt - 1); 
    // 
    /* --- Get SPACETIMEMATRIX object --- */
    std::vector<int> n_px = {};
    if (px != -1) {
        if (dim >= 1) {
            n_px.push_back(px);
        }
        if (dim >= 2) {
            n_px.push_back(py);
        }
    }
    
    // Build SpaceTime object
    FDadvection STmatrix(MPI_COMM_WORLD, pit, mass_exists, timeDisc, nt, 
                            dt, dim, refLevels, order, FD_ProblemID, n_px);
    
    // Set parameters
    STmatrix.SetAMGParameters(AMG);
    STmatrix.SetSolverParameters(solver);
    
    //STmatrix.SetAIRHyperbolic();
    //STmatrix.SetAIR();
    //STmatrix.SetAMG();
    
    // Solve PDE
    STmatrix.Solve();
        
            
    if (saveLevel >= 1) {
        std::string filename;
        if (std::string(out) == "") {
            filename =  "data/U" + std::to_string(pit); // Default filename...
        } else {
            filename = out;
        }
        
        if (saveLevel >= 2) STmatrix.SaveX(filename);
        if (saveLevel >= 3) STmatrix.SaveMatrix("A");
        if (saveLevel >= 3) STmatrix.SaveRHS("b");
        
        
        double discerror;    
        bool gotdiscerror = STmatrix.GetDiscretizationError(discerror);
        
        
        // Save data to file enabling easier inspection of solution            
        if (rank == 0) {
            int nx = pow(2, refLevels);
            std::map<std::string, std::string> space_info;

            space_info["space_order"]     = std::to_string(order);
            space_info["nx"]              = std::to_string(nx);
            space_info["space_dim"]       = std::to_string(dim);
            space_info["space_refine"]    = std::to_string(refLevels);
            space_info["problemID"]       = std::to_string(FD_ProblemID);
            for (int d = 0; d < n_px.size(); d++) {
                space_info[std::string("p_x") + std::to_string(d)] = std::to_string(n_px[d]);
            }
            
            // Not sure how else to ensure disc error is cast to a string in scientific format...
            if (gotdiscerror) {
                space_info["discerror"].resize(16);
                space_info["discerror"].resize(std::snprintf(&space_info["discerror"][0], 16, "%.6e", discerror));
            } 
            
            if (dx != -1.0) {
                space_info["dx"].resize(16);
                space_info["dx"].resize(std::snprintf(&space_info["dx"][0], 16, "%.6e", dx));
            }

            STmatrix.SaveSolInfo(filename, space_info);    
        }
    }
        
    MPI_Finalize();
    return 0;
}

