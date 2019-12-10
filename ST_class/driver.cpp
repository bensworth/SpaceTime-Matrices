#include <iostream>
#include <fstream>
#include "CGdiffusion.hpp"
#include "DGadvection.hpp"
#include "FDadvection.hpp"
#include "mfem.hpp"
using namespace mfem;

// Sample command line:
//  srun -n 240 ./driver -s 2 -l 6 -nt 40 -t 31 -Ar 10 -AsR 0.2 -AsC 0.25 -AIR 1 -dt 0.0025 -lump 0

/* Some examples for solving 2nd-order, 2D finite difference problems
---IMPLICIT TIME-STEPPING:
mpirun -np 4 ./driver -pit 0 -s 3 -d 2 -t 222 -o 2 -l 4 -p 2 -gmres 1 -rebuild -1 -ppre -1 -tol 1e-5 -FD 2 -saveX 1
---IMPLICIT SPACE-TIME:
mpirun -np 4 ./driver -pit 1 -s 3 -d 2 -t 222 -o 2 -l 4 -p 2 -gmres 1 -rebuild -1 -ppre 1 -tol 1e-5 -FD 2 -saveX 1
---EXPLICIT TIME-STEPPING:
mpirun -np 4 ./driver -pit 0 -s 3 -d 2 -t 122 -o 2 -l 4 -p 2 -FD 2 -saveX 1
---EXPLICIT SPACE-TIME:
mpirun -np 4 ./driver -pit 1 -s 3 -d 2 -t 122 -o 2 -l 4 -p 2 -gmres 1 -ppre 1 -tol 1e-5 -FD 2 -saveX 1
*/

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
    bool isTimeDependent = true;
    int numTimeSteps = 2; // TODO: I think this should be removed...
    int nt           = 2;
    int order        = 1;
    int dim          = 2;
    
    int save_mat     = 0;
    int save_sol     = 0;
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
    

    /* --- Spatial discretization parameters --- */
    int spatialDisc  = 3;
    int refLevels    = 1;

    // Finite-difference specific parameters
    int FD_ProblemID = 1;
    int px = -1;
    int py = -1;


    // Initialize solver options struct with default parameters */
    Solver_parameters solver = {tol, maxiter, printLevel, bool(use_gmres), gmres_preconditioner, 
                                    AMGiters, precon_printLevel, rebuildRate, bool(binv_scale), bool(lump_mass)};



    
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
    AMG_parameters AMG = {1.5, "", "FA", 100, 10, 10, 0.1, 0.05, 0.0, 1e-5, 1};
    const char* temp_prerelax = "A";
    const char* temp_postrelax = "FA";

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
                  
    /* Spatial discretization */
    args.AddOption(&spatialDisc, "-s", "--spatial-disc",
                   "Spatial discretization (1=CG diffusion, 2=DG advection, 3=FD advection");
    args.AddOption(&order, "-o", "--order",
                  "Finite element order."); // TODO : general space disc refinement?
    args.AddOption(&refLevels, "-l", "--level",
                  "Number levels mesh refinement.");
    args.AddOption(&dim, "-d", "--dim",
                  "Problem dimension.");
    args.AddOption(&FD_ProblemID, "-FD", "--FD-prob-ID",
                  "Finite difference problem ID.");  
    args.AddOption(&px, "-px", "--procx",
                  "Number of procs in x-direction.");
    args.AddOption(&py, "-py", "--procy",
                  "Number of procs in y-direction.");                          

    /* --- Text output of solution etc --- */              
    args.AddOption(&out, "-out", "--out",
                  "Name of output file."); 
    args.AddOption(&save_mat, "-saveA", "--save-mat",
                  "Boolean to save matrix to file.");
    args.AddOption(&save_sol, "-saveX", "--save-sol",
                  "Boolean to save solution to file.");
                  
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
    /* -------------------------------------------------------------- */
    /* ------ Continuous-Galerkin discretizations of diffusion ------ */
    /* -------------------------------------------------------------- */
    if (spatialDisc == 1) {
        mass_exists = true; // Have a mass matrix
        CGdiffusion STmatrix(MPI_COMM_WORLD, pit, mass_exists, timeDisc, nt, dt, refLevels, order, lump_mass);
            
        STmatrix.SetSolverParameters(solver);                 
        STmatrix.Solve();                            
        
        STmatrix.PrintMeshData();
    }
    

    /* ----------------------------------------------------------------- */
    /* ------ Discontinuous-Galerkin discretizations of advection ------ */
    /* ----------------------------------------------------------------- */
    else if (spatialDisc == 2) {
        mass_exists = true; // Have a mass matrix
        DGadvection STmatrix(MPI_COMM_WORLD, pit, mass_exists, timeDisc, nt, dt, refLevels, order, lump_mass);
        
        STmatrix.SetSolverParameters(solver);
        STmatrix.Solve();

        STmatrix.PrintMeshData();
    }
    
    
    
    /* ------------------------------------------------------------ */
    /* ------ Finite-difference discretizations of advection ------ */
    /* ------------------------------------------------------------ */
    else if (spatialDisc == 3) {
        bool usingRK        = false;
        bool usingMultistep = false;
        int  smulti         = 0;
        
        double CFL_fraction;
        double CFLlim;
        // Set CFL number 
        
        // Explicit Runge-Kutta
        if (timeDisc > 100 && timeDisc < 200) {
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
        } else if (timeDisc > 200 && timeDisc < 300) {
            CFLlim = 1.0;
            CFL_fraction = 1.0; // Use a CFL number of ...
        
        // TODO: 
        //--- Work out what sensible CFLs look like for multistep schemes!  ------
        
        // Adams--Bashforth (explicit)
        } else if  (timeDisc >= 10 && timeDisc < 20) {
            CFLlim = 1.0;
            CFL_fraction = 0.85; // Use a CFL number of ...
            
        // Adams--Moulton (implcit)
        } else if  (timeDisc >= 20 && timeDisc < 30) {
            CFLlim = 1.0;
            CFL_fraction = 1.0; // Use a CFL number of ...
            
        // BDF (implicit)
        } else if  (timeDisc >= 30 && timeDisc < 40) {
            CFLlim = 1.0;
            CFL_fraction = 1.0; // Use a CFL number of ...
            
            usingMultistep = true;
            smulti = timeDisc % 10;
        }
            
        
        // Time step so that we run at CFL_fraction of the CFL limit 
        if (dim == 1) {
            double dx = 2.0 / pow(2.0, refLevels + 2); // Assume nx = 2^(refLevels + 2), and x \in [-1,1] 
            dt = dx * CFLlim;
        } else if (dim == 2) {
            double dx = 2.0 / pow(2.0, refLevels + 2); // Assume nx = 2^(refLevels + 2), and x \in [-1,1] 
            double dy = dx;
            dt = CFLlim/(1/dx + 1/dy);
        }
        
        dt *= CFL_fraction;
        
        // Manually set time to integrate to
        double T = 2.0; // For 1D in space...
        //double T = 2.0  * dt; // For 1D in space...
        if (dim == 2) T = 0.5; // For 2D in space...
        
        // Time step so that we run at approximately CFL_fraction of CFL limit, but integrate exactly up to T
        nt = ceil(T / dt);
        //nt = 6;
        //dt = T / (nt - 1);
        
        // Round up nt so that numProcess evenly divides number of unknowns. Assuming time-only parallelism...
        // NOTE: this will slightly change T... But actually, enforce this always so that 
        // tests are consistent accross time-stepping and space-time system
        if (usingRK) {
            int rem = nt % numProcess; // There are s*nt DOFs for integer s
            if (rem != 0) nt += (numProcess-rem); 
        } else if (usingMultistep) {
            int rem = (nt + 1 - smulti) % numProcess; // There are nt+1-s unknowns
            if (rem != 0) nt += (numProcess-rem); 
        }
        
        // TODO : I get inconsistent results if I set this before I set nt... But it shouldn't really matter.... :/ 
        dt = T / (nt - 1); 
        
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
        STmatrix.SetSolverParameters(solver);
        
        //STmatrix.SetAMGParameters(AMG);
        
        //STmatrix.SetAIRHyperbolic();
        STmatrix.SetAIR();
        //STmatrix.SetAMG();
        
        // Solve PDE
        STmatrix.Solve();
                
        if (save_sol) {
            std::string filename;
            if (std::string(out) == "") {
                filename =  "data/U" + std::to_string(pit); // Default filename...
            } else {
                filename = out;
            }
            STmatrix.SaveX(filename);
            //STmatrix.SaveRHS("b");
            //STmatrix.SaveMatrix("A");
            // Save data to file enabling easier inspection of solution            
            if (rank == 0) {
                int nx = pow(2, refLevels+2);
                std::map<std::string, std::string> space_info;
                space_info["space_order"]     = std::to_string(order);
                space_info["nx"]              = std::to_string(nx);
                space_info["space_dim"]       = std::to_string(dim);
                space_info["space_refine"]    = std::to_string(refLevels);
                space_info["problemID"]       = std::to_string(FD_ProblemID);
                for(int d = 0; d < n_px.size(); d++) {
                    space_info[std::string("p_x") + std::to_string(d)] = std::to_string(n_px[d]);
                }
                STmatrix.SaveSolInfo(filename, space_info);    
            }
        }
        
        
    }                         
    MPI_Finalize();
    return 0;
}

