## Space-time Block Preconditioning for Incompressible Resistive magnetohydrodynamics

This repository contains the source code used to run the experiments reported in the manuscript
> *Space-time Block Preconditioning for Incompressible Resistive Magnetohydrodynamics*\
> by [F. Danieli](https://www.maths.ox.ac.uk/people/federico.danieli),
> and [B.S. Southworth](http://ben-southworth.science/).



### Installation requirements
For the code to run properly, it requires [PETSc](https://www.mcs.anl.gov/petsc/ "PETSc"), [hypre](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/software "hypre") and [MFEM](https://mfem.org/ "MFEM") to be installed and linked accordingly. Cross-dependencies might pose some challenges: we've found that installing PETSc first, setting it up so that it automatically includes all common dependencies (metis, hypre, and mumps), and then installing MFEM, helps significantly in this regard. To install PETSc with the required packages, first download the source files, then configure and install it via
```
cd <petsc_directory>
./configure --download-hypre --download-metis --download-mumps
make PETSC_DIR=<petsc_directory> PETSC_ARCH=<petsc_arch_name>
```
Next comes MFEM. In the code we're making use of a function `SetEssentialTrueDofs()` for the class `BlockNonLinearForm`, which is not included in standard MFEM. As this is a very simple and pretty useful function (it allows to impose essential BC for different nodes on the different components of a multi-dimensional polynomial FE space), sooner or later we'll open a ticket to include it in MFEM, but for now one can find its implementation in the folder `mfem/fem/` (lines 553-566 of `nonlinearform.cpp` and line 278-279 of `nonlinearform.hpp`): this should be included in the source code for MFEM *before* it's configured and installed, via
```
cd <MFEM directory>
make BUILD_DIR=<MFEM-build-directory> config MFEM_USE_MPI=YES MFEM_USE_PETSC=YES MFEM_USE_METIS_5=YES PETSC_ARCH=<petsc_arch_name> PETSC_DIR=<petsc_directory>/<petsc_arch_name> HYPRE_DIR=<petsc_directory>/<petsc_arch_name>/externalpackages/git.hypre/src/hypre METIS_DIR=<petsc_directory>/<petsc_arch_name>/externalpackages/git.metis/petsc-build
cd <MFEM-build-directory>
make
```
With this, all the required software should be installed. For a quick (or a more in-depth) check on MFEM installation, run
```
cd <MFEM-build-directory>
make check
(make test)
```





### Code description
The source code provided implements an all-at-once space-time solver for incompressible resistive MHD applications. It's patchy in some places, due to the heritage from the various iterations and reorganisations it went through before reaching its "final" form, but the general structure is as follows:
- `main3.cpp`: Main file. This is responsible for setting-up the various experiments, and solving the corresponding nonlinear systems. In particular, it patches together all the relevant space-time operators assembled by the `IMHD2DSTOperatorAssembler` class, to define the global space-time system and its preconditioners. It also implements a non-linear solver (Newton) for recovering its solution.
- `imhd2dstoperatorassembler.cpp/hpp`: Most relevant class. Responsible for assembling and updating the various FE matrices used throughout. In particular, it assembles and owns all the single time-step operators, both for the system and for its preconditioners, as well as the relevant right-hand side; it then combines them to define the individual space-time operators for each variable, as well as their solvers employed in the preconditioners. For this task, it relies heavily on other "support" classes for each sub-task. Most noticeably:
	- `imhd2dspaceintegrator.cpp/hpp` and `imhd2dtimeintegrator.cpp/hpp`: Integrators defining the block-operator associated to the MHD system, split into its spatial and temporal part, respectively. They implement MFEM's `BlockNonLinearFormIntegrator` interface, and are responsible for computing the residual and the gradient of such operator at each time-step. Basically `IMHD2DSTOperatorAssembler` relies on these classes to assemble the "building blocks" for the space-time system (ie, the operator evaluations for each time-step), and then takes care of combining them to generate the whole space-time operators.
	- `oseenstpressureschurcomplement.cpp/hpp`: Class implementing the space-time version of the pressure-convection-diffusion preconditioner by Elman, Silvester & Wathen. Contains references to the relevant matrices for the pressure field (mass, "laplacian", and advection, Mp, Ap, Wp) owned by `IMHD2DSTOperatorAssembler`, and owns the associated solvers (Mpinv and Apinv)
	- `imhd2dstmagneticschurcomplement.cpp/hpp`: Class implementing the space-time version of the magnetic-part preconditioner by Cyr et al. Contains references to the relevant matrices for the vector potential field (mass, laplacian+advection, and space-time wave-like operator, Ma, Wa, CCa) owned by `IMHD2DSTOperatorAssembler`, and owns the associated solvers (Mainv and CCainv). In some experiments, however, a different preconditioner is used.
	- `parblocklowtrioperator.cpp/hpp`: Class representing a block lower-triangular operator, where each block row represents a time-step and is owned by a different processor. Pretty much every space-time operator is represented by this class.
	- `spacetimesolver.cpp/hpp` and `spacetimewavesolver.cpp/hpp`: Classes representing solvers for specific `ParBlockLowTriOperator`s (with one or two block subdiagonals, used for the velocity/magnetic potential space-time operators, FFu/FFa, and for the wave-like space-time operator CCa appearing in the magnetic Schur complement, respectively). Useful to solve space-time systems via sequential time-stepping: they contain references to the off-diagonal matrices, and own a solver for the main diagonal.
- `testcases.cpp/hpp`: Descriptors of various model problems.

The "flow" of the code is as follows: `main3` reads the prescribed options to identify the required test-case and solver configuration chosen. It passes this information to `IMHD2DSTOperatorAssembler`, which assembles and stores the relevant space-time operators / space-time solvers for the individual variables. This gives back their references to `main3`, which uses them to assemble the "global" system and preconditioner, according to the options specified. It then triggers the Newton solver, which relies on `IMHD2DSTOperatorAssembler` to iteratively update the relevant matrices in the system (and computing the system residual), until - hopefully! - convergence is reached.

There is an exception to this flow, which is triggered by selecting sequential time-stepping as a global solver for the system. In this case, after assembly, the control is given entirely to `IMHD2DSTOperatorAssembler`, which executes the function `TimeStep()`. This mimics the Newton solver implemented by `main3`, but in a single-time-step fashion.


###### Other relevant files
- `rc_SpaceTimeIMHD2D` and `rc_SpaceTimeIMHD2D_approx`: Option files for two sets of PETSc solvers configurations: one "exact", where direct solvers are used for inverting all the relevant matrices, and one "approximate", where various iterative solvers are used instead.
- `experiments.sh`: Convenience script file for running a sequence of simulations for various refinement levels (this can be done by changing the extrema in the `for` loops)
- `/results/*`: The convergence results are stored in this folder. Particularly:
	- Each sub-folder title defines the type of experiment analysed
	- Within each sub-folder, a `.txt` file collects general info on the specific experiment (ie, convergence and timing results for various refinement levels)
	- Each sub-sub-folder contains more specific info for a given refinement level (a pair `Np`,`r`). Particularly:
		- The `NEWTconv.txt` file contains info regarding the Newton iterations (residual evolution, no of inner iterations, timing)
		- The `GMRESconv_Nit*.txt` files contain info regarding the residual evolution for the inner (F)GMRES iterations per specific Newton iteration
- `/ParaView/*`: The solution plots are stored in this folder
- `/Meshes/*`: The meshes used are stored in this folder



###### Performance improvements
<span style="color:red">⚠️WARNING⚠️</span>
While the code provides a framework for profiling and timing the most relevant functions invoked, it is far from being optimised, and real performance measurements require a more careful implementation. Nonetheless, it can still useful for the purpose of comparing the relative performances of the space-time and single time-step approaches, as they fundamentally rely on the same function calls - be they called collectively or singularly by they processors. A few of the things that *should* be addressed, however, in order to get a more performant code, are the following:
- Having `IncompressibleMHD2DSpaceIntegrator` taking care of assembling the single-time step MHD operator is useful in which it neatly separates and delegates some of the tasks that otherwise should fall on the `IMHD2DSTOperatorAssembler` class, which is already code-heavy. It also relies on some already-existing MFEM interface, which saves us the effort of having to reimplement it. However, it is mostly a vestigial organ remaining from the attempt of building two versions of the MHD system - one of which was including SUPG-stabilisation, which I never got around to make work, for some reason. It is also quite inefficient, since the MFEM interface is not the most flexible one. In particular:
	- When `IMHD2DSTOperatorAssembler::UpdateLinearisedOperators()` (and in particular `IncompressibleMHD2DSpaceIntegrator::GetGradient()`) is invoked, we recover the gradient (separated in blocks) of the MHD operator. The relevant blocks are then *deep-copied* into the internal variables held by `IMHD2DSTOperatorAssembler`, so that the assembler always owns them. A much better approach would be to make them a reference, and transfer ownership, but all my attempts resulted in some memory error: this must have something to do with the way the `BlockNonLinearFormIntegrator` stores the gradient, but can't really figure out how to circumvent this. On the one hand, `BlockNonLinearFormIntegrator` complains if it loses ownership of the gradient blocks, on the other I need `IMHD2DSTOperatorAssembler` to own it, as many operators need to be preserved.
	- Whenever the method `GetGradient()` is called, it re-assembles the *whole* linearised system, while some of the blocks are actually linear - and hence need not be updated. I didn't find a way to selectively flag which blocks needed to be rebuild just relying on MFEM interface, but this would be quite helpful to improve performance.
	- The way I've implemented it, as the integrator loops over the elements to assemble the Galerkin matrices, it computes the contributions for every block at the same time (rather than, say, first looping to assemble the first block, then repeating the loop to assemble the second, and so on). This is also - to some extent - related to the kind of interface that MFEM was providing, but is very much a - possibly unfortunate - implementation choice, which carries some limitations with it. In particular, it forces to use the *same* integration rule to assemble *every* block, while different blocks demand in general different levels of accuracy. This allows to reuse some basis functions evaluations (for example, the gradient of the velocity basis functions must be used to assemble a lot of terms in a lot of blocks, and can be computed once and for all during the loop), but it's unclear whether this is advantageous over using lower-order integration rules for lower-degree terms.
- The method `IMHD2DSTOperatorAssembler::AssembleCCaBlocks()` might need some optimisation. It takes care of assembling the various blocks (Cp, C0, Cm) appearing in CCa; some of these, however, are constant, and can be assembled once and for all. Moreover, it gets to the final results by assembling lots of intermediate matrices: I'm sure there's a way to speed up this process.
- When selecting BoomerAMG+AIR as a solver for the space-time systems FFu and CCa (or FFa, depending on the type of magnetic Schur complement approximation picked), the MFEM solver interface requires assembling a Hypre matrix for the space-time system. This means performing an extra copy of Fu and Mu (or Fa and Ma, or Cp C0 and Cm), which seems superfluous: can't I just pass pointers to the already existing Fu and Mu data, instructing the code that they represent the diagonal and off-diagonal components of the parallel matrix? Luckily in the update phase (when Fu changes due to the new linearisation) I can simply change the diagonal block, which spares me from reassembling the whole hypre matrix from scratch, so I can spare some effort there, but it still seems like a lot of superfluous operations; also, it relies on the fact that the sparsity pattern of Fu doesn't change! Which should actually be the case (there's a mass matrix included in the operator after all), but feels...risky.
- Again when selecting BoomerAMG+AIR, the solvers setup must be optimised for. As of now I'm using the same values for inverting FFa, FFu and CCa: the first two might make sense, as they represent the same operator, but the third definitely needs some fine-tuning. The relevant functions that do so are in `imhd2dstoperatorassembler.cpp`: `SetUpBoomerAMGFFu()`,  `SetUpBoomerAMGFFa()`,  and `SetUpBoomerAMGCCa()`, respectively.
- The `TimeStep()` function of `IMHD2DSTOperatorAssembler` is a huge hack: this stems from my attempt at avoiding code duplication (I was doing a lot of changes to the code, and having to double them every single time to compare the parallel and sequential versions was driving me crazy), but the way it works is ugly as hell. Basically, it relies on the initialisation from the parallel version of the code: as such, it instructs each processor to assemble its own copies of every operator. It then "freezes" each processor until their turn comes: when a processor receives the solution from the previous one, it then updates its own operators and applies Newton to find the solution at its instant, before passing it on to the next. So, massive waste of memory and computational power, as each processor remains idle for like, 1/Np-th of the time. I'm still being careful when I time this, so that I don't include this fake idle time, but still. It's handy, though, because it ensures that the sequential and parallel versions of the code rely on the same functions - also, these functions are designed so that no unnecessary operators are assembled in the time-stepping case: for example, CCa relies on three blocks being assembled (Cp, C0, Cm) which describe the "dialogue" between the various time-steps: if the sequential code is invoked, though, only Cp is assembled, so no waste there, and the comparison is indeed fair.





### Running experiments
To aid in compilation, one might use the `makefile` included, and particularly the command `make main3`; in the makefile, the first few lines needs to be changed to include the directories where the local installations of MFEM and PETSc can be found. After compilation, running `./main3 -h` via terminal prints a thorough description of all the options accepted by the executable, but be aware that not all of them have been thoroughly tested. A typical program invocation might look like
```
mpirun -np <2^i> ./main3 -r <j> -STA <sA> -STU <su> -Pb <prob> -P <prec> -petscopts rc_SpaceTimeIMHD2D
```
where
- `i` identifies the number of processors involved, which also defines $\Delta t$
- `j` identifies the spatial refinement level
- `prob` defines the test case. Most experiments were down with `prob=6` (island coalescence), or `prob=6` (tearing mode)
- `sA`, `sU` and `prec` all contribute to defining the preconditioner used:
	- `prec` is more "global": it prescribes various simplifications to the preconditioner used by Cyr et all, progressively rendering it more upper triangular
	- `sU` defines the solver for the velocity block. Particularly, `sU=0` uses sequential time-stepping, while `sU=5` uses BoomerAMG+AIR
	- `sA` defines the approximation for the magnetic Schur complement. There is a plethora of choices here, but `sA=3` (which time-steps using the diagonal of Ma to define CCa, but the whole Ma externally in the definition of Sa) seems the most reliable. Alternatively, `sA=8`, time-steps using simply FFa to approximate Sa. Their time-parallel counterparts are given by `sA=5` and `sA=6`, respectively, which use BoomerAMG+AIR instead of time-stepping
	- Notice that choosing `sA=9`, `sA=10`, or `sU=9` triggers time-stepping for the *whole* system: this bypasses many other options. Notice that `sA=9` takes Cyr's approximation of the magnetic Schur complement, while `sA=10` simplifies it to Fa (the single time-step version of `sA=8`). In the latter case, the solver options can be tweaked by changing the options for `-AWaveSolver` in `rc_SpaceTimeIMHD2D`. <span style="color:red">⚠️WARNING⚠️</span> When solving via global time-stepping, we need to impose a stricter tolerance on the solver for the comparison wrt PinT to be fair: this is done internally, but *only* if no tolerance is prescribed in the PETSc option file, so make sure to use the `_noTol` version of `rc_SpaceTimeIMHD2D` in this case
