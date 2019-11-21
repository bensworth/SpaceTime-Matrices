#! /bin/bash

# Run the "driver" executable in ../ST_class for many different spatial refinements, nx.
# Then run the Python script "compute_errors.py" for each solve to compute the numerical errors.
# The solution information, including the errors, will be combined for all nx into one .npy file
# for which the data can be read and plotted.

# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory within the current directory.
#       Otherwise HYPRE will just ignore your request to output the data

date


np=4            # Number of processes
FD=2            # ID of problem to be solved
min_refine=1    # Minimum spatial refinement
max_refine=2    # Maximum spatial refinement nx == 2^(2+refine)
pit=0           # Parallel-in-time flag
dim=2

# Choose one of these space+time pairs...
### --- Explicit Runge-Kutta --- ###
space=1
RK=111
#space=2
#RK=122
#space=3
#RK=133
#space=4
#RK=144

### --- Implicit Runge-Kutta --- ###
#space=1
#RK=211
#space=2
#RK=222
#space=3
#RK=233
#space=4
#RK=254

# Name of file to be output... "^REFINE^" will be replaced with the actual refinement...
dir=data
out=RK"$RK"_U"$space"_d"$dim"_l^REFINE^_FD"$FD"_pit"$pit"

# Run solves at different spatial refinements...
echo "solving..."
for l in `seq $min_refine $max_refine`
do
    echo "Spatial refinement = $l"

    # Make sure we're not appending to old python data, so delete it on first spatial refinement...
    if [ $l -eq $min_refine ]
    then
        rm -f $dir/U_${out/_l^REFINE^/}.npy
    fi

    # Run solver at current spatial refinement
    mpirun -np $np ./../ST_class/driver -s 3 -t $RK -o $space -l $l -d $dim -FD $FD -pit $pit -p 3 -saveX 1 -out $dir/U_${out/^REFINE^/$l} -tol 1e-10 -gmres 1 -maxit 40

    # Compute errors...
    python compute_errors.py $dir/U_${out/^REFINE^/$l}

    # Remove the output files created from solve... (this won't delete the .npy file created above)
    rm -f $dir/U_${out/^REFINE^/$l}*
done
