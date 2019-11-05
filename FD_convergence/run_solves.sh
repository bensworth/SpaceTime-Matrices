# Run the "driver" executable in ../ST_class for many different spatial refinements, nx.
# Then run the Python script "compute_errors.py" to compute the numerical errors associated with the computed solutions.

# Notes:
#   -Run me like >> sh run_solves.sh
#   -For the data to be saved, you must have created a "data" subdirectory within the current directory.
#       Otherwise HYPRE will just ignore your request to output the data

date


min_refine=1
max_refine=7
pit=1

#spaceID=1
#RKID=111
spaceID=3
RKID=133
#spaceID=3
#RKID=133
#spaceID=4
#RKID=144

echo "solving..."
for l in `seq $min_refine $max_refine`
do
    echo "refinement $l"
    mpirun -np 1 ./../ST_class/driver -s 3 -t "$RKID" -o "$spaceID" -l "$l" -d 1 -FD 1 -pit "$pit" -p 0 -saveX 1 -outsuf _"$spaceID""$l"
done

echo "computing errors..."
for l in `seq $min_refine $max_refine`
do
    python compute_errors.py data/U_FD"$pit"_"$spaceID""$l".txt
done


