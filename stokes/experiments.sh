for i in {1..7};
do
	for j in {2..8}
	do
		# fixed temporal domain:
		# mpirun -np $((2**i)) ./test -r $j -oU 2 -oP 1 -T 1 -P 1 -Pb 1 -petscopts rc_SpaceTimeStokes
		# fixed dt:
		mpirun -np $((2**i)) ./test -r $j -oU 2 -oP 1 -T $((2**(i-1))) -P 1 -Pb 0 -petscopts rc_SpaceTimeStokes_approx_fixdt
	done
done
