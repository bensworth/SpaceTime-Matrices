for i in {4..7};
do
	for j in {2..8}
	do
		# fixed temporal domain:
		mpirun -np $((2**i)) ./test -r $j -oU 2 -oP 1 -T 1 -P 1 -ST 0 -Pb 4 -Pe 256 -V 0 -petscopts rc_SpaceTimeStokes_SingAp
		# fixed dt:
		# mpirun -np $((2**i)) ./test -r $j -oU 2 -oP 1 -T $((2**(i-1))) -P 1 -ST 0 -Pb 1 -Pe 0 -V 0 -petscopts rc_SpaceTimeStokes_fixdt_SingAp
	done
done
