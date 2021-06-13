for i in {2..6};
do
	for j in {2..6}
	do
		# fixed temporal domain:
		# mpirun -np $((2**i)) --oversubscribe ./main3 -Pb 7 -r $j -T 1 -oU 3 -oP 2 -oA 1 -oZ 1 -P 2 -petscopts rc_SpaceTimeIMHD2D
		# mpirun -np 1 --oversubscribe ./testTimeStep -Pb 6 -r $j -T 1 -NT $((2**i)) -oU 3 -oP 2 -oA 1 -oZ 1 -P 2 -petscopts rc_SpaceTimeIMHD2D
		# fixed dt:
		# mpirun -np $((2**i)) --oversubscribe ./main3 -Pb 6 -r $j -T $((2**(i-1))) -oU 3 -oP 2 -oA 1 -oZ 1 -P 0 -petscopts rc_SpaceTimeIMHD2D_fixdt
		mpirun -np 1 --oversubscribe ./testTimeStep -Pb 6 -r $j -T $((2**(i-1))) -NT $((2**i)) -oU 3 -oP 2 -oA 1 -oZ 1 -P 2 -petscopts rc_SpaceTimeIMHD2D_fixdt
	done
done
