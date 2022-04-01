for i in {2..2};
do
	for j in {5..5}
	do
		# fixed temporal domain:
		mpirun -np $((2**i)) --oversubscribe ./main3 -Pb 7 -r $j -T 1 -STA 5 -STU 5 -oU 3 -oP 2 -oA 1 -oZ 1 -P 2 -petscopts rc_SpaceTimeIMHD2D_approx
		# mpirun -np 1 --oversubscribe ./testTimeStep -Pb 7 -r $j -T 1 -NT $((2**i)) -oU 3 -oP 2 -oA 1 -oZ 1 -P 2 -petscopts rc_SpaceTimeIMHD2D_test
		# fixed dt:
		# mpirun -np $((2**i)) --oversubscribe ./main3 -Pb 7 -r $j -T $((2**(i-1))) -STA 3 -oU 3 -oP 2 -oA 1 -oZ 1 -P 2 -petscopts rc_SpaceTimeIMHD2D_fixdt_test
		# mpirun -np 1 --oversubscribe ./testTimeStep -Pb 6 -r $j -T $((2**(i-1))) -NT $((2**i)) -oU 3 -oP 2 -oA 1 -oZ 1 -P 2 -petscopts rc_SpaceTimeIMHD2D_fixdt_test
	done
done
