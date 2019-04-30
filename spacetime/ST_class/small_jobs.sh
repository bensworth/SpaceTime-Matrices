#!/bin/bash
#MSUB -l nodes=8
#MSUB -l partition=quartz
#MSUB -l walltime=00:30:00
#MSUB -q pdebug
#MSUB -V
#MSUB -o ./results/small.out

# MPI ranks

##### These are shell commands
date
for i in `seq 1 2`
do
        
  echo "n4/n16"
    srun --exclusive -N 1 -n 4 ./driver -c 1 -s 2 -l 6 -nt 8 -t 11 -dt 0.0156 >> ./results/n4_BE_V.txt 
    srun --exclusive -N 1 -n 4 ./driver -c 0 -s 2 -l 6 -nt 8 -t 11 -dt 0.0156 >> ./results/n4_BE_F.txt 
    srun --exclusive -N 1 -n 4 ./driver -c 1 -s 2 -l 6 -nt 8 -t 31 -dt 0.007 >> ./results/n4_FE_V.txt 
    srun --exclusive -N 1 -n 4 ./driver -c 0 -s 2 -l 6 -nt 8 -t 31 -dt 0.007 >> ./results/n4_FE_F.txt 

    srun --exclusive -N 1 -n 16 ./driver -c 1 -s 2 -l 6 -nt 32 -t 11 -dt 0.0156 >> ./results/n16_BE_V.txt 
    srun --exclusive -N 1 -n 16 ./driver -c 0 -s 2 -l 6 -nt 32 -t 11 -dt 0.0156 >> ./results/n16_BE_F.txt 
    srun --exclusive -N 1 -n 16 ./driver -c 1 -s 2 -l 6 -nt 32 -t 31 -dt 0.007 >> ./results/n16_FE_V.txt 
    srun --exclusive -N 1 -n 16 ./driver -c 0 -s 2 -l 6 -nt 32 -t 31 -dt 0.007 >> ./results/n16_FE_F.txt 
    # wait

  echo "n32"
    srun --exclusive -N 1 -n 32 ./driver -c 1 -s 2 -l 6 -nt 64 -t 11 -dt 0.0156 >> ./results/n32_BE_V.txt 
    srun --exclusive -N 1 -n 32 ./driver -c 0 -s 2 -l 6 -nt 64 -t 11 -dt 0.0156 >> ./results/n32_BE_F.txt 
    srun --exclusive -N 1 -n 32 ./driver -c 1 -s 2 -l 6 -nt 64 -t 31 -dt 0.007 >> ./results/n32_FE_V.txt 
    srun --exclusive -N 1 -n 32 ./driver -c 0 -s 2 -l 6 -nt 64 -t 31 -dt 0.007 >> ./results/n32_FE_F.txt 
    # wait

  echo "n64"
    srun --exclusive -N 2 -n 64 ./driver -c 1 -s 2 -l 6 -nt 128 -t 11 -dt 0.0156 >> ./results/n64_BE_V.txt 
    srun --exclusive -N 2 -n 64 ./driver -c 0 -s 2 -l 6 -nt 128 -t 11 -dt 0.0156 >> ./results/n64_BE_F.txt 
    srun --exclusive -N 2 -n 64 ./driver -c 1 -s 2 -l 6 -nt 128 -t 31 -dt 0.007 >> ./results/n64_FE_V.txt 
    srun --exclusive -N 2 -n 64 ./driver -c 0 -s 2 -l 6 -nt 128 -t 31 -dt 0.007 >> ./results/n64_FE_F.txt 
    # wait

  echo "n64"
    srun --exclusive -N 4 -n 128 ./driver -c 1 -s 2 -l 6 -nt 256 -t 11 -dt 0.0156 >> ./results/n128_BE_V.txt
    srun --exclusive -N 4 -n 128 ./driver -c 0 -s 2 -l 6 -nt 256 -t 11 -dt 0.0156 >> ./results/n128_BE_F.txt 
    # wait
    srun --exclusive -N 4 -n 128 ./driver -c 1 -s 2 -l 6 -nt 256 -t 31 -dt 0.007 >> ./results/n128_FE_V.txt 
    srun --exclusive -N 4 -n 128 ./driver -c 0 -s 2 -l 6 -nt 256 -t 31 -dt 0.007 >> ./results/n128_FE_F.txt 
    # wait

  echo "n256"
    srun --exclusive -N 8 -n 256 ./driver -c 1 -s 2 -l 6 -nt 512 -t 11 -dt 0.0156 >> ./results/n256_BE_V.txt
    srun --exclusive -N 8 -n 256 ./driver -c 0 -s 2 -l 6 -nt 512 -t 11 -dt 0.0156 >> ./results/n256_BE_F.txt
    srun --exclusive -N 8 -n 256 ./driver -c 1 -s 2 -l 6 -nt 512 -t 31 -dt 0.007 >> ./results/n256_FE_V.txt 
    srun --exclusive -N 8 -n 256 ./driver -c 0 -s 2 -l 6 -nt 512 -t 31 -dt 0.007 >> ./results/n256_FE_F.txt 

  echo RUN-${i} Done
done
