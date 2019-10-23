#!/bin/bash
#MSUB -l nodes=64
#MSUB -l partition=quartz
#MSUB -l walltime=00:30:00
#MSUB -q pbatch
#MSUB -V
#MSUB -o ./results/large.out


##### These are shell commands
date
for i in `seq 1 2`
do

  echo "n512"
    srun -N 16 -n 512 ./driver -c 1 -s 2 -l 6 -nt 1024 -t 11 -dt 0.0156 >> ./results/n512_BE_V.txt
    srun -N 16 -n 512 ./driver -c 0 -s 2 -l 6 -nt 1024 -t 11 -dt 0.0156 >> ./results/n512_BE_F.txt
    
    srun -N 16 -n 512 ./driver -c 1 -s 2 -l 6 -nt 1024 -t 31 -dt 0.007 >> ./results/n512_FE_V.txt
    srun -N 16 -n 512 ./driver -c 0 -s 2 -l 6 -nt 1024 -t 31 -dt 0.007 >> ./results/n512_FE_F.txt
    wait
    
  echo "n1024"
    srun -N 32 -n 1024 ./driver -c 1 -s 2 -l 6 -nt 2048 -t 11 -dt 0.0156 >> ./results/n1024_BE_V.txt
    srun -N 32 -n 1024 ./driver -c 0 -s 2 -l 6 -nt 2048 -t 11 -dt 0.0156 >> ./results/n1024_BE_F.txt
    
    srun -N 32 -n 1024 ./driver -c 1 -s 2 -l 6 -nt 2048 -t 31 -dt 0.007 >> ./results/n1024_FE_V.txt
    srun -N 32 -n 1024 ./driver -c 0 -s 2 -l 6 -nt 2048 -t 31 -dt 0.007 >> ./results/n1024_FE_F.txt
    wait
    
  echo "n2048"
    srun -N 64 -n 2048 ./driver -c 1 -s 2 -l 6 -nt 4096 -t 11 -dt 0.0156 >> ./results/n2048_BE_V.txt 
    srun -N 64 -n 2048 ./driver -c 0 -s 2 -l 6 -nt 4096 -t 11 -dt 0.0156 >> ./results/n2048_BE_F.txt 
    
    srun -N 64 -n 2048 ./driver -c 1 -s 2 -l 6 -nt 4096 -t 31 -dt 0.007 >> ./results/n2048_FE_V.txt 
    srun -N 64 -n 2048 ./driver -c 0 -s 2 -l 6 -nt 4096 -t 31 -dt 0.007 >> ./results/n2048_FE_F.txt 
    
  echo RUN-${i} Done
done
