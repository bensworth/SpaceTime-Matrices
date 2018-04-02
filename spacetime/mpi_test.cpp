#include "/usr/local/include/mpi.h"
#include <iostream>
#include<stdio.h>
#include<stdlib.h>


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    std::cout << "Hello world, - Rank " << rank << "\n";

    MPI_Finalize();
    return 0;
}


