#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>

#define V 5 // Number of vertices
#define ROOT 0

void printMST(int parent[V], int graph[V][V]);
void primMST(int graph[V][V], int myRank, int numProcs);
void printGraph(int graph[V][V]);
void printArray(int arr[], int size);


int minKey(int key[], bool mstSet[],int rank) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++) {
        if (mstSet[v] == false && key[v] < min) {
            min = key[v];
            min_index = v;
        }
    }
    
    return min_index;
}

void primMST(int graph[V][V], int myRank, int numProcs) {
    int parent[V], key[V];
    bool mstSet[V];
    double total_time;
    
    int chunk = V / numProcs;
    int start = myRank * chunk;
    int end = (myRank == numProcs - 1) ? V : (myRank + 1) * chunk;
    

    for (int i = 0; i < V; i++){
        key[i] = INT_MAX, mstSet[i] = false;parent[i]=0;}
    key[start] = 0;
    parent[start] = -1;
        clock_t start_time = clock();
    for (int count = start; count < end; count++) {
        int u = minKey(key, mstSet,myRank);
        
        mstSet[u] = true;
        
        for (int v = 0; v < V; v++){
            if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }}
    
   
    if(myRank!=ROOT){
    	parent[start]=INT_MAX;
    	key[start]=INT_MAX;}
    	
    

     int *combinedParentArray = NULL;
     int *combinedKeyArray = NULL;
    if (myRank == ROOT){
        combinedParentArray = malloc(V * numProcs * sizeof(int));
        combinedKeyArray = malloc(V * numProcs * sizeof(int));}

    MPI_Gather(parent, V, MPI_INT, combinedParentArray, V, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(key, V, MPI_INT, combinedKeyArray, V, MPI_INT, ROOT, MPI_COMM_WORLD);

	
	
    if (myRank == ROOT) {
    	parent[0]=-1;
    	key[0]=0;
        for (int i = 1; i < V; i++) {
            int temp = combinedKeyArray[i];
            int temp1;
            for (int j = 1; j < numProcs; j++) {
                if (temp > combinedKeyArray[j * V] && combinedKeyArray[j * V] != 0) {
                	
                    temp = combinedKeyArray[j * V];
                    temp1= combinedParentArray[j * V];
                }
            }

            
            key[i]=temp;
            parent[i] = temp1;
        }

    }
    
    clock_t end_time = clock();

    
    
            


    


    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    if (myRank == ROOT) {
    	
        printf("Execution Time: %.6f seconds\n", total_time);
    }
    
    free(combinedParentArray);
}

void printMST(int parent[], int graph[V][V]) {
    printf("Edge \tWeight\n");
    for (int i = 1; i < V; i++)
        printf("%d - %d \t%d \n", parent[i], i, graph[i][parent[i]]);
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int myRank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int(*graph)[V] = malloc(sizeof(int[V][V]));
    if (graph == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Finalize();
        return 1;
    }

   
    if (myRank == ROOT) {
        for (int i = 0; i < V; i++) {
            for (int j = i + 1; j < V; j++) {
                int weight = rand() % 100 + 1;
                graph[i][j] = weight;
                graph[j][i] = weight;
            }
            graph[i][i] = 0;
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);
    primMST(graph, myRank, numProcs);
    MPI_Barrier(MPI_COMM_WORLD);
    free(graph);
    MPI_Finalize();
    return 0;
}
void printGraph(int graph[V][V]) {
    printf("Graph:\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            printf("%d\t", graph[i][j]);
        }
        printf("\n");
    }
}

void printArray(int arr[], int size) {
    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

