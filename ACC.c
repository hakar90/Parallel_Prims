#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <time.h>



int minKey(int key[], int visited[]) {
    int min = INT_MAX, index;
#pragma acc parallel loop reduction(min:min) num_gangs(32) vector_length(256)
    for (int i = 0; i < V; i++) {
        if (visited[i] == 0 && key[i] < min) {
            min = key[i];
            index = i;
        }
    }
    return index;
}#define V 10

void printMST(int from[], int n, int **graph) {
    printf("Edge   Weight\n");
    for (int i = 1; i < V; i++)
        printf("%d - %d    %d \n", from[i], i, graph[i][from[i]]);
}

void primMST(int **graph) {
    int from[V];
    int key[V];
    int visited[V];
    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, visited[i] = 0;

    key[0] = 0;
    from[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, visited);
        visited[u] = 1;

#pragma acc parallel loop num_gangs(32) vector_length(256)
        for (int v = 0; v < V; v++) {
            if (graph[u][v] && visited[v] == 0 && graph[u][v] < key[v])
                from[v] = u, key[v] = graph[u][v];
        }
    }
}

int main() {
    int **graph = (int **)malloc(V * sizeof(int *));
    for (int x = 0; x < V; x++)
        graph[x] = (int *)malloc(V * sizeof(int));

    // Generate random adjacency matrix
    srand(time(NULL));
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            graph[i][j] = rand() % 10;

    for (int i = 0; i < V; i++)
        graph[i][i] = 0;

    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            graph[j][i] = graph[i][j];

    double start = omp_get_wtime();
    primMST(graph);
    double end = omp_get_wtime();
    printf("Time taken for par = %f seconds\n", end - start); // Convert to milliseconds

    return 0;
}
