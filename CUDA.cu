#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>

#define V 100
#define INF INT_MAX

__global__ void primMST(int *graph, int *parent, bool *mstSet, int *key) {
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    
    u*=V;
    if (u < V*V) {

        for(int v=0;v<V;v++){
        if (!mstSet[v] && graph[u + v] && graph[u + v] < key[v]) {
            key[v] = graph[u + v];
            parent[v]=u/V;
        }
        }
                
    }
}

void printMST(int parent[], int graph[V][V]) {
    printf("Edge \tWeight\n");
    for (int i = 1; i < V; i++)
        printf("%d - %d \t%d \n", parent[i], i, graph[i][parent[i]]);
}

int main() {
    int *graph, *parent, *key;
    bool *mstSet;
    int graph1[V][V];

    cudaMallocManaged(&graph, V * V * sizeof(int));
    cudaMallocManaged(&parent, V * sizeof(int));
    cudaMallocManaged(&mstSet, V * sizeof(bool));
    cudaMallocManaged(&key, V * sizeof(bool));
    

    // Initialize graph1, mstSet, and parent
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            int weight = rand() % 100 + 1;
            graph1[i][j] = weight;
            graph1[j][i] = weight;
        }
        graph1[i][i] = 0;
        mstSet[i] = false;
        parent[i] = 0;
        key[i]=INF;
    }
    parent[0] = -1;

    // Copy graph1 to graph using memcpy
    cudaMemcpy(graph, graph1, V * V * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads_per_block = 100;
    int blocks = (V + threads_per_block - 1) / threads_per_block;
    primMST<<<blocks, threads_per_block>>>(graph, parent, mstSet, key);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time: %.6f milliseconds\n", milliseconds);


    printMST(parent,graph1);
    // Free managed memory
    cudaFree(graph);
    cudaFree(parent);
    cudaFree(mstSet);

    return 0;
}
