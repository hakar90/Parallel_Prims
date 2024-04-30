#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>
#define V 10000
#define INF INT_MAX

__global__ void primMST(int *graph, int *parent, bool *mstSet, int *key) {
    extern __shared__ int s_data[];
    int *s_key = s_data;
    bool *s_mstSet = (bool*)&s_key[blockDim.x];

    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + thread_id;

    if (global_id < V) {
        s_key[thread_id] = key[global_id];
        s_mstSet[thread_id] = mstSet[global_id];
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = 2 * stride * thread_id;
            if (index < blockDim.x) {
                if (!s_mstSet[index] && (s_mstSet[index + stride] || s_key[index] < s_key[index + stride])) {
                    s_key[index] = s_key[index + stride];
                    s_mstSet[index] = s_mstSet[index + stride];
                }
            }
            __syncthreads();
        }

        if (thread_id == 0) {
            int min_key = INF;
            int u = -1;
            for (int i = 0; i < blockDim.x && i < V; i++) {
                if (!s_mstSet[i] && s_key[i] < min_key) {
                    min_key = s_key[i];
                    u = i + blockIdx.x * blockDim.x;
                }
            }
            if (u != -1) {
                mstSet[u] = true;
                for (int v = 0; v < V; v++) {
                    int weight = graph[u * V + v];
                    if (weight && !mstSet[v] && weight < key[v]) {
                        parent[v] = u;
                        key[v] = weight;
                    }
                }
            }
        }
    }
}

void setupAndRunPrim(int *graph, int *parent, bool *mstSet, int *key) {
    int shared_mem_size = 2 * V * sizeof(int); // Adjust size for your data structures
    primMST<<<(V + 255) / 256, 256, shared_mem_size>>>(graph, parent, mstSet, key);
}

int main() {
    int *graph, *parent, *key;
    bool *mstSet;

    cudaMallocManaged(&graph, V * V * sizeof(int));
    cudaMallocManaged(&parent, V * sizeof(int));
    cudaMallocManaged(&mstSet, V * sizeof(bool));
    cudaMallocManaged(&key, V * sizeof(int));

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            graph[i * V + j] = (i != j) ? (rand() % 100 + 1) : INF;
        }
        key[i] = INF;
        mstSet[i] = false;
        parent[i] = -1;
    }
    key[0] = 0; // Start from the first vertex
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    setupAndRunPrim(graph, parent, mstSet, key);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Runtime: %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

