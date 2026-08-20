#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA_ERR %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); \
    return -1; } } while(0)

__global__ void while_body(cudaGraphConditionalHandle h, int* counter, int target) {
    int c = atomicAdd(counter, 1) + 1;
    unsigned int cont = (c < target) ? 1u : 0u;
    cudaGraphSetConditional(h, cont);
}

__global__ void set_if_from_device(cudaGraphConditionalHandle h, const int* pred) {
    cudaGraphSetConditional(h, (*pred) ? 1u : 0u);
}
__global__ void if_body(int* flag) { *flag = 42; }

static int run_while(int target, int* d_counter) {
    cudaGraph_t graph;
    CK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));

    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeWhile;
    params.conditional.size = 1;
    cudaGraphNode_t condNode;
    CK(cudaGraphAddNode(&condNode, graph, nullptr, 0, &params));

    cudaGraph_t body = params.conditional.phGraph_out[0];
    cudaGraphNode_t kn;
    cudaKernelNodeParams knp = {};
    void* args[] = { &handle, &d_counter, &target };
    knp.func = (void*)while_body;
    knp.gridDim = dim3(1); knp.blockDim = dim3(1);
    knp.kernelParams = args;
    CK(cudaGraphAddKernelNode(&kn, body, nullptr, 0, &knp));

    cudaGraphExec_t exec;
    CK(cudaGraphInstantiate(&exec, graph, 0));

    for (int rep = 0; rep < 2; ++rep) {
        CK(cudaMemset(d_counter, 0, sizeof(int)));
        CK(cudaGraphLaunch(exec, 0));
        CK(cudaStreamSynchronize(0));
        int h = -1;
        CK(cudaMemcpy(&h, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        printf("WHILE  target=%d  replay#%d body_ran=%d  %s\n",
               target, rep, h, (h == target) ? "OK" : "MISMATCH");
        if (h != target) { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); return 1; }
    }
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    return 0;
}

static int run_if(int pred_value, int* d_pred, int* d_flag) {
    cudaGraph_t graph;
    CK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CK(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));

    cudaGraphNode_t setNode;
    cudaKernelNodeParams sknp = {};
    void* sargs[] = { &handle, &d_pred };
    sknp.func = (void*)set_if_from_device;
    sknp.gridDim = dim3(1); sknp.blockDim = dim3(1);
    sknp.kernelParams = sargs;
    CK(cudaGraphAddKernelNode(&setNode, graph, nullptr, 0, &sknp));

    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t condNode;
    CK(cudaGraphAddNode(&condNode, graph, &setNode, 1, &params));

    cudaGraph_t body = params.conditional.phGraph_out[0];
    cudaGraphNode_t kn;
    cudaKernelNodeParams knp = {};
    void* args[] = { &d_flag };
    knp.func = (void*)if_body;
    knp.gridDim = dim3(1); knp.blockDim = dim3(1);
    knp.kernelParams = args;
    CK(cudaGraphAddKernelNode(&kn, body, nullptr, 0, &knp));

    cudaGraphExec_t exec;
    CK(cudaGraphInstantiate(&exec, graph, 0));

    CK(cudaMemset(d_flag, 0, sizeof(int)));
    CK(cudaMemcpy(d_pred, &pred_value, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(exec, 0));
    CK(cudaStreamSynchronize(0));
    int h = -1;
    CK(cudaMemcpy(&h, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    int expect = pred_value ? 42 : 0;
    printf("IF     pred=%d flag=%d expect=%d  %s\n",
           pred_value, h, expect, (h == expect) ? "OK" : "MISMATCH");
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    return (h == expect) ? 0 : 1;
}

__global__ void round_kernel(cudaGraphConditionalHandle h, int* counter, int target) {
    int c = atomicAdd(counter, 1) + 1;
    cudaGraphSetConditional(h, (c < target) ? 1u : 0u);
}
__global__ void round_kernel_flag(int* counter, int target, int* cont_out) {
    int c = atomicAdd(counter, 1) + 1;
    *cont_out = (c < target) ? 1 : 0;
}

static int bench(int N, int reps, int* d_counter, int* d_cont) {
    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaGraph_t graph; CK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeWhile;
    params.conditional.size = 1;
    cudaGraphNode_t condNode;
    CK(cudaGraphAddNode(&condNode, graph, nullptr, 0, &params));
    cudaGraph_t body = params.conditional.phGraph_out[0];
    cudaGraphNode_t kn; cudaKernelNodeParams knp = {};
    void* args[] = { &handle, &d_counter, &N };
    knp.func = (void*)round_kernel; knp.gridDim = dim3(1); knp.blockDim = dim3(1);
    knp.kernelParams = args;
    CK(cudaGraphAddKernelNode(&kn, body, nullptr, 0, &knp));
    cudaGraphExec_t exec; CK(cudaGraphInstantiate(&exec, graph, 0));

    double t_host = 0, t_dev = 0;
    int last_host = 0, last_dev = 0;
    for (int r = 0; r < reps; ++r) {
        CK(cudaMemset(d_counter, 0, sizeof(int)));
        cudaDeviceSynchronize();
        auto a0 = std::chrono::high_resolution_clock::now();
        int cont = 1, guard = 0;
        while (cont && guard++ < N + 2) {
            round_kernel_flag<<<1,1,0,s>>>(d_counter, N, d_cont);
            CK(cudaMemcpyAsync(&cont, d_cont, sizeof(int), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
        }
        auto a1 = std::chrono::high_resolution_clock::now();
        t_host += std::chrono::duration<double, std::micro>(a1 - a0).count();
        CK(cudaMemcpy(&last_host, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

        CK(cudaMemset(d_counter, 0, sizeof(int)));
        cudaDeviceSynchronize();
        auto b0 = std::chrono::high_resolution_clock::now();
        CK(cudaGraphLaunch(exec, s));
        CK(cudaStreamSynchronize(s));
        auto b1 = std::chrono::high_resolution_clock::now();
        t_dev += std::chrono::duration<double, std::micro>(b1 - b0).count();
        CK(cudaMemcpy(&last_dev, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    }
    double per_round_host = t_host / (reps * (double)N);
    double per_round_dev  = t_dev  / (reps * (double)N);
    printf("BENCH  N=%d reps=%d  host_arm=%d dev_arm=%d (want %d)\n",
           N, reps, last_host, last_dev, N);
    printf("BENCH  host-decided-loop:   %.3f us/round total, %.3f us/round\n",
           t_host / reps, per_round_host);
    printf("BENCH  device-WHILE-graph:  %.3f us/round total, %.3f us/round\n",
           t_dev / reps, per_round_dev);
    printf("BENCH  removable host round-trip per round = %.3f us\n",
           per_round_host - per_round_dev);
    CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
    CK(cudaStreamDestroy(s));
    return (last_host == N && last_dev == N) ? 0 : 1;
}

int main() {
    int dev = 0; cudaDeviceProp prop;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    printf("device=%s sm_%d%d runtime=%d driver=%d\n",
           prop.name, prop.major, prop.minor, rt, drv);

    int *d_counter=nullptr, *d_pred=nullptr, *d_flag=nullptr;
    CK(cudaMalloc(&d_counter, sizeof(int)));
    CK(cudaMalloc(&d_pred, sizeof(int)));
    CK(cudaMalloc(&d_flag, sizeof(int)));

    int rc = 0;
    rc |= run_while(1, d_counter);
    rc |= run_while(5, d_counter);
    rc |= run_while(17, d_counter);
    rc |= run_if(1, d_pred, d_flag);
    rc |= run_if(0, d_pred, d_flag);

    int *d_cont=nullptr; CK(cudaMalloc(&d_cont, sizeof(int)));
    rc |= bench(16, 200, d_counter, d_cont);
    cudaFree(d_cont);

    cudaFree(d_counter); cudaFree(d_pred); cudaFree(d_flag);
    printf("PROBE_RESULT=%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
