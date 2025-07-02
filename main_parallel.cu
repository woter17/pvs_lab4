//#include <limits.h>
//#include <stdint.h>
//#include <asm-generic/errno.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
//#include <stdbool.h>

const long long int DEFAULT_ARRAY_SIZE = 100000000;
const int DEFAULT_RUNS = 2;
const int DEFAULT_THREADS = 256;
const int DEFAULT_BLOCKS = 8;

// Код взят из
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// Очень интересная и полезная презентация
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, const long long int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned long long int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int* CreateArray( const long long int SIZE) {
    int* llint_array = (int*) malloc(sizeof(int) * SIZE);
    for (int i = 0; i < SIZE; i++) {
        llint_array[i] = rand()%100;
    }
    return llint_array;
}

long long int GetEnvArraySize() {
    char* array_size_char = getenv("ARRAY_SIZE");
    long long int array_size_int = DEFAULT_ARRAY_SIZE;
    if (array_size_char != NULL) {
        array_size_int = atoll(array_size_char);
    } else {
        printf(
            "Переменная среды ARRAY_SIZE не получена, "
            "используем значение по умолчанию: %lld \n", DEFAULT_ARRAY_SIZE
        );
    }
    return array_size_int;
}

int GetEnvThreads() {
    char* thread_char = getenv("THREADS");
    int thread_int = DEFAULT_THREADS;
    if (thread_char != NULL) {
        thread_int = atoi(thread_char);
    } else {
        printf(
            "Переменная среды THREADS не получена, "
            "используем значение по умолчанию: %d \n", DEFAULT_THREADS
        );
    }
    return thread_int;
}

// int GetEnvBlocks() {
//     char* block_char = getenv("BLOCKS");
//     int block_int = DEFAULT_BLOCKS;
//     if (block_char != NULL) {
//         block_int = atoi(block_char);
//     } else {
//         printf(
//             "Переменная среды BLOCKS не получена, "
//             "используем значение по умолчанию: %d \n", DEFAULT_BLOCKS
//         );
//     }
//     return block_int;
// }

int GetEnvRuns() {
    char* runs_char = getenv("RUNS");
    int runs_int = DEFAULT_RUNS;
    if (runs_char != NULL) {
        runs_int = atoi(runs_char);
    } else {
        printf(
            "Переменная среды RUNS не получена, "
            "используем значение по умолчанию: %d \n", DEFAULT_RUNS
        );
    }
    return runs_int;
}

void CheckCudaError(cudaError_t err){
    if (err != cudaSuccess) {
        fprintf(stderr, "Fail (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

long long int SumElementsOfArray(const int* array, const long long int SIZE) {
    long long int result = 0;
    for (long long int i = 0; i < SIZE; i++) {
        result += array[i];
    }
    return result;
}

void PrintArray(const int* array, const long long int SIZE) {
    for (long long int i = 0; i < SIZE; i++) {
        printf("%d ",array[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    srand(time(0));
    //srand(1);
    const long long int ARRAY_SIZE = GetEnvArraySize();
    const int RUNS = GetEnvRuns();
    const int THREADS = GetEnvThreads();
    //const int BLOCKS = GetEnvBlocks();
    const int BLOCKS = (ARRAY_SIZE + (2 * THREADS) - 1) / (2 * THREADS);

    printf("\n\nПараллельная программа\n");
    printf("Размер массива: %lld\n", ARRAY_SIZE);
    printf("Выполнений: %d\n", RUNS);
    printf("Потоков в блоке: %d\n", THREADS);
    printf("Блоков (ДЛЯ ДАННОГО ЗАДАНИЯ НАСТРОЙКА КОЛ-ВА БЛОКОВ ИГНОРИРУЕТСЯ,\n\
            ПРОГРАММА САМА ВЫСЧИТАЛА НУЖНОЕ КОЛИЧЕСТВО БЛОКОВ НА ОСНОВЕ КОЛ-ВА ПОТОКОВ): %d\n", BLOCKS);

    int result_array_size = (ARRAY_SIZE + (THREADS-1)) / THREADS;
    
    // Таймер
    struct timespec begin, end;
    double exec_time = 0.0;
    double data_allocation_time = 0.0;

    // Цикл выполнения задачи и подсчёта времени её выполнения
    for (int i = 0; i < RUNS; i++) {

        // Массив хоста с данными
        int* host_float_array = NULL;
        host_float_array = CreateArray(ARRAY_SIZE);

        int* host_result_float_array = NULL;
        host_result_float_array = (int*) malloc(sizeof(int) * result_array_size);

        clock_gettime(CLOCK_REALTIME, &begin); // Начало таймера

        // Выделение глобальной памяти под массив, который будет передан GPU
        int* device_float_array = NULL;
        err = cudaMalloc(&device_float_array, ARRAY_SIZE * sizeof(int));
        CheckCudaError(err);
        //printf("Глоб массив выделен\n");

        // Выделение глобальной памяти под массив результат, который будет передан GPU
        int* device_result_float_array = NULL;
        err = cudaMalloc(&device_result_float_array, sizeof(int) * result_array_size);
        CheckCudaError(err);
        //printf("Глоб массив результата выделен\n");
        
        //Копирование массива в GPU
        err = cudaMemcpy(device_float_array,
                         host_float_array,
                         ARRAY_SIZE * sizeof(int),
                         cudaMemcpyHostToDevice
                        );
        CheckCudaError(err);
        //printf("Глоб массив скопирован\n");

        clock_gettime(CLOCK_REALTIME, &end); // Конец таймера
        data_allocation_time += (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec)/1e9;
        clock_gettime(CLOCK_REALTIME, &begin); // Начало таймера
        
        // Выполнение задачи
        switch (THREADS) {
            case 1024:
                reduce6<1024><<<BLOCKS, 1024, 1024 * sizeof(int)>>>(device_float_array, device_result_float_array, ARRAY_SIZE);
                break;
            case 512:
                reduce6<512><<<BLOCKS, 512, 512 * sizeof(int)>>>(device_float_array, device_result_float_array, ARRAY_SIZE);
                break;
            case 256:
                reduce6<256><<<BLOCKS, 256, 256 * sizeof(int)>>>(device_float_array, device_result_float_array, ARRAY_SIZE);
                break;
            case 128:
                reduce6<128><<<BLOCKS, 128, 128 * sizeof(int)>>>(device_float_array, device_result_float_array, ARRAY_SIZE);
                break;
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        CheckCudaError(err);
        //printf("Задача выполнена\n");

        clock_gettime(CLOCK_REALTIME, &end); // Конец таймера
        exec_time += (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec)/1e9;
        clock_gettime(CLOCK_REALTIME, &begin); // Начало таймера

        // Берём результат от GPU
        err = cudaMemcpy(host_result_float_array,
                         device_result_float_array,
                         result_array_size * sizeof(int),
                         cudaMemcpyDeviceToHost
                        );
        CheckCudaError(err);
        //printf("Результат получен\n");
        
        // Освобождаем глобальную память GPU
        err = cudaFree(device_float_array);
        CheckCudaError(err);
        err = cudaFree(device_result_float_array);
        CheckCudaError(err);
        //printf("Память очищена\n");

        long long int final_device_res = SumElementsOfArray(host_result_float_array, result_array_size);

        clock_gettime(CLOCK_REALTIME, &end); // Конец таймера
        data_allocation_time += (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec)/1e9;
        
        //PrintArray(host_result_float_array, result_array_size);
        long long int final_host_res = SumElementsOfArray(host_float_array, ARRAY_SIZE);
        long long int diff = final_host_res - final_device_res;
        printf("Погрешность между вычислением на CPU и GPU (CPU - GPU): %lld\n", diff);
        //printf("сумма на CPU %f\n", final_host_res);
        //printf("сумма на GPU %f\n", final_device_res);

        free(host_float_array);
        free(host_result_float_array);
    }

    double mean_data_alloc_time = data_allocation_time / RUNS;
    double mean_exec_time = exec_time / RUNS;
    printf("Общее время выделения памяти, передачи данных и финального счёта: %f сек. \n", data_allocation_time);
    printf("Среднее время выделения памяти передачи данных и финального счёта: %f сек. \n\n", mean_data_alloc_time);
    printf("Общее время выполнения кода на GPU: %f сек. \n", exec_time);
    printf("Среднее время выполнения кода на GPU: %f сек. \n\n", mean_exec_time );
    printf("Общее время выполнения: %f сек. \n", exec_time + data_allocation_time);
    printf("Среднее время выполнения: %f сек.", mean_exec_time + mean_data_alloc_time);

    return 0;
}
