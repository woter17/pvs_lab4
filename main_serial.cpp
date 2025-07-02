//#include <limits.h>
//#include <stdint.h>
//#include <asm-generic/errno.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <time.h>
//#include <stdbool.h>

const long long int DEFAULT_ARRAY_SIZE = 1000000;
const int DEFAULT_RUNS = 100;

int* CreateArray( const long long int SIZE) {
    int* llint_array = (int*) malloc(sizeof(int) * SIZE);
    for (int i = 0; i < SIZE; i++) {
        llint_array[i] = rand()%100;
    }
    return llint_array;
}

void PrintArray(const float* array, const int SIZE) {
    for (int i = 0; i < SIZE; i++) {
        printf("%f ",array[i]);
    }
    printf("\n");
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

long long int SumElementsOfArray(const int* array, const long long int SIZE) {
    long long int result = 0;
    for (long long int i = 0; i < SIZE; i++) {
        result += array[i];
    }
    return result;
}

int main(int argc, char** argv) {
    printf("Последовательная программа\n");
    srand(time(0));
    //srand(1);
    const long long int ARRAY_SIZE = GetEnvArraySize();
    const int RUNS = GetEnvRuns();
    
    printf("Размер массива: %lld\n", ARRAY_SIZE);
    printf("Выполнений: %d\n", RUNS);

    // Таймер
    struct timespec begin, end;
    double exec_time = 0.0;

    // Цикл выполнения задачи и подсчёта времени её выполнения
    for (int i = 0; i < RUNS; i++) {

        int* float_array = NULL;
        float_array = CreateArray(ARRAY_SIZE);

        clock_gettime(CLOCK_REALTIME, &begin); // Начало таймера

        SumElementsOfArray(float_array, ARRAY_SIZE);

        clock_gettime(CLOCK_REALTIME, &end); // Конец таймера
        exec_time += (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec)/1e9;

        free(float_array);
    }

    double mean_exec_time = exec_time / RUNS;
    printf("Общее время выполнения: %f сек. \n", exec_time);
    printf("Среднее время выполнения: %f сек.", mean_exec_time);

    return 0;
}
