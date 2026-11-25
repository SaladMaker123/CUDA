#include "utils.h"

int main(int argc, char** argv)
{
    return runBenchmark(
        "cuBLAS HGEMM",
        nullptr,
        argc,
        argv
    );
}