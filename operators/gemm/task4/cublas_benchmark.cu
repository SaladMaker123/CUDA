#include "utils.h"

int main(int argc, char** argv)
{
    return runBenchmark(
        "cuBLAS SGEMM",
        nullptr,
        argc,
        argv
    );
}
