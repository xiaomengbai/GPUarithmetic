# GPUarithmetic
The code executes the major parts of arithmetic operations in TPC-H Q1, i.e., *sum(l_quantity)*, *sum(l_extendedprice)*, *sum(l_extendedprice*(1-l_discount))*, *sum(l_extendedprice*(1-l_discount)*(1+l_tax))*, and *avg(l_quantity)*. 



## Build and run
The test program with the old implementation: old.cu, decimal.h
```bash
make old
./old
```

The test program with the new implementation: new.cu, utils.h
```bash
make new
./new
```

## Change the datafile path

old.cu/new.cu:
```c++
    const char *datafile = "/data/tpch/data/scale_1/csv/org/lineitem.tbl";
```