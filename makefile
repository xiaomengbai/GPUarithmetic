.PHONY: clean

ARCH := sm_75
NVCC := nvcc
CXXFLAGS += -arch=$(ARCH) --expt-relaxed-constexpr -O3
#LDFLAGS += -L.
#-ldecimal


old: old.cu decimal.h
	$(NVCC) $(CXXFLAGS) $(LDFLAGS) -o $@ old.cu

run: test
	./test

clean:
	rm -rf *.o
	rm -rf test

new: new.cu utils.h
	nvcc -std=c++11 -arch=sm_75 --expt-relaxed-constexpr -O3 -o $@ new.cu
