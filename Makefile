OBJECTS=main.o timer.o

CFLAGS=-std=gnu99 -O3 -Wall
# LDFLAGS=-lm -framework OpenCL
LDFLAGS =-lm -lrt -lOpenCL

# INPUTFILE = ./eval/step.in
# OUTPUTFILE = ./eval/step.out

INPUTFILE = ./eval/650.in
OUTPUTFILE = ./eval/650.cl

EXECUTABLE = music_opencl
# EXECUTABLE = music_seq

all: seq opencl

seq: $(OBJECTS) music_seq.o
	$(CC) -o music_seq $^ $(LDFLAGS)

opencl: $(OBJECTS) music_opencl.o
	$(CC) -o music_opencl $^ $(LDFLAGS)

clean:
	rm -rf music_seq music_opencl $(OBJECTS) music_seq.o music_opencl.o

run: $(EXECUTABLE)
	@thorq --add --mode single --device gpu/1080 --name test ./$(EXECUTABLE) $(INPUTFILE) $(OUTPUTFILE)