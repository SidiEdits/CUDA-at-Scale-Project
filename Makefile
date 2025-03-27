main: main.cu
	nvcc main.cu -o main `pkg-config --cflags --libs opencv4`

clean:
	rm -f main *.png
