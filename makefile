DEBUG=0
CC=clang++
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG)

CFILES = main.cpp knn_prep.cpp kmeans.cpp
HFILES = knn_header.h kmeans.h

all: knn

knn: knn_prep.o kmeans.o cycletimer.o
	$(CC) $(CFLAGS) main.cpp -o knn

knn_prep.o:
	$(CC) $(CFLAGS) -c knn_prep.cpp -o knn_prep.o

kmeans.o: knn_prep.o
	$(CC) $(CFLAGS) -c kmeans.cpp -o kmeans.o

cycletimer.o:
	$(CC) $(CFLAGS) -c cycletimer.c -o cycletimer.o

clean:
	rm -f knn knn_prep.o kmeans.o
