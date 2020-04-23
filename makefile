DEBUG=0
CC=clang++
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG)

CFILES = main.cpp knn_prep.cpp
HFILES = knn_header.h

all: knn

knn: knn_prep.o
	$(CC) $(CFLAGS) main.cpp -o knn

knn_prep.o:
	$(CC) $(CFLAGS) -c knn_prep.cpp -o knn_prep.o

clean:
	rm -f knn knn_prep.o
