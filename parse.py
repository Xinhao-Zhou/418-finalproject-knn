import re
import numpy as np


data = []

with open("color100.txt","r") as f:
	line = f.readline();
	while line:
		words = line.split()
		read_data = [float(x) for x in words[1:9]]
		read_data.append(1);
		for j in range(9):
			if j < 8:
				print(str(read_data[j]),end=",")
			else:
				print(read_data[j],end="\n")
		line = f.readline()
	data.append(read_data)

