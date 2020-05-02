import random

for i in range(1,5):
	for j in range(0,1000000):
		randX = random.uniform(i * 10, i * 10 + 10)
		randY = random.uniform(i * 10, i * 10 + 10)
		print(str(randX) + ","+ str(randY) + ",1")
