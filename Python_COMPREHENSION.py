liste = [[i for i in range(3)] for j in range(5)]

liste

age = [1985, 1987, 1990, 1994]
prenom = ["hicham", "khadija", "Hajer", "abba"]

dixio = {prenom:age for prenom,age in zip(prenom,age) if age <1986}
dixio


toto = tuple((i**2 for i in range(7)))

toto
