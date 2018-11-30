n = 1000
while n > 0:
    print(n)
    n = n - 1
print("Blastoff")
print(n)

for i in [5, 4, 3, 2, 1]:
    print(i)
print("Blastoff")

print("Before")
for thing in [9, 41, 12, 3, 74, 15]:
    print(thing)
print("Blastoff")

zork = 0
print("Before", zork)
for thing in [9, 41, 12, 3, 74, 15]:
    zork = zork + thing
    print(zork, thing)
print("after", zork)
