x = [[1, 2, 3, 4]]
for i in range(0, 4):
    part_one = x[0]
    one = part_one[0:i]
    two = part_one[i+1:]
    print(one+two)
