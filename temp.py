# Open the file in read mode
data = []
with open("output.txt", "r") as file:
    while True:
        line = file.readline()
        if not line:  # Exit the loop if no more lines are left
            break
        data.append(line)

used = [i for i in range(32)]
replaced = [i for i in range(32)]

pair = set()
dic = {}
for item in data:
  
  temp = item.split(' ')

  temp[1]=temp[1]
  # print(temp)
  used[int(temp[0])] += 1
  replaced[int(temp[1])] += 1
  pair.add(tuple(temp))
  if tuple(temp) in dic.keys():
    dic[tuple(temp)]+=1
  else:
    dic[tuple(temp)] = 0


print(used)
print(replaced)
print(dic)