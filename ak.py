N = int(input())
integers = list(map(int, input().split()))
results = []
moos = []
for i in range(len(integers)):
  right_side = integers[i+1:]
  results.append(right_side)
for i, result in enumerate(results): 
  new_result = list(set(result))  
  for j in new_result:   
      if result.count(j) > 1:
        if integers[i] != j:
          moos.append((integers[i], j))
moos = list(set(moos))
print(len(moos))

