for value in range(1,6):
    print(value)

numbers = list(range(2,11,2)) #用list()函数直接创建列表，内容为2到11间的偶数
print(numbers)

odd_numbers = list(range(2,12,3)) #range(a,b,c)即从a到b，每间隔c取一个值，最后一个数值小于b为止
for i in odd_numbers:
    s = i + 1
    print(s)