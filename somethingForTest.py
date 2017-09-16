a = 0

def fun():
    global a
    a = 1
    print(a)

if __name__ == '__main__':
    fun()
    print(a)