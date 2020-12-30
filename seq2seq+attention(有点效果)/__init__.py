import time

def f():
    for i in range(10):
        time.sleep(2)
        yield i

data=f()
start =time.time()
for j in data:
    print(j)

print(time.time()-start)

