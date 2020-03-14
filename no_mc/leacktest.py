import PyTetris
import tracemalloc

tracemalloc.start(10)

t1 = tracemalloc.take_snapshot()

a = PyTetris.State(10, 20)
for i in range(100000):
    a = a.transitions()[0][0]
a = None


t2 = tracemalloc.take_snapshot()



stats = t2.compare_to(t1, 'traceback')
top = stats[0]
print('\n'.join(top.traceback.format()))