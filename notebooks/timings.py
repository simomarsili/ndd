"""docs."""
import numpy
from numpy import random
from scipy.stats import entropy
import ndd

random.seed(123)

def fn_timer(function):
    from functools import wraps
    @wraps(function)
    def function_timer(*args, **kwargs):
        import time
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        return t1 - t0
    return function_timer

def random_counts(n=None, k=None, a=None):
    pp = random.dirichlet([a]*k)
    pp /= numpy.sum(pp)
    return (pp, random.multinomial(n, pp))

@fn_timer
def timer_NSB(x):
    result = ndd.entropy(x[x>0])
    #result = entropy(x[x>0])
    return result

@fn_timer
def timer_ML(x):
    result = ndd.entropy(x, dist=True)
    return result

@fn_timer
def timer_scipy(x):
    result = entropy(x[x>0])
    return result

names = ['scipy', 'ML', 'NSB']
timers = [timer_scipy, timer_ML, timer_NSB]
def timer_case(n, k, a):
    pp, counts = random_counts(n=n, k=k, a=a)
    ts = [timer(counts) for timer in timers]
    return ts

n = 1000
k = 1000
a = 1.0
#ns = numpy.logspace(1,9,num=9,dtype='int')
ks = numpy.logspace(1,8,num=8,dtype='int')
ts_NSB = []
ts_scipy = []
ts_ML = []
times = []
#for n in ns:
for k in ks:
    ts_mat = numpy.array([timer_case(n=n, k=k, a=a) for i in range(10)])
    times.append(numpy.mean(ts_mat, axis=0))

#ns = numpy.log10(ns)
ks = numpy.log10(ks)
times = numpy.log10(times)
#for j, n in enumerate(ns):
for j, n in enumerate(ks):
    print(j, n, " ".join([str(t) for t in times[j]]))


exit()
for n in ns:
    k = 1000000000
    #pp, counts = random_counts(n=n, k=k, a=a)
    pp = [0.1, 0.1]
    counts = numpy.random.random_integers(0,2,size=1000000000)
    result = get_ipython().magic(u'timeit -o entropy(counts)')
    t_scipy.append(result.average)
    result = get_ipython().magic(u'timeit -o ndd.entropy(counts)')
    t_nsb.append(result.average)
    result = get_ipython().magic(u'timeit -o ndd.entropy(counts, dist=True)')
    t_ml.append(result.average)

exit()


# In[52]:

from matplotlib import pyplot as plt
#increase the font size and markers size

from matplotlib import rc
rc('font', size=20)
rc('lines', markersize=12)

fig, axs = plt.subplots(1,1,figsize=(12,6))
axs.semilogx()
axs.semilogy()
axs.plot(ns, t_scipy, '-o', label='scipy')
axs.plot(ns, t_nsb, '-o', label='NSB')
axs.plot(ns, t_ml, '-o', label='ML')
plt.legend(loc='upper left', shadow=True)



# In[30]:

x = 1.0
result = get_ipython().magic(u'timeit -o x**2')


# In[32]:

type(result.average)

