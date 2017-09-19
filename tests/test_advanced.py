from __future__ import absolute_import,division,print_function,unicode_literals
from builtins import *
import numpy as np
import sys
import ndd
import time


# seed the generator
np.random.seed(seed=123)

# initialize the eq. distribution 
nclasses = 21*21
a0 = 1.0
al = [a0]*nclasses

npoints = 20
nlist = [int(x) for x in np.logspace(1, 4, num=npoints)]
nrep = nclasses
#nlist = [10000]
#nrep = 100
names = ['plugin','pseudo','dirichlet','nsb']

verbose=False

for n in nlist: 
    ms = np.zeros(4)
    m2s = np.zeros(4)
    timings = np.zeros(5)
    
    for i in range(nrep): 

        start_time = time.time()
        pp = np.random.dirichlet(al)
        h0 = -sum([x*np.log(x) for x in pp if x > 0.0])
        hist = np.random.multinomial(n,pp)
        if verbose: 
            dt = time.time() - start_time
            timings[0] += dt
            print("random_dist  --- %s seconds ---" % (timings[0]/float(i+1)),file=sys.stderr)
            sys.stderr.flush()

        # plugin estimate 
        start_time = time.time()
        xpl = ndd.entropy(hist,algorithm='plugin') - h0
        if verbose: 
            dt = time.time() - start_time
            timings[1] += dt
            print("plugin       --- %s seconds ---" % (timings[1]/float(i+1)),file=sys.stderr)
            sys.stderr.flush()

        # pseudocounts with a = 1
        start_time = time.time()
        xpseudo = ndd.entropy(hist,alpha=1.0,algorithm='pseudo') - h0
        if verbose: 
            dt = time.time() - start_time
            timings[2] += dt
            print("pseudocounts --- %s seconds ---" % (timings[2]/float(i+1)),file=sys.stderr)
            sys.stderr.flush()

        # dirichlet prior with alpha = 1
        start_time = time.time()
        xdir = ndd.entropy(hist,alpha=a0,algorithm='dirichlet') - h0
        dt = time.time() - start_time
        if verbose: 
            timings[3] += dt
            print("dirichlet    --- %s seconds ---" % (timings[3]/float(i+1)),file=sys.stderr)
            sys.stderr.flush()

        # NSB algorithm 
        start_time = time.time()
        xnsb = ndd.entropy(hist) - h0
        if verbose: 
            dt = time.time() - start_time
            timings[4] += dt
            print("NSB   %s %s %s %s  --- %s seconds ---" % (h0,xpl,xpseudo,xnsb,timings[4]/float(i+1)),file=sys.stderr)
            sys.stderr.flush()

        ms +=  np.asarray([xpl,xpseudo,xdir,xnsb])
        m2s +=  np.asarray([xpl**2,xpseudo**2,xdir**2,xnsb**2])

    ms /= float(nrep)
    m2s = np.sqrt(m2s / float(nrep))
    print (n,h0,' '.join([str(x) for x in ms]),' '.join([str(x) for x in m2s]),file=sys.stdout)
    sys.stdout.flush()

