#!/usr/bin/python
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
# http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
import time

class CTIMER(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
            print 'Elapsed: %s' % (time.time() - self.tstart)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>

if (__name__ == "__main__"):
    with CTIMER('foo_stuff'):
        for i in range(10):
            print "%d" % i

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
