#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This class perform mux and demux of vectors. The dimensions 
#               of each output vector is defined once at construction.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy               as np;

#====================================#
## \brief class cmyquad ##
## \author luciano kruk     ##
#
## \description:
#====================================#

class CMUXDEMUX:

    def __init__(self, dims, isOutColumnVector=False):

        self.dims  = dims
        self.cdims = list(np.cumsum([0] + dims))
        self.N     = len(dims)
        self.isOutColumnVector = isOutColumnVector

    def mux(self, *vec):
        out_a = np.hstack(( vec[i].squeeze() for i in range(self.N) ))

        if self.isOutColumnVector:
            out_b = out_a.reshape((sum(self.dims),1))
        else:
            out_b = np.asarray(out_a)

        return out_b

    def demux(self, vec):
        out_a = vec.squeeze()

        if self.isOutColumnVector:
            out_b = [ out_a[self.cdims[i]:self.cdims[i+1]].reshape((self.dims[i],1)) for i in range(self.N) ]
        else:
            out_b = [ out_a[self.cdims[i]:self.cdims[i+1]] for i in range(self.N) ]

        return out_b



#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):
    a = np.random.randn(1,2)
    b = np.random.randn(1,3)
    m = CMUXDEMUX([2,3])

    print "a:"
    print a
    print 'b:'
    print b
    print

    print "mux:"
    print m.mux(a,b)
    print "demux:"
    print m.demux(m.mux(a,b))
    print


    del m
    m = CMUXDEMUX([2,3], True)
    print "mux:"
    print m.mux(a,b)

    print "demux:"
    n = m.mux(a,b)
    print m.demux(n)

#====================================#
