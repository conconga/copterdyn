#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This files beautify a matrix in cmdline in a way it can easily be
#               imported to code.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy                as np;

class CMATRIXBEAUTIFY:
    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, M):
        s = M.shape
        if len(s) > 2:
            print "dimension not yet supported"
            assert(False)

        for i in range(s[0]):
            print "[ ",
            for j in range(s[1]):
                print self.fmt % M[i,j],

                if (s[1]-j) > 1:
                    print ", ",

            if (s[0]-i) == 1:
                print "]"
            else:
                print "],"


#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):
    a = np.random.randn(5,5)
    b = CMATRIXBEAUTIFY("%10.3e")
    b(a)

#################################
