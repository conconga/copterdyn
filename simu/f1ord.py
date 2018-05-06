#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This file contains two implementations of a first order filter:
#               1) continuous mode;
#               2) discrete mode.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy                  as np;
from   scipy.integrate    import odeint

###################################
## First Order Continuous System ##
###################################
class CF1ORD_C:

    def __init__(self, a, y0):
        """
        T(s) = -a/(s-a)
        """

        self.a = a
        self.y = y0
        self.t = 0.

    def dydt(self, y, t, x_):
        """
        dydt(t) = a.y(t) - a.x(t)
        """

        x = x_
        return self.a*(y-x)

    def update(self, t, x):
        _,y = odeint(self.dydt, self.y, [self.t, t], (x,)) # returns y[t-1] e y[t]
        self.y = float(y)
        self.t = t

    def y(self):
        return self.y


#################################
## First Order Discrete System ##
#################################
class CF1ORD_D:

    def __init__(self, a, Ts, y0):
        """
        y[t] = k*( b.x[t] - b.x[t-1] + c.y[t-1] )
        with:
          b = -a.T
          c = (2+a.T)
          k = 1/(2-a.T)
        """

        k = 2.-(a*Ts)

        self.b = -a*Ts/k
        self.c = (2.+(a*Ts))/k
        self.y = y0
        self.x = 0.
        self.t = 0.

    def update(self, t, x):
        self.y = (self.b*x) + (self.b*self.x) + (self.c*self.y)
        self.t = t
        self.x = x

    def y(self):
        return self.y


#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):

    import matplotlib.pyplot      as plt;

    a    = -10.  # pole for the transfer function:
    tmax = 2.0
    Ts   = 5e-3 # sample rate
    T    = [i*Ts for i in range(int(tmax/Ts)+1)] # time vector
    f1c  = CF1ORD_C(a, 0)
    f1d  = CF1ORD_D(a, Ts, 1)

    log_x  = list()
    log_yc = list()
    log_yd = list()
    for t in T:
        if (not (t%0.5)):
            x = float(np.random.rand()) # plant input

        f1c.update(t, x)
        f1d.update(t, x)
        log_x.append(x)
        log_yc.append(f1c.y)
        log_yd.append(f1d.y)
    

    #.............................................#
    #---- new figure:
    fig = 1;
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('continuous')

    plt.plot(T, log_x, T, log_yc, T, log_yd, hold=False)
    plt.grid(True)
    plt.legend(('reference', 'continuous', 'discrete'))

    #.............................................#
    plt.show(block=False)
    #.............................................#
