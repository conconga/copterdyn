#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This file contains an implementation of the forgetting-factor
#               RLS algorithm, with an unit test of first order continuos
#               transfer function.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy               as np
import f1ord               as f1
from   numpy           import dot
from   kdebug          import CKDEBUG

###########################
## Forgetting Factor RLS ##
###########################
class CFFRLS(CKDEBUG):
    """
    y[k] = x[k]' . theta
    with:
       x[k] = [Nx1]

       lbd = forgetting factor (0<lbd<=1)
    """

    def __init__(self, theta_0, P_0=None, lbd=1., dblevel=10):

        self.n       = len(theta_0)
        self.lbd     = lbd 
        self.theta   = np.asarray(theta_0).reshape((self.n,1))
        self.dblevel = dblevel

        if P_0==None:
            self.P = np.eye(self.n)
        else:
            self.P = P_0

        CKDEBUG.__init__(self,1)

    def update(self, y, x):
        """
        y = x^T . theta
        """

        x = x.reshape((self.n,1))

        gamma       = dot(self.P, x) / (self.lbd + dot(dot(x.T, self.P),x))
        self.show_matrix_details("gamma", gamma, self.dblevel)

        self.theta  = self.theta + dot(gamma, y - dot(x.T, self.theta))
        self.show_matrix_details("theta", self.theta, self.dblevel)

        self.P      = (self.P - (gamma*dot(x.T,self.P)))/self.lbd
        self.show_matrix_details("P", self.P, self.dblevel)


#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):

    from   scipy.integrate    import odeint
    import matplotlib.pyplot      as plt;

    a    = -15.  # pole for the transfer function:
    tmax = 2.0
    Ts   = 5e-3 # sample rate
    T    = [i*Ts for i in range(int(tmax/Ts)+1)] # time vector
    sys  = f1.CF1ORD_D(a, Ts, 1) # unknown system
    rls  = CFFRLS([0.,0.,0.,0.], lbd=0.85)

    log_x     = [0]
    log_y     = [sys.y]
    log_theta = [rls.theta.squeeze().tolist()]

    for t in T:
        if (not (t%0.5)):
            x = float(np.random.rand()) # plant input

        sys.update(t, x)
        rls.update(-0.55+sys.y, np.asarray([
            1.,
            x,
            log_x[-1],
            log_y[-1]
        ]))
        
        log_x.append(float(x))
        log_y.append(sys.y)
        log_theta.append(rls.theta.squeeze().tolist())

    T.append(tmax)
    

    #.............................................#
    #---- new figure:
    fig = 1;
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('continuous')

    plt.plot(T, log_x, T, log_y, hold=False)
    plt.grid(True)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('theta')

    plt.plot(
            T, log_theta, 
            plt.xlim(), (sys.b, sys.b), 'k--', 
            plt.xlim(), (sys.c, sys.c), 'k--',
            plt.xlim(), (-0.55, -0.55), 'k--',
            hold=True)

    plt.grid(True)

    #.............................................#
    plt.show(block=False)
    #.............................................#
