#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This class simulates the generation of thrust and moments of a
#               propeller. The model has 2 inputs and 3 outputs. The inputs are:
#  
#          INPUTS:
#              u : [0<u<1] throttle command
#              w : [rad/s] angular speed of propeller relative to wind
#
#          OUTPUTS:
#              Telet   : [Nm] electrical torque dependent on 'u'
#              Tload   : [Nm] load torque dependent on 'w'
#              Fthrust : [N] thrust dependent on 'w'
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy             as np
from   cltisatsec        import CLTISATSEC

#====================================#
## \brief class cprop               ##
## \author luciano kruk             ##
##                                  ##
## \parameters:                     ##
#====================================#

class CPROP:
    # permanent:

    def __init__(self, isCounterClockWise=False):
        qsi        = 0.7
        wn         = 2.*3.14*1

        # constants:
        self.Kelet   = 20.         # [N/u], 0<u<1
        self.Kload   = 7./1000.;   # [Nm/(rad/s)]
        self.Kthrust = 11./1000.;  # [N/(rad/s)]

        # some signal convention:
        if isCounterClockWise:
            # torque aligned with -z:
            self.dir_torque = -1.0

            # torque electrical for CCW (only negative):
            self.Telet  = CLTISATSEC(qsi, wn, 0, -1e5, 1e5, -1e5, 0)  # [Nm]
        else:
            # torque aligned with z:
            self.dir_torque = 1.0

            # torque electrical for CW (only positive):
            self.Telet  = CLTISATSEC(qsi, wn, 0, -1e5, 1e5, 0, 1e5)  # [Nm]


        # internal state:
        self.state  = self.Telet.get_state_asarray()
        self.u      = 0
        self.w      = 0

    def dstate_dt(self, x, t):
        return np.asarray(self.Telet.dstate_dt(x,t,self.u*self.dir_torque*self.Kelet))

    # 0.0 <= u <= 1.0
    def pre_update(self, t, u, w):
        #assert(0.0<=u and u<=1.0)

        self.u = u
        self.w = w

    def pos_update(self, t, x):
        self.Telet.pos_update(t,x)
        self.state = x

        if False:
            Telet   = self.state[0]
            Tload   = self.w * self.Kload
            print "Telet = %20.15e; w = %20.15e; Tload = %20.15e" % (Telet, self.w, Tload)

    def get_FT(self):
        """
        Reference:
            "Modeling and Simulation of a Propeller-Engine System for 
            Unmanned Aerial Vehicles", Martinez-Alvarado, R. et al.

        From the reference, eq. (1):

                    T_elet = T_J + T_B + T_L

        where:
         T_elet = electrical torque, dependent on current 'i', or command 'u' here.
         T_J    = mechanical torque calculated by multibody simulation, J.dw/dt.
         T_B    = resistance torque, from damping.
         T_L    = load torque, aerodynamic.

        Rewriting and isolating T_J, we have:

                    T_J = T_elet - T_B - T_L
        """

        Telet   = self.state[0]
        Tload   = self.w * self.Kload
        Fthrust = abs(self.w) * self.Kthrust # always aligned with +z

        #return Fthrust,(self.dir_torque*max(Telet-Tload, 0))
        return Fthrust,(Telet-Tload)


#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):

    from   scipy.integrate    import odeint
    import matplotlib.pyplot  as plt


    prop = CPROP()

    T   = np.linspace(0,5,100)
    U   = (T>2)*1.
    W   = T*1000./5
    t0  = 0
    buf = list()

    for t,u,w in zip(T,U,W):
        # pre:
        prop.pre_update(t,u,w)

        # integration:
        _,y = odeint(prop.dstate_dt, prop.state, [t0, t])

        # pos:
        t0 = t
        prop.pos_update(t, y)

        # buffer:
        buf.append(prop.get_FT())

    plt.figure()

    plt.plot(T, buf, T, U)
    plt.legend(('thrust', 'torque total', 'throttle'))
    plt.grid()

    plt.show(block=False)

#====================================#
