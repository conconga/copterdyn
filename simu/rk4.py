#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: Runge-Kutta ord 4
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.
#====================================#

##WWww=--  import section: --=wwWW##
#import numpy                as np;

def rk4 ( t0, u0, dt, f, PARAM ):
    """
        t0 : current time
        u0 : current state
        dt : time step
        f  : callback to derivtive function dxdt(u,t)
    """

    if isinstance(PARAM, tuple):
        PARAM = PARAM[0]

    f1 = f (u0 ,  t0, PARAM)
    f2 = f (u0 + dt * f1 / 2.0 ,  t0 + dt / 2.0, PARAM)
    f3 = f (u0 + dt * f2 / 2.0 ,  t0 + dt / 2.0, PARAM)
    f4 = f (u0 + dt * f3 ,  t0 + dt, PARAM)

    u1 = u0 + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

    return u1

#====================================#
