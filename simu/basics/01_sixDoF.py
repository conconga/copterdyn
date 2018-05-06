# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.03.18
#
# \description:
#    In order to get some experience with several necessary tools to model the
#    full quad, this script simulates a body free with 6 DoF.
#


#====================================#
##WWww=--  import section: --=wwWW##
import numpy              as np;
import navfunc            as nav;
#import scipy              as sp;
#import scipy.io           as io;
import matplotlib.pyplot  as plt;
#import scipy.io           as io;
#import sys;
from   numpy           import dot;
from   numpy           import linalg;
from   kdebug          import KDEBUG;
from   scipy.integrate import odeint;
from   somemath        import *

#====================================#
##WWww=--  vector $c^+$ --=wwWW##
def fn_c_plus(q, PARAM):
    c_plus = q[0:3]
    return np.asarray(c_plus).reshape((len(c_plus),1))

#====================================#
##WWww=--  vector $cp^+$  --=wwWW##
def fn_cp_plus(qp):
    cp_plus = qp[0:3]
    return np.asarray(cp_plus).reshape((len(cp_plus),1))

#====================================#
##WWww=--  matrix $A^-$  --=wwWW##
def fn_A_minus():
    A = np.eye(3)
    return A

#====================================#
##WWww=--  matrix $A^+$  --=wwWW##
# (A+ = Rb2I)
def fn_A_plus(qI2b, g):
    A = g.Q2C(qI2b).T
    return A

#====================================#
##WWww=--  vector $\Omega^-$  --=wwWW##
def fn_Omega_minus(qp):
    Omega_minus = qp[3:6]
    return np.asarray(Omega_minus).reshape((len(Omega_minus),1))

#====================================#
##WWww=--  mux state vector  --=wwWW##

def fn_muxstate(T):
    q    = T[0].squeeze()
    qp   = T[1].squeeze()
    qI2b = np.asarray(T[2]).squeeze()

    if (False):
        print type(q)
        print q
        print type(qp)
        print qp
        print type(qI2b)
        print qI2b
    
    state = np.hstack((q,qp))
    state = np.hstack((state, np.asarray(qI2b)))

    #kd.show_matrix_details("state", state, level=1)
    return state.squeeze()

#====================================#
##WWww=--  demux state vector  --=wwWW##

def fn_demuxstate(state):
    q     = state[0:6]
    qp    = state[6:12]
    qI2b  = state[12:]

    # kd.show_matrix_details("q", q, level=1)
    # kd.show_matrix_details("qp", qp, level=1)
    # kd.show_matrix_details("qI2b", qI2b, level=1)
    return q, qp, qI2b

#====================================#
##WWww=--  derivative  --=wwWW##

def fn_deriv(state, t, PARAM):
    """ Calculates de derivative of the multibody model. """ 

    T_plus   = PARAM['T_plus']
    P        = PARAM['P']
    k        = PARAM['k']
    m        = PARAM['m']
    J        = PARAM['J']

    ##WWww=--  state rescue:  --=wwWW##
    q, qp, qI2b = fn_demuxstate(state)

    ##WWww=--  calc a1:  --=wwWW##
    A_plus      = fn_A_plus(qI2b,g)
    A_minus     = fn_A_minus()
    c_plus      = fn_c_plus(q, PARAM)
    c_plus_X    = fn_blockskew(c_plus)
    Omega_minus = fn_Omega_minus(qp)

    kd.show_matrix_details("T", T, level=1)
    kd.show_matrix_details("A_plus", A_plus, level=1)
    kd.show_matrix_details("c_plus_X", c_plus_X, level=1)
    kd.show_matrix_details("T_plus", T_plus, level=1)
    kd.show_matrix_details("A_minus", A_minus, level=1)
    kd.show_matrix_details("P", P, level=1)

    # temp:
    #a1_1 = dot(A_plus, dot(c_plus_X, dot(A_plus.T, P)))
    #a1_2 = -dot(A_plus, k)
    #kd.show_matrix_details("a1_1", a1_1, level=1)
    #kd.show_matrix_details("a1_2", a1_2, level=1)

    a1 = dot(T, (dot(A_plus, dot(c_plus_X, dot(A_plus.T, dot(T_plus, dot(A_minus, P))))) - dot(A_plus, k)))
    kd.show_matrix_details("a1", a1, level=1)

    ##WWww=--  calc a20:  --=wwWW##
    omega_minus     = dot(A_minus, Omega_minus)
    omega_miminus   = dot(A_minus.T, omega_minus)
    omega_miminus_X = fn_blockskew(omega_miminus)

    omega_plus      = dot(T_plus,  dot(A_minus, Omega_minus))
    omega_pluplus   = dot(A_plus.T, omega_plus)
    omega_pluplus_X = fn_blockskew(omega_pluplus)

    cp_plus = fn_cp_plus(qp)

    a20_1 = -dot(A_plus, dot(omega_pluplus_X, dot(omega_pluplus_X, c_plus)))
    a20_2 = dot(A_plus, dot(c_plus_X, dot(A_plus.T, dot(T_plus, dot(A_minus, dot(omega_miminus_X, Omega_minus))))))
    a20_3 = -2.0 * dot(A_plus, dot(omega_pluplus_X, cp_plus))
    a20   = dot(T, a20_1 + a20_2 + a20_3)
    #kd.show_matrix_details("a20_1", a20_1, level=1)
    #kd.show_matrix_details("a20_2", a20_2, level=1)
    #kd.show_matrix_details("a20_3", a20_3, level=1)
    kd.show_matrix_details("a20", a20, level=1)

    ##WWww=--  calc b1:  --=wwWW##
    b1 = P
    kd.show_matrix_details("b1", b1, level=1)

    ##WWww=--  calc b20:  --=wwWW##
    b20 = dot(omega_miminus_X, Omega_minus)
    kd.show_matrix_details("b20", b20, level=1)

    ##WWww=--  calc body matrixes:  --=wwWW##
    a_b1   = dot(A_plus.T, a1)
    b_b1   = dot(A_plus.T, b1)
    a_b20  = dot(A_plus.T, a20)
    b_b20  = dot(A_plus.T, b20)
    kd.show_matrix_details("a_b1", a_b1, level=1)
    kd.show_matrix_details("b_b1", b_b1, level=1)
    kd.show_matrix_details("a_b20", a_b20, level=1)
    kd.show_matrix_details("b_b20", b_b20, level=1)

    ##WWww=-- force vector:  --=wwWW##
    F = np.zeros((3,1))
    if (t > 6) and (t < 7):
        F = np.asarray([1,1,1]).reshape((3,1)) # inertial frame
    RI2b = g.Q2C(qI2b)
    F = dot(RI2b, F) # body frame
    kd.show_matrix_details("F", F, level=1)

    ##WWww=-- moment vector:  --=wwWW##
    M = np.zeros((3,1))
    if (t < 5):
        M = np.asarray([0.1,0,0.1]).reshape((3,1))
    if (t > 15) and (t < 16):
        M = np.asarray([0,0,0.1]).reshape((3,1))
    M = dot(RI2b, M) # body frame
    kd.show_matrix_details("M", M, level=1)

    ##WWww=--  calc G:  --=wwWW##
    G = dot(a_b1.T, dot(m, a_b1)) + dot(b_b1.T, dot(J, b_b1))
    kd.show_matrix_details("G", G, level=1)

    ##WWww=--  calc H:  --=wwWW##
    omega_b = dot(A_plus.T, omega_plus)
    H_1     = dot(a_b1.T, F - dot(m, a_b20))
    H_2     = dot(b_b1.T, M - dot(J, b_b20) - dot(fn_blockskew(omega_b), dot(J, omega_b)))
    H       = H_1 + H_2
    kd.show_matrix_details("H", H, level=1)

    ##WWww=--  calc qpp:  --=wwWW##
    qpp = linalg.solve(G,H)
    kd.show_matrix_details("qpp", qpp, level=1)

    ##WWww=--  only for debug:  --=wwWW##
    if True:
        rp   = dot(a1, qp)
        rpp  = a20 + dot(a1, qpp)
        w    = dot(b1, qp)
        wp   = b20 + dot(b1, qpp)

        kd.show_matrix_details("rp",  rp, level=1)
        kd.show_matrix_details("rpp", rpp, level=1)
        kd.show_matrix_details("w",   w, level=1)
        kd.show_matrix_details("wp",  wp, level=1)

    ##WWww=--  derivative of the quaternions  --=wwWW##
    kd.show_matrix_details("qI2b", qI2b, level=1)
    qI2bp = g.dqdt(qI2b, dot(A_plus.T, omega_plus))
    kd.show_matrix_details("qI2bp", qI2bp, level=1)

    return fn_muxstate((qp, qpp, qI2bp))

#====================================#
##WWww=--  main:  --=wwWW##

if (__name__ == "__main__"):
    g  = nav.NAVFUNC()
    kd = KDEBUG(0)

    PARAM = {
            'mass': 1,
            'J': np.eye(3)
        }

    # initial equations:
    q  = np.zeros((6,1))
    q  = np.asarray([0,0,0,  0,0,0]).reshape((6,1))
    qp = np.zeros(q.shape) # first derivative
    qp = np.asarray([0,0,0,  0*5./57,0,0]).reshape((6,1))

    # description of joint connections:
    T_plus     = np.eye(3)
    T          = T_plus.copy()

    # description of joint constraints:
    P = np.hstack((np.zeros((3,3)), np.eye(3)))
    kd.show_matrix_details("P", P, level=1)

    k = np.hstack((np.eye(3), np.zeros((3,3))))
    kd.show_matrix_details("k", k, level=1)

    # mass and inertia:
    m = PARAM['mass'] * np.eye(3)

    # quaternions
    qI2b = np.asarray(g.euler2Q((0,0,0))).reshape((4,1))

    # new parameters:
    PARAM['T_plus'] = T_plus
    PARAM['P']      = P
    PARAM['k']      = k
    PARAM['m']      = m

    # state vector
    state = fn_muxstate((q,qp,qI2b))

    # diff eq.
    Fs   = 200
    Ts   = 1./Fs
    Tmax = 20.
    time = Ts * np.asarray(range(1, int(Tmax*Fs)))
    fn_deriv(state, 0, PARAM)
    #time = np.asarray([float(i)/Fs for i in range(int(Tmax*Fs))])
    y = odeint(fn_deriv, state, time, (PARAM,), ixpr=True)

    q    = y[:, 0:6]
    qp   = y[:, 6:12]
    qI2b = y[:, 12:]

    # calculation of position:
    r = list()
    rp = list()
    for i in range(q.shape[0]):
        A_plus      = fn_A_plus(qI2b[i,:],g)
        A_minus     = fn_A_minus()
        c_plus      = fn_c_plus(q[i,:], PARAM)
        c_plus_X    = fn_blockskew(c_plus)
        Omega_minus = fn_Omega_minus(qp[i,:])

        RI2b = g.Q2C(qI2b[i,:])
        r.append(- dot(RI2b.T, c_plus))

        a1 = dot(T, (dot(A_plus, dot(c_plus_X, dot(A_plus.T, dot(T_plus, dot(A_minus, P))))) - dot(A_plus, k)))
        rp.append(dot(a1, qp[i,:]))

    r = np.asarray(r).squeeze()
    rp = np.asarray(rp).squeeze()

    #---- new figure:
    fig = 1;
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('general coordinates')
    plt.subplot(2,1,1)
    plt.plot(time, q[:,0:3], hold=False)
    plt.ylabel('pos [m]')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(time, q[:,3:6]*57., hold=False)
    plt.ylabel('ang [deg]')
    plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('d/dt general coordinates')
    plt.subplot(2,1,1)
    plt.plot(time, qp[:,0:3], hold=False)
    plt.ylabel('d pos/dt [m/s]')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(time, qp[:,3:6]*57., hold=False)
    plt.ylabel('d ang/dt [deg/s]')
    plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('euler')
    plt.plot(time, 57.*g.matrix_Q2euler(qI2b[:, 0:4]), hold=False)
    plt.legend(('phi','theta','psi'))
    plt.grid()
    plt.ylabel('[deg]')

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('vetors r_I, rp_I')
    plt.subplot(2,1,1);
    plt.plot(time, r, hold=False)
    plt.legend(('x','y','z'))
    plt.grid()
    plt.ylabel('[m]')
    plt.subplot(2,1,2);
    plt.plot(time, rp, hold=False)
    plt.legend(('xp','yp','zp'))
    plt.grid()
    plt.ylabel('[m]')

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('norm(velocity)')
    plt.plot(time, np.sqrt(np.sum(rp**2, axis=1)), hold=False)
    plt.ylabel('[m/s]')
    plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('quaternions')
    plt.plot(time, qI2b, hold=False)
    plt.grid()

    #------------------#
    plt.show(block=False);
    #------------------#

#====================================#
