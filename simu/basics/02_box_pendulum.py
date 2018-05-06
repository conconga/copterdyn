# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.03.25
#
# \description:
#    In order to get some experience with several necessary tools to model the
#    full quad, this script simulates a box attached to a pendulum.
#
#          +----+
#          |    |
#          |  o-------------
#          |    |
#          +----+
#   -----------------------------
#   / / / / / / / / / / / / / / /


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
from   mpl_toolkits.mplot3d   import axes3d

#====================================#
##WWww=--  vector $c^+$ --=wwWW##
def fn_c_plus(q, PARAM):
    x      = q[0]
    theta  = q[1]
    c_plus = [-x, 0, 0, -PARAM["half_pend"], 0, 0]
    return np.asarray(c_plus).reshape((len(c_plus),1))

#====================================#
##WWww=--  vector $cp^+$  --=wwWW##
def fn_cp_plus(qp):
    xp      = qp[0]
    cp_plus = [-xp, 0, 0, 0, 0, 0]
    return np.asarray(cp_plus).reshape((len(cp_plus),1))

#====================================#
##WWww=--  matrix $A^-$  --=wwWW##
def fn_A_minus(quat, g):
    qI21       = quat[0]

    A          = np.zeros((6,6))
    A[0:3,0:3] = np.eye(3)
    A[3:6,3:6] = g.Q2C(qI21).T
    return A

#====================================#
##WWww=--  matrix $A^+$  --=wwWW##
def fn_A_plus(quat, g):
    qI21       = quat[0]
    qI22       = quat[1]

    A          = np.zeros((6,6))
    A[0:3,0:3] = g.Q2C(qI21).T
    A[3:6,3:6] = g.Q2C(qI22).T
    return A

#====================================#
##WWww=--  vector $\Omega^-$  --=wwWW##
def fn_Omega_minus(qp):
    Omega_minus = [
            0,0,0, \
            0,0,qp[1]
    ]
    return np.asarray(Omega_minus).reshape((len(Omega_minus),1))

#====================================#
##WWww=--  mux state vector  --=wwWW##

def fn_muxstate(T):
    q    = T[0]
    qp   = T[1]
    quat = T[2]
    
    state = np.hstack((q,qp))

    for i in range(len(quat)):
        state = np.hstack((state, quat[i]))

    #kd.show_matrix_details("state", state, level=2)
    return state

#====================================#
##WWww=--  demux state vector  --=wwWW##

def fn_demuxstate(state):
    q     = state[0:2]
    qp    = state[2:4]
    quat  = [list(state[4+(i*4):8+(i*4)]) for i in range(2)]

    # kd.show_matrix_details("q", q, level=2)
    # kd.show_matrix_details("qp", qp, level=2)
    # kd.show_matrix_details("quat", quat, level=2)
    return q, qp, quat

#====================================#
##WWww=--  derivative  --=wwWW##

def fn_deriv(state, t, PARAM):
    """ Calculates de derivative of the multibody model. """ 

    T_minus  = PARAM['T_minus']
    T_plus   = PARAM['T_plus']
    P        = PARAM['P']
    k        = PARAM['k']
    m        = PARAM['m']
    J        = PARAM['J']

    ##WWww=--  state rescue:  --=wwWW##
    q, qp, quat = fn_demuxstate(state)

    ##WWww=--  calc a1:  --=wwWW##
    A_plus      = fn_A_plus(quat,g)
    A_minus     = fn_A_minus(quat, g)
    c_plus      = fn_c_plus(q, PARAM)
    c_plus_X    = fn_blockskew(c_plus)
    Omega_minus = fn_Omega_minus(qp)

    kd.show_matrix_details("T", T, level=2)
    kd.show_matrix_details("A_plus", A_plus, level=2)
    kd.show_matrix_details("c_plus_X", c_plus_X, level=2)
    kd.show_matrix_details("T_plus", T_plus, level=2)
    kd.show_matrix_details("A_minus", A_minus, level=2)
    kd.show_matrix_details("P", P, level=2)

    a1 = dot(T, (dot(A_plus, dot(c_plus_X, dot(A_plus.T, dot(T_plus, dot(A_minus, P))))) - dot(A_plus, k)))
    kd.show_matrix_details("a1", a1, level=2)

    ##WWww=--  calc a20:  --=wwWW##
    omega_minus     = dot(T_minus, dot(A_minus, Omega_minus))
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
    kd.show_matrix_details("a20", a20, level=2)

    ##WWww=--  calc b1:  --=wwWW##
    b1 = dot(T, dot(A_minus, P))
    kd.show_matrix_details("b1", b1, level=2)

    ##WWww=--  calc b20:  --=wwWW##
    b20 = dot(T, dot(A_minus, dot(omega_miminus_X, Omega_minus)))
    kd.show_matrix_details("b20", b20, level=2)

    ##WWww=--  calc body matrixes:  --=wwWW##
    a_b1   = dot(A_plus.T, a1)
    b_b1   = dot(A_plus.T, b1)
    a_b20  = dot(A_plus.T, a20)
    b_b20  = dot(A_plus.T, b20)
    kd.show_matrix_details("a_b1", a_b1, level=3)
    kd.show_matrix_details("b_b1", b_b1, level=3)
    kd.show_matrix_details("a_b20", a_b20, level=3)
    kd.show_matrix_details("b_b20", b_b20, level=3)

    ##WWww=-- transformation matrixes  --=wwWW##
    RI22  = g.Q2C(quat[1])

    ##WWww=-- force vector:  --=wwWW##
    B     = 2.0 * qp[0] # damping (N/(m/s))
    F1    = np.asarray([-B, (-PARAM['mass_box']*9.81), 0]).reshape((3,1))
    F2    = dot(RI22, np.asarray([0, (-PARAM['mass_pend']*9.81), 0]).reshape((3,1)))

    F = np.vstack((F1, F2))
    kd.show_matrix_details("F", F, level=1)

    ##WWww=-- moment vector:  --=wwWW##
    M = np.zeros((6,1))

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

        kd.show_matrix_details("rp",  rp,  level=1)
        kd.show_matrix_details("rpp", rpp, level=1)
        kd.show_matrix_details("w",   w,   level=1)
        kd.show_matrix_details("wp",  wp,  level=1)

    ##WWww=--  derivative of the quaternions  --=wwWW##
    quatp = list()
    for i in range(2):
        bomega = omega_plus[range(3*i, 3*(i+1)), :]
        bA     = A_plus[range(3*i, 3*(i+1))][:,range(3*i, 3*(i+1))]
        kd.show_matrix_details("bomega", bomega, level=3)
        kd.show_matrix_details("bA", bA, level=3)

        quatp.append(g.dqdt(quat[i], dot(bA.T, bomega)))

    state = fn_muxstate((qp, qpp.squeeze(), quatp))
    kd.show_matrix_details("[qp,qpp,quatp]", state, level=1)
    return state

#====================================#
##WWww=--  main:  --=wwWW##
if (__name__ == "__main__"):
    g  = nav.NAVFUNC()
    kd = KDEBUG(0)

    PARAM = {
            'half_pend': 1,      # distance from joint to GC at body 2
            'mass_box': 1,       # mass body 1
            'mass_pend': 0.4,    # mass body 2
            'J_box': np.eye(3),
            'J_pend': 0.33e-6*np.eye(3)
        }

    # x position and pendulum angle:
    q          = np.asarray([0., 0.])
    qp         = np.zeros(q.shape) # first derivative

    # description of joint connections:
    T_minus    = fn_elwisemult(np.asarray([ [0,0], [1,0] ]), np.eye(3))
    T_plus     = fn_elwisemult(np.asarray([ [1,0], [1,1] ]), np.eye(3))
    T          = T_plus

    # description of joint constraints:
    P          = np.zeros((6,2))
    P[5,1]     = 1
    kd.show_matrix_details("P", P, level=2)

    k          = np.zeros((6,2))
    k[0,0]     = -1
    kd.show_matrix_details("k", k, level=2)

    # mass:
    m = np.eye(6)
    for i in range(2):
        m[ (3*i):(3*(i+1)), (3*i):(3*(i+1)) ] = [PARAM['mass_box'], PARAM['mass_pend']][i] * np.eye(3)

    # inertia:
    J = np.eye(6)
    for i in range(2):
        J[ (3*i):(3*(i+1)), (3*i):(3*(i+1)) ] = [PARAM['J_box'], PARAM['J_pend']][i]
    kd.show_matrix_details("J", J, level=2)

    # quaternions for A(1 to i) and A(2 to i):
    quat = [g.euler2Q((0,0,0)), g.euler2Q((0,0,0))]

    # new parameters:
    PARAM['T_minus'] = T_minus
    PARAM['T_plus']  = T_plus
    PARAM['P']       = P
    PARAM['k']       = k
    PARAM['m']       = m
    PARAM['J']       = J

    # state vector
    state = fn_muxstate((q,qp,quat))

    # diff eq.
    Fs   = 200
    Ts   = 1./Fs
    Tmax = 20
    time = Ts * np.asarray(range(1, int(Tmax*Fs)))
    fn_deriv(state, 10, PARAM) # test only, useless
    y = odeint(fn_deriv, state, time, (PARAM,))

    # calculation of position:
    r = list()
    rp = list()
    for i in range(y.shape[0]):
        q,qp,quat   = fn_demuxstate(y[i,:])
        A_plus      = fn_A_plus(quat,g)
        A_minus     = fn_A_minus(quat,g)
        c_plus      = fn_c_plus(q, PARAM)
        c_plus_X    = fn_blockskew(c_plus)
        Omega_minus = fn_Omega_minus(qp)

        r.append(- dot(A_plus, c_plus))

        a1 = dot(T, (dot(A_plus, dot(c_plus_X, dot(A_plus.T, dot(T_plus, dot(A_minus, P))))) - dot(A_plus, k)))
        rp.append(dot(a1, qp))

    # for the figures:
    r    = np.asarray(r).squeeze()
    rp   = np.asarray(rp).squeeze()
    q    = y[:, 0:2]
    qp   = y[:, 2:4]
    quat = y[:, 4:]


    #---- new figure:
    fig = 1;
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('general coordinates')
    plt.subplot(2,1,1)
    plt.plot(time, q[:,0], hold=False)
    plt.ylabel('x [m]')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(time, q[:,1]*57., hold=False)
    plt.ylabel('ang [deg]')
    plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('d/dt general coordinates')
    plt.subplot(2,1,1)
    plt.plot(time, qp[:,0], hold=False)
    plt.ylabel('dx/dt [m/s]')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(time, qp[:,1]*57., hold=False)
    plt.ylabel('d ang/dt [deg/s]')
    plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('quat')
    plt.subplot(2,1,1)
    plt.plot(time, quat[:, 0:4], hold=False)
    plt.ylim((-1.1, 1.1))
    plt.grid();
    plt.subplot(2,1,2)
    plt.plot(time, quat[:, 4:8], hold=False)
    plt.ylim((-1.1, 1.1))
    plt.grid();

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('vetors r_I, rp_I')
    for i in range(2):
        plt.subplot(2,2,i+1);
        plt.plot(time, r[:, range(i*3, (i*3)+3)], hold=False)
        plt.legend(('x','y','z'))
        plt.grid()
        plt.ylabel('[m]')
        plt.title("body %d" % (i+1))

        plt.subplot(2,2,3+i);
        plt.plot(time, rp[:, range(i*3, (i*3)+3)], hold=False)
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
    pfig.canvas.set_window_title('3D damping')
    ax = pfig.add_subplot(111, projection='3d')
    ax.plot_wireframe(time, r[:,3], r[:,4], rstride=5, cstride=5)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('X [m]')
    ax.set_zlabel('Y [m]')
    ax.set_title('body 2')

    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()

    #------------------#
    plt.show(block=False);
    #------------------#

#====================================#
