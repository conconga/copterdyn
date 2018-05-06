# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This script contains the first matemathical kernel simulating
#                the dynamic behaviour of a quadcopter. See documentation 
#                attached.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.


#====================================#
##WWww=--  import section: --=wwWW##
import numpy              as np;
import navfunc            as nav;
import matplotlib.pyplot  as plt;
#import scipy.io           as io;
#import sys
#import scipy              as sp;
#import scipy.io           as io;
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
    y      = q[1]
    z      = q[2]

    c11_1 = [-x, -y, -z]
    c22_2 = [-PARAM['l2'], 0, 0]
    c33_3 = [-PARAM['l3'], 0, 0]
    c44_4 = [-PARAM['l4'], 0, 0]
    c55_5 = [-PARAM['l5'], 0, 0]
    c_plus = c11_1 + c22_2 + c33_3 + c44_4 + c55_5

    return np.asarray(c_plus).reshape((len(c_plus),1))

#====================================#
##WWww=--  vector $cp^+$  --=wwWW##
def fn_cp_plus(qp):
    xp      = qp[0]
    yp      = qp[1]
    zp      = qp[2]

    zeros   = [0,0,0]
    cp_plus = [-xp, -yp, -zp] + zeros + zeros + zeros + zeros

    return np.asarray(cp_plus).reshape((len(cp_plus),1))

#====================================#
##WWww=--  matrix $A^-$  --=wwWW##
def fn_A_minus(quat, g):
    qI21  = quat[0]

    A              = np.zeros((15,15))
    A[0:3,0:3]     = np.eye(3)
    A[3:6,3:6]     = g.Q2C(qI21).T
    A[6:9,6:9]     = g.Q2C(qI21).T
    A[9:12,9:12]   = g.Q2C(qI21).T
    A[12:15,12:15] = g.Q2C(qI21).T

    return A

#====================================#
##WWww=--  matrix $A^+$  --=wwWW##
def fn_A_plus(quat, g):
    qI21  = quat[0]
    qI22  = quat[1]
    qI23  = quat[2]
    qI24  = quat[3]
    qI25  = quat[4]

    A              = np.zeros((15,15))
    A[0:3,0:3]     = g.Q2C(qI21).T
    A[3:6,3:6]     = g.Q2C(qI22).T
    A[6:9,6:9]     = g.Q2C(qI23).T
    A[9:12,9:12]   = g.Q2C(qI24).T
    A[12:15,12:15] = g.Q2C(qI25).T

    return A

#====================================#
##WWww=--  vector $\Omega^-$  --=wwWW##
def fn_Omega_minus(qp, P):
    Omega_minus = dot(P, qp.reshape((qp.shape[0],1)))
    return Omega_minus

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
    kd.assert_infnan(state, 'state')
    return state

#====================================#
##WWww=--  demux state vector  --=wwWW##

def fn_demuxstate(state):
    kd.assert_infnan(state, 'state')

    q     = state[0:10]
    qp    = state[10:20]
    quat  = [list(state[20+(i*4):24+(i*4)]) for i in range(5)]

    # kd.show_matrix_details("q", q, level=2)
    # kd.show_matrix_details("qp", qp, level=2)
    # kd.show_matrix_details("quat", quat, level=2)
    return q, qp, quat

#====================================#
##WWww=--  derivative  --=wwWW##

def fn_deriv(state, t, PARAM):
    """ Calculates de derivative of the multibody model. """ 
    #print "time = %f" % t

    T_minus  = PARAM['T_minus']
    T_plus   = PARAM['T_plus']
    T        = PARAM['T']
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
    Omega_minus = fn_Omega_minus(qp, P)

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
    # kd.assert_infnan(a_b1, "a_b1")
    # kd.assert_infnan(b_b1, "b_b1")
    # kd.assert_infnan(a_b20, "a_b20")
    # kd.assert_infnan(b_b20, "b_b20")

    ##WWww=-- transformation matrixes  --=wwWW##
    RI21  = g.Q2C(quat[0])
    RI22  = g.Q2C(quat[1])
    RI23  = g.Q2C(quat[2])
    RI24  = g.Q2C(quat[3])
    RI25  = g.Q2C(quat[4])

    ##WWww=-- force vector:  --=wwWW##
    F1 = np.zeros((3,1))
    F2 = np.zeros((3,1))
    F3 = np.zeros((3,1))
    F4 = np.zeros((3,1))
    F5 = np.zeros((3,1))

    F4 = np.asarray([0,0,-100]).reshape((3,1))

    F = np.vstack((F1, F2, F3, F4, F5)) # body's frames
    kd.show_matrix_details("F", F, level=1)

    ##WWww=-- moment vector:  --=wwWW##
    M1 = np.zeros((3,1))
    M2 = np.zeros((3,1))
    M3 = np.zeros((3,1))
    M4 = np.zeros((3,1))
    M5 = np.zeros((3,1))

    M4 = np.asarray([0,0,10]).reshape((3,1))

    M = np.vstack((M1, M2, M3, M4, M5)) # body's frames
    kd.show_matrix_details("M", M, level=1)

    ##WWww=--  calc G:  --=wwWW##
    G = dot(a_b1.T, dot(m, a_b1)) + dot(b_b1.T, dot(J, b_b1))
    kd.show_matrix_details("G", G, level=1)
    # kd.assert_infnan(G, "G")

    ##WWww=--  calc H:  --=wwWW##
    omega_b = dot(A_plus.T, omega_plus)
    H_1     = dot(a_b1.T, F - dot(m, a_b20))
    H_2     = dot(b_b1.T, M - dot(J, b_b20) - dot(fn_blockskew(omega_b), dot(J, omega_b)))
    H       = H_1 + H_2
    kd.show_matrix_details("H", H, level=1)
    # kd.assert_infnan(H, 'H')

    ##WWww=--  calc qpp:  --=wwWW##
    qpp = linalg.solve(G,H)
    kd.show_matrix_details("qpp", qpp, level=1)
    # kd.assert_infnan(qpp, 'qpp')

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
    for i in range(5):
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
    kd = KDEBUG(2)

    PARAM = {
            'mass_body': 1,
            'mass_2': 0.1,
            'mass_3': 0.1,
            'mass_4': 0.1,
            'mass_5': 0.1,
            'J_body': np.eye(3),
            'J_2':    0.1 * np.eye(3),
            'J_3':    0.1 * np.eye(3),
            'J_4':    0.1 * np.eye(3),
            'J_5':    0.1 * np.eye(3),
            'l2': 0.1,
            'l3': 0.1,
            'l4': 0.1,
            'l5': 0.1
        }

    # description of joint connections:
    Tm = np.asarray([ # T^-
        [0,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0]
    ])

    TM = np.asarray([ # T^+
        [1,1,1,1,1],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1]
    ])

    T_minus    = fn_elwisemult(Tm,   np.eye(3))
    T_plus     = fn_elwisemult(TM.T, np.eye(3))
    T          = fn_elwisemult(TM,   np.eye(3))

    # angle between body frame and copter arms:
    delta_2_rad = np.radians(0  +45)
    delta_3_rad = np.radians(90 +45)
    delta_4_rad = np.radians(180+45)
    delta_5_rad = np.radians(270+45)

    # misalignment between body's arm and rotor axis:
    alfa_2_rad = np.radians(0)
    alfa_3_rad = np.radians(0)
    alfa_4_rad = np.radians(0)
    alfa_5_rad = np.radians(0)

    # transformation matrices:
    # ( R_from_rotor(i)_to_bodyframe  =  R_from_arm(i)_to_bodyframe . R_from_rotor(i)_to_arm(i) )

    # TODO use quaternions product directly

    q221 = g.C2Q(dot(
        g.Q2C(g.euler2Q((0,0,-delta_2_rad))), # from arm(i) to body frame
        g.Q2C(g.euler2Q((-alfa_2_rad,0,0)))   # from rotor(i) to arm(i)
    ))

    q321 = g.C2Q(dot(
        g.Q2C(g.euler2Q((0,0,-delta_3_rad))), # from arm(i) to body frame
        g.Q2C(g.euler2Q((-alfa_3_rad,0,0)))   # from rotor(i) to arm(i)
    ))

    q421 = g.C2Q(dot(
        g.Q2C(g.euler2Q((0,0,-delta_4_rad))), # from arm(i) to body frame
        g.Q2C(g.euler2Q((-alfa_4_rad,0,0)))   # from rotor(i) to arm(i)
    ))

    q521 = g.C2Q(dot(
        g.Q2C(g.euler2Q((0,0,-delta_5_rad))), # from arm(i) to body frame
        g.Q2C(g.euler2Q((-alfa_5_rad,0,0)))   # from rotor(i) to arm(i)
    ))

    # state vector (generalized coordinates):
    q = np.asarray([
            0,0,0, # initial position
            0,0,0, # body initial angles
            0,0,0,0 # rotor angles
        ])
    qp = np.zeros(q.shape) # first derivative

    # [qI21, qI22, qI23, qI23, qI25]
    # ( Ri2I =  R12I  . Ri21 )
    # (      = eye(3) . Ri21 )

    quat = [
            g.euler2Q((0,0,0)), 
            g.C2Q(g.Q2C(q221).T),
            g.C2Q(g.Q2C(q321).T),
            g.C2Q(g.Q2C(q421).T),
            g.C2Q(g.Q2C(q521).T)
    ]

    # description of joint constraints:
    # \omega^- = P.\dot(q)
    P = np.hstack((np.zeros((3,3)), np.eye(3), np.zeros((3,4))))

    R = g.Q2C(q221)
    P = np.vstack((P, np.hstack((np.zeros((3,6)), -R[:,2].reshape((3,1)), np.zeros((3,3)))) ))

    R = g.Q2C(q321)
    P = np.vstack((P, np.hstack((np.zeros((3,7)), -R[:,2].reshape((3,1)), np.zeros((3,2)) )) ))

    R = g.Q2C(q421)
    P = np.vstack((P, np.hstack((np.zeros((3,8)), -R[:,2].reshape((3,1)), np.zeros((3,1)) )) ))

    R = g.Q2C(q521)
    P = np.vstack((P, np.hstack((np.zeros((3,9)), -R[:,2].reshape((3,1)) )) ))

    kd.show_matrix_details("P", P, level=2)

    # description of joint constraints:
    # \dot(c)^+ = k.\dot(q)
    k           = np.zeros((15,10))
    k[0:3, 0:3] = -np.eye(3)
    kd.show_matrix_details("k", k, level=2)

    # mass:
    m = np.eye(15)
    for i in range(5):
        m[ (3*i):(3*(i+1)), (3*i):(3*(i+1)) ] = [
            PARAM['mass_body'], 
            PARAM['mass_2'],
            PARAM['mass_3'],
            PARAM['mass_4'],
            PARAM['mass_5']
        ][i] * np.eye(3)

    # inertia:
    J = np.eye(15)
    for i in range(5):
        J[ (3*i):(3*(i+1)), (3*i):(3*(i+1)) ] = [
            PARAM['J_body'],
            PARAM['J_2'],
            PARAM['J_3'],
            PARAM['J_4'],
            PARAM['J_5']
        ][i]
    kd.show_matrix_details("J", J, level=2)

    # new parameters:
    PARAM['T_minus'] = T_minus
    PARAM['T_plus']  = T_plus
    PARAM['T']       = T
    PARAM['P']       = P
    PARAM['k']       = k
    PARAM['m']       = m
    PARAM['J']       = J

    # state vector
    state = fn_muxstate((q,qp,quat))

    # diff eq.
    Fs   = 200
    Ts   = 1./Fs
    Tmax = 0.20
    time = Ts * np.asarray(range(1, int(Tmax*Fs)))
    fn_deriv(state, 10, PARAM) # test only, useless

    # for i in range(1,5):
    #     t   = Ts * i
    #     ddt = fn_deriv(state, t, PARAM) # test only, useless
    #     qp,qpp,dquat = fn_demuxstate(ddt)
    #     q += Ts*q
    #     qp += Ts*qp
    #     print dquat

    y = odeint(fn_deriv, state, time, (PARAM,))

    # calculation of position:
    r  = list()
    rp = list()
    for i in range(y.shape[0]):
        q,qp,quat   = fn_demuxstate(y[i,:])
        A_plus      = fn_A_plus(quat,g)
        A_minus     = fn_A_minus(quat,g)
        c_plus      = fn_c_plus(q, PARAM)
        c_plus_X    = fn_blockskew(c_plus)
        Omega_minus = fn_Omega_minus(qp, P)

        r.append(- dot(A_plus, c_plus))

        a1 = dot(T, (dot(A_plus, dot(c_plus_X, dot(A_plus.T, dot(T_plus, dot(A_minus, P))))) - dot(A_plus, k)))
        rp.append(dot(a1, qp))

    # for the figures:
    r    = np.asarray(r).squeeze()
    rp   = np.asarray(rp).squeeze()
    q    = y[:, 0:10]
    qp   = y[:, 10:20]
    quat = y[:, 20:]

    #---- new figure:
    fig = 1;
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('general coordinates body 1')
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
    pfig.canvas.set_window_title('euler')
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(time, 57.*g.matrix_Q2euler(quat[:,(4*i):(4*(i+1))]), hold=False)
        plt.legend(('phi','theta','psi'))
        plt.grid()
        plt.ylabel('[deg]')
        plt.title("body %d" % (i+1))

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('vectors r_I')
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(time, r[:,(i*3):(3*(i+1))], hold=False)
        plt.legend(('x','y','z'))
        plt.grid()
        plt.title("body %d" % (i+1))
        plt.ylabel('[m]')

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('d/dt general coordinates body 1')
    plt.subplot(2,1,1)
    plt.plot(time, qp[:,0:3], hold=False)
    plt.ylabel('dx/dt [m/s]')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(time, qp[:,3:6]*57., hold=False)
    plt.ylabel('d ang/dt [deg/s]')
    plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('quat')
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(time, quat[:, (i*4):(4*(i+1))], hold=False)
        plt.ylim((-1.1, 1.1))
        plt.grid();
        plt.title("body %d" % (i+1))

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('vetors r_I, rp_I')


    #------------------#
    plt.show(block=False);
    #------------------#

#====================================#
