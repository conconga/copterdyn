#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: Control with continuous model.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.
#====================================#



#====================================#
#
#  Model:
#
#      x1 = [ phi,  tta,  psi,    z ]^T
#      x2 = [   p,    q,    r,   zp ]^T
#      x3 = [  uT, uphi, utta, upsi ]^T
#       u = [  u2,   u3,   u4,   u5 ]^T
#
#   x1p =      g1.x2
#
#   x2p = f2 + g2.x3
#
#   x3p = f3 + g3.u
#
#====================================#

import numpy                  as np
import matplotlib.pyplot      as plt
from   cltisatsec         import CLTISATSEC_MIMO
from   rk4                import *
from   scipy.integrate    import odeint
from   numpy              import dot,sin,cos,tan


def g1(x1):
    phi = x1[0]
    tta = x1[1]

    g1 = np.asarray([
        [1.0, sin(phi)*tan(tta), cos(phi)*tan(tta), 0.0],
        [0.0,          cos(phi),         -sin(phi), 0.0],
        [0.0, sin(phi)/cos(tta), cos(phi)/cos(tta), 0.0],
        [0.0,               0.0,               0.0, 1.0]
    ])

    return g1


def g2(x1):
    phi = x1[0]
    tta = x1[1]

    g2 = np.asarray([
        [0.0, 3.66e1, 0.0, 0.0],
        [0.0, 0.0, 1.06e2, 0.0],
        [0.0, 0.0, 0.0, 3.96e1],
        [1.67*cos(phi)*cos(tta), 0.0, 0.0, 0.0]
    ])

    return g2


def f2(x2, x3):
    p  = x2[0]
    q  = x2[1]
    r  = x2[2]
    zp = x2[3]
    gz = 9.8
    Om = x3[0]

    f2 = np.asarray([
        (-3.55e-1*p*q)+(9.27e-1*q*Om),
        (4.73e-2*p*r)-(3.50e-1*p*Om),
        7.41e-2*p*r,
        gz
    ])

    return f2


def f3(x3):
    A = np.asarray([
        [  -2.780e-01 ,   0.000e+00 ,   0.000e+00 ,   0.000e+00 ],
        [   0.000e+00 ,  -1.509e+01 ,   0.000e+00 ,   0.000e+00 ],
        [   0.000e+00 ,   0.000e+00 ,  -6.059e+00 ,   0.000e+00 ],
        [   0.000e+00 ,   0.000e+00 ,   0.000e+00 ,  -1.362e+01 ]
    ])

    return dot(A,x3)


def x123_dmux(x123):
    return [x123[(i*4):((i+1)*4)] for i in range(3)]


def state_dmux(state):
    return (
        state[0:12],  # x123
        state[12:16], # qsi_1
        state[16:20], # qsi_2
        state[20:25], # tta_til_f1
        state[25:30], # tta_til_f2
        state[30:35], # tta_til_f3
    )


def d_model_dt(state, t, x1d, debug=False):
    x123, qsi1, qsi2, tta_til_f1, tta_til_f2, tta_til_f3 = state_dmux(state)
    x1,x2,x3 = x123_dmux(x123)

    _g1 = g1(x1)
    _g2 = g2(x1)
    _f2 = f2(x2,x3)
    _f3 = f3(x3)

    # deinterleave:
    x1c_state_div = X1Cd.deinterleave(X1Cd.get_state())
    x2c_state_div = X2Cd.deinterleave(X2Cd.get_state())
    x3c_state_div = X3Cd.deinterleave(X3Cd.get_state())

    x1c    = x1c_state_div[0:4]
    x1cp   = x1c_state_div[4:8]
    x2c    = x2c_state_div[0:4]
    x2cp   = x2c_state_div[4:8]
    x3c    = x3c_state_div[0:4]
    x3cp   = x3c_state_div[4:8]

    x1_til = x1 - x1c
    x1_bar = x1_til - qsi1
    Phi_f1 = np.tile(np.hstack((1, x1)), (4,1)).T
    f1_hat = dot(Phi_f1.T, tta_til_f1)
    alfa1  = np.linalg.solve(_g1, -f1_hat-dot(K1, x1_til)+x1cp)

    x2_til = x2 - x2c
    x2_bar = x2_til - qsi2
    Phi_f2 = np.tile(np.hstack((1, x2)), (4,1)).T
    f2_hat = _f2 + dot(Phi_f2.T, tta_til_f2)
    alfa2  = np.linalg.solve(_g2, -f2_hat-dot(K2,x2_til)+x2cp-dot(_g1,x1_bar))

    x3_til = x3 - x3c
    Phi_f3 = np.tile(np.hstack((1, x3)), (4,1)).T
    f3_hat = _f3 + dot(Phi_f3.T, tta_til_f3)

    if t < 30:
        DISTURBANCE = 0.5
    else:
        DISTURBANCE = -0.5
    #DISTURBANCE = 0

    u      = np.linalg.solve( g3,  DISTURBANCE    -f3_hat-dot(K3,x3_til)+x3cp-dot(_g2,x2_bar))

    # saturate:
    if False and t > 0.5:
        u = u/np.sqrt(np.sum(u**2))
        u[u<0] = 0
        u[u>1] = 1

    #u   = np.zeros(4)
    #u   = np.asarray([0,0,1,0])

    x1p    =       dot(_g1, x2)
    x2p    = _f2 + dot(_g2, x3)
    x3p    = _f3 + dot( g3, u)

    qsi1p  = -dot(K1,qsi1) + dot(_g1, x2c-alfa1+qsi2)
    qsi2p  = -dot(K2,qsi2) + dot(_g2, x3c-alfa2)

    ttap_til_f1 = dot(Gamma_f1, dot(Phi_f1, x1_bar))
    ttap_til_f2 = dot(Gamma_f2, dot(Phi_f2, x2_bar))
    ttap_til_f3 = dot(Gamma_f3, dot(Phi_f3, x3_til))

    if (False):
    #if (True):
        print "x1p ="
        print x1p
        print "x2p ="
        print x2p
        print "x3p ="
        print x3p
        print "x1cp ="
        print x1cp
        print "x2cp ="
        print x2cp
        print "x3cp ="
        print x3cp

    if debug:
        out = {
            'x1'         : x1,
            'x2'         : x2,
            'x3'         : x3,
            'qsi1'       : qsi1,
            'qsi2'       : qsi2,
            'x1c'        : x1c,
            'x1cp'       : x1cp,
            'x2c'        : x2c,
            'x2cp'       : x2cp,
            'x3c'        : x3c,
            'x3cp'       : x3cp,
            'x1_til'     : x1_til,
            'x1_bar'     : x1_bar,
            'alfa1'      : alfa1,
            'x2_til'     : x2_til,
            'x2_bar'     : x2_bar,
            'alfa2'      : alfa2,
            'x3_til'     : x3_til,
            'u'          : u,
            'x1p'        : x1p,
            'x2p'        : x2p,
            'x3p'        : x3p,
            'qsi1p'      : qsi1p,
            'qsi2p'      : qsi2p,
            'tta_til_f1' : tta_til_f1,
            'tta_til_f2' : tta_til_f2,
            'tta_til_f3' : tta_til_f3,
            'ttap_til_f1': ttap_til_f1,
            'ttap_til_f2': ttap_til_f2,
            'ttap_til_f3': ttap_til_f3,
        }

        return out
    else:
        return np.hstack((x1p,x2p,x3p,qsi1p,qsi2p,ttap_til_f1,ttap_til_f2, ttap_til_f3))


#################################
if (__name__ == "__main__"):

    g3 = np.asarray([
        [  -1.371e+00 ,  -1.371e+00 ,  -1.371e+00 ,  -1.371e+00 ],
        [  -1.192e+00 ,  -1.192e+00 ,   1.192e+00 ,   1.192e+00 ],
        [   1.294e+00 ,  -1.294e+00 ,  -1.294e+00 ,   1.294e+00 ],
        [  -5.351e-01 ,   5.351e-01 ,  -5.351e-01 ,   5.351e-01 ]
    ])

    K1 = 1e0 * np.eye(4)
    K2 = 1e0 * np.eye(4)
    K3 = 1e3 * np.eye(4)

    Gamma_f1 = 100*np.diag([1e-3, 1e-5, 1e-5, 1e-5, 1e-5])
    Gamma_f2 = 0*np.diag([1e-1, 1e-4, 1e-4, 1e-4, 1e-4])
    Gamma_f3 = 0*np.diag([ 1e1, 1e-1, 1e-1, 1e-1, 1e-1])

    #se nao funcionar assim, que tal colocar um \sigma no calculo de \dot{tta} para voltar para o zero?

    # some configuration:
    x123      = np.zeros(12)
    Fs        = 100
    Ts        = 1./Fs
    Tmax      = 200
    Tmax      = 60
    #Tmax      = 30
    #Tmax      = 20
    #Tmax      = 10
    #Tmax      = 5
    #Tmax      = 1
    #Tmax      = 0.5
    #Tmax      = 0.1
    time      = Ts * np.asarray(range(1, int(Tmax*Fs)))
    t0        = 0
    log_x     = []
    log_x1d   = []
    log_u     = []
    log_x1bar = []
    log_x2bar = []
    log_qsi1  = []
    log_qsi2  = []
    log_alfa1 = []
    log_alfa2 = []
    log_ttaf1 = []
    log_ttaf2 = []
    log_ttaf3 = []
    log_ttapf1 = []
    log_ttapf2 = []
    log_ttapf3 = []

    # reference value:
    x1d = np.asarray([0,0,10./57,-10])

    # x1 = [ phi,  tta,  psi,    z ]^T
    X1Cd = CLTISATSEC_MIMO(0.7, 2.*np.pi*1, np.zeros(4),
            np.asarray([-np.inf, -np.inf, -np.inf, -np.inf]),
            np.asarray([ np.inf,  np.inf,  np.inf,  np.inf]),
            np.asarray([-np.inf, -np.inf, -np.inf, -np.inf]),
            np.asarray([ np.inf,  np.inf,  np.inf,  np.inf]),
            Ts=Ts
        )

    # x2 = [   p,    q,    r,   zp ]^T
    X2Cd = CLTISATSEC_MIMO(0.7, 2.*np.pi*1, np.zeros(4),
            np.asarray([-np.inf, -np.inf, -np.inf, -np.inf]),
            np.asarray([ np.inf,  np.inf,  np.inf,  np.inf]),
            np.asarray([-np.inf, -np.inf, -np.inf, -5.0]),
            np.asarray([ np.inf,  np.inf,  np.inf,  5.0]),
            Ts=Ts
        )

    # x3 = [  uT, uphi, utta, upsi ]^T
    X3Cd = CLTISATSEC_MIMO(0.7, 2.*np.pi*100, np.zeros(4),
            np.asarray([-10., -np.inf, -np.inf, -np.inf]),
            np.asarray([ 10.,  np.inf,  np.inf,  np.inf]),
            np.asarray([-15, -np.inf, -np.inf, -np.inf]),
            np.asarray([  0.,  np.inf,  np.inf,  np.inf]),
            Ts=Ts
        )

    # current state:
    state = np.hstack((
        x123,
        np.zeros(8),      # qsi1+qsi2
        np.zeros(5),      # tta_til_f1
        np.zeros(5),      # tta_til_f2
        np.zeros(5),      # tta_til_f3
    ))

    # time update:
    for t in time:
        t = float(t)

        # show time:
        if (t%0.5) < 1e-5:
            print "from %1.03f to %1.03f (max %1.03f) ..." % (t0, t, Tmax)

        # reference value:
        if (t%10.) < 1e-5:
            x1d = np.asarray([
                0.4*((2.0*np.random.rand())-1.) * (np.pi/2.),
                0.4*((2.0*np.random.rand())-1.) * (np.pi/2.),
                ((2.0*np.random.rand())-1.) * (np.pi/2.),
                np.random.rand() * 40.
            ])

            print "new reference:"
            print "x1d =",
            print x1d


        # numerical integration
        _,y   = odeint(d_model_dt, state, [t0,t], (x1d,)) # returns y[t-1] e y[t]
        #y     = rk4(t0, state, t-t0, d_model_dt, (x1d,))

        # pos integration:
        t0     = t
        state  = y
        dstate = d_model_dt(state, t, x1d, debug=True)

        # log:
        log_x.append(state)
        log_x1d.append(x1d)
        log_u.append(dstate['u'])
        log_x1bar.append(dstate['x1_bar'])
        log_x2bar.append(dstate['x2_bar'])
        log_qsi1.append(dstate['qsi1'])
        log_qsi2.append(dstate['qsi2'])
        log_alfa1.append(dstate['alfa1'])
        log_alfa2.append(dstate['alfa2'])
        log_ttaf1.append(dstate['tta_til_f1'])
        log_ttaf2.append(dstate['tta_til_f2'])
        log_ttaf3.append(dstate['tta_til_f3'])
        log_ttapf1.append(dstate['ttap_til_f1'])
        log_ttapf2.append(dstate['ttap_til_f2'])
        log_ttapf3.append(dstate['ttap_til_f3'])

        # references:
        X1Cd.d_update(t, x1d)
        X2Cd.d_update(t, dstate['alfa1']-dstate['qsi2'])
        X3Cd.d_update(t, dstate['alfa2'])

        if False:
            i = ["%10.4e"%(j) for j in
                    X1Cd.get_state().tolist() +
                    X2Cd.get_state().tolist() +
                    X3Cd.get_state().tolist()
                    ]

            print "si,", "  ".join(i)


    log_x     = np.asarray(log_x)
    log_x1d   = np.asarray(log_x1d)
    log_u     = np.asarray(log_u)
    log_x1bar = np.asarray(log_x1bar)
    log_x2bar = np.asarray(log_x2bar)
    log_qsi1  = np.asarray(log_qsi1)
    log_qsi2  = np.asarray(log_qsi2)
    log_alfa1 = np.asarray(log_alfa1)
    log_alfa2 = np.asarray(log_alfa2)
    log_ttaf1 = np.asarray(log_ttaf1)
    log_ttaf2 = np.asarray(log_ttaf2)
    log_ttaf3 = np.asarray(log_ttaf3)
    log_ttapf1 = np.asarray(log_ttapf1)
    log_ttapf2 = np.asarray(log_ttapf2)
    log_ttapf3 = np.asarray(log_ttapf3)

    x1    = log_x[:, 0: 4]
    x2    = log_x[:, 4: 8]
    x3    = log_x[:, 8:12]
    x1c   = log_x[:,12:20]
    x2c   = log_x[:,20:28]
    x3c   = log_x[:,28:36]




    #---- new figure:
    fig = 1;
    r2d = 180./np.pi

    #   legend             scale
    leg = [
        '$\phi$ [deg]',    r2d,
        '$\\theta$ [deg]', r2d,
        '$\psi$ [deg]',    r2d,
        'z [m]',           1.0,
        'p [deg/s]',       r2d,
        'q [deg/s]',       r2d,
        'r [deg/s]',       r2d,
        'zp [m/s]',        1.0,
        '$u_T$',           1.0,
        '$u_{\phi}$',      1.0,
        '$u_{\\theta}$',   1.0,
        '$u_{\psi}$',      1.0,
    ]

    k = 0 # idx in leg
    for i in range(3):
        fig = fig + 1; pfig = plt.figure(fig); plt.clf();
        pfig.canvas.set_window_title('x%d'%(i+1))
        x = eval("x%d"%(i+1))

        for j in range(4):
            plt.subplot(2,2,j+1)
            plt.plot(time, x[:,j]*leg[k+1])
            plt.ylabel(leg[k])

            # references:
            if (i==0):
                plt.plot(time, log_x1d[:,j]*leg[k+1])

            plt.tick_params(labelsize=6)
            plt.grid()
            k += 2


    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('u_2:5')

    plt.plot(time, log_u)
    plt.legend(( '$u_2$', '$u_3$', '$u_4$', '$u_5$' ))
    plt.grid(True)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('x_bar')

    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.plot(time, eval("log_x%dbar"%(i+1)))
        plt.ylabel("x$_%d$bar"%(i+1))
        plt.grid(True)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('qsi')

    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.plot(time, eval("log_qsi%d"%(i+1)))
        plt.ylabel("$\\xi_%d$"%(i+1))
        plt.grid(True)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('alpha')

    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.plot(time, eval("log_alfa%d"%(i+1)))
        plt.ylabel("$\\alpha_%d$"%(i+1))
        plt.grid(True)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('tta_til_fi')

    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(time, eval("log_ttaf%d"%(i+1)))
        plt.ylabel("$\\theta_{f_%d}$"%(i+1))
        plt.grid(True)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('ttap_til_fi')

    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(time, eval("log_ttapf%d"%(i+1)))
        plt.ylabel("$dot(\\theta)_{f_%d}$"%(i+1))
        plt.grid(True)

    #------------------#
    plt.show(block=False);
    #------------------#
