#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This files contains the system description as well as a
#               complete simulation of a quadcopter.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy                as np;
import cshowquad3d          as sq;
from   rk4              import *
from   scipy.integrate  import odeint
from   cmyquad          import CMYQUAD

#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):
    import matplotlib.pyplot      as plt;
    from   mpl_toolkits.mplot3d   import axes3d

    # instance of quadcopter object:
    quad = CMYQUAD()

    # state vector
    state = quad.state

    # diff eq.
    Fs   = 100
    Ts   = 1./Fs
    Tmax = 20
    #Tmax = 2
    #Tmax = 1
    #Tmax = 0.5
    #Tmax = 0.1
    time = Ts * np.asarray(range(1, int(Tmax*Fs)))
    dstate_dt = quad.dsys_dt(state, 10) # test only, useless
    t0   = 0

    # a step into the future!
    # time update: 
    for t in time:
        t = float(t)
        print "from %1.03f to %1.03f ..." % (t0, t)

        # pre-hook section
        quad.pre_update(t)

        # numerical integration
        _,y = odeint(quad.dsys_dt, quad.state, [t0, t]) # returns y[t-1] e y[t]
        #_,y = odeint(quad.dsys_dt, quad.state, [t0, t], hmax=1e-4, ixpr=True, atol=1e-13, rtol=1e-13) # returns y[t-1] e y[t]
        #y = rk4(t0, quad.state, t-t0, quad.dsys_dt)
        t0  = t

        # pos-hook section
        quad.pos_update(t,y)

    # for the figures:
    # calculation of position and velocity:
    r,rp = quad.dynquad.calc_pos_vel()

    # recover all generalized coordinates:
    q, qp, quat = quad.dynquad.state_to_q_qp_quat()

    # recover forces and moments:
    t,F,M = quad.dynquad.forcemoments_fetch()
    F     = np.asarray(F).squeeze()
    M     = np.asarray(M).squeeze()

    # recover all \Omega^-
    Omin  = np.asarray([quad.dynquad.Omega_minus(qp[i,:]) for i in range(qp.shape[0])]).squeeze()

    #---- new figure:
    fig = 1;
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('general coordinates')
    j = [1,2,3,5,6,7,9,10,11,12]
    k = [
        'x [m]',
        'y [m]',
        'z [m]',
        'ang_1 [deg]',
        'ang_2 [deg]',
        'ang_3 [deg]',
        'rot_1 [deg]',
        'rot_2 [deg]',
        'rot_3 [deg]',
        'rot_4 [deg]'
    ]

    for i in range(10):
        plt.subplot(3,4,j[i])
        if i >= 3:
            plt.plot(time, q[:,i]*180./3.14159, hold=False)
        else:
            plt.plot(time, q[:,i], hold=False)

        plt.legend(
            (k[i],), 
            loc='best',
            fancybox=True,
            framealpha=0.5,
            fontsize=8
        )

        plt.tick_params(labelsize=6)
        plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('d/dt general coordinates')
    j = [1,2,3,5,6,7,9,10,11,12]
    k = [
        'x [m/s]',
        'y [m/s]',
        'z [m/s]',
        'ang_1 [deg/s]',
        'ang_2 [deg/s]',
        'ang_3 [deg/s]',
        'rot_1 [rpm]',
        'rot_2 [rpm]',
        'rot_3 [rpm]',
        'rot_4 [rpm]'
    ]

    for i in range(10):
        plt.subplot(3,4,j[i])
        if i >= 6:
            # [rpm]
            plt.plot(time, qp[:,i]*60/6.28, hold=False)
        elif i>=3:
            # [deg/s]
            plt.plot(time, qp[:,i]*180./3.14159, hold=False)
        else:
            # [m/s]
            plt.plot(time, qp[:,i], hold=False)

        plt.legend(
            (k[i],), 
            loc='best',
            fancybox=True,
            framealpha=0.5,
            fontsize=8
        )

        plt.tick_params(labelsize=6)
        plt.grid()

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('euler')
    for i in range(5):
        euler = quad.matrix_Q2euler(quat[:,(4*i):(4*(i+1))])
        for j in range(3):
            plt.subplot(5,3,1+j+(i*3))
            plt.tick_params(labelsize=6)
            plt.plot(time, 57.*euler[:,j], hold=False)
            plt.grid()
            plt.legend(
                ("body %d, %s" % ( (i+1), ('phi','theta','psi')[j] ),),
                loc='best',
                fancybox=True,
                framealpha=0.5,
                fontsize=8
            )
            plt.ylabel('[deg]', fontsize=6)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('vectors r_I')
    for i in range(5):
        for j in range(3):
            plt.subplot(5,3,1+j+(3*i))
            plt.tick_params(labelsize=6)
            plt.plot(time, r[:,(i*3)+j], hold=False)
            plt.grid()
            plt.legend(("body %d, %s" % ((i+1), chr(ord('x')+j)),),
                loc='best',
                fancybox=True,
                framealpha=0.5,
                fontsize=8
            )
            plt.ylabel('[m]', fontsize=6)

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
    pfig.canvas.set_window_title('forces')
    for i in range(5):
        for j in range(3):
            plt.subplot(5,3,1+j+(3*i))
            plt.tick_params(labelsize=6)
            plt.plot(time, F[:,(i*3)+j], hold=False)
            plt.grid()
            plt.legend(("body %d, %s" % ((i+1), chr(ord('x')+j)),),
                loc='best',
                fancybox=True,
                framealpha=0.5,
                fontsize=8
            )
            plt.ylabel('[N]', fontsize=6)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('moments')
    for i in range(5):
        for j in range(3):
            plt.subplot(5,3,1+j+(3*i))
            plt.tick_params(labelsize=6)
            plt.plot(time, M[:,(i*3)+j], hold=False)
            plt.grid()
            plt.legend(("body %d, %s" % ((i+1), chr(ord('x')+j)),),
                loc='best',
                fancybox=True,
                framealpha=0.5,
                fontsize=8
            )
            plt.ylabel('[Nm]', fontsize=6)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('Omega minus')
    for i in range(5):
        for j in range(3):
            plt.subplot(5,3,1+j+(i*3))
            plt.tick_params(labelsize=6)
            plt.plot(time, 57.*Omin[:,(i*3)+j], hold=False)
            plt.grid()
            plt.legend(("body %d, %s" % ((i+1), chr(ord('x')+j)),),
                loc='best',
                fancybox=True,
                framealpha=0.5,
                fontsize=8
            )
            plt.ylabel('[deg/s]', fontsize=6)

    #---- new figure:
    fig = fig + 1; pfig = plt.figure(fig); plt.clf();
    pfig.canvas.set_window_title('model 3D quadcopter')

    if False:
    #if True:
        showquad3d = sq.cshowquad3d(pfig, r, quat)
        #showquad3d.bExportFrames = True
        showquad3d.tr_maxbuflen = 50
        showquad3d.do_it()


    #------------------#
    plt.show(block=False);
    #------------------#

#====================================#
