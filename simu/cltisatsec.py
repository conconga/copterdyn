#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: These classes implement the dynamical behaviour of a second order LTI system
#               either SISO or MIMO and saturated in input and output rate.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy           as np
from   numpy           import dot
from   numpy           import inf

#====================================#
## \brief class cltisatsec          ##
## \author luciano kruk             ##
##                                  ##
## \parameters:                     ##
##     qsi : damping factor         ##
##     wn  : [rad/s] natural freq.  ##
##     x0  : initial state          ##
##     min_dxdt : min rate slope    ##
##     max_dxdt : max rate slope    ##
##     min_x    : min output        ##
##     max_x    : max output        ##
##     Ts       : discrete interval ##
##                                  ##
#====================================#

#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
#                                                                                 #
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#

class CLTISATSEC_SISO:
    # permanent:

    def __init__(self, qsi, wn, x0, min_dxdt, max_dxdt, min_x, max_x, Ts=0):
        """
            Scalar SISO second order LTI filter.

            For continuous simulation, use these methods:
                .dstate_dt() to calculate the derivative, and
                .c_update() to update the current state.

            For discrete simulation (Ts>0), use this method:
                .d_update() to update the current state for a given reference input value.

            The parameters 'qsi,wn,min_dxdt, max_dxdt, min_x, max_x' shall be scalars.
        """

        if (isinstance(x0, int) or isinstance(x0, float)):
            x0 = np.asarray([x0, 0])

        if (len(x0) == 1):
            self.x1  = x0[0] # output x
            self.x2  = 0.0   # dx/dt
        else:
            self.x1  = x0[0] # output x
            self.x2  = x0[1] # dx/dt

        self.x   = np.asarray(x0)

        self.a = wn**2.0
        self.b = 2.*qsi*wn
        self.u = 0.0

        self.min_dxdt = min_dxdt
        self.max_dxdt = max_dxdt
        self.min_x    = min_x
        self.max_x    = max_x

        # for discrete system:
        self.Ts = Ts
        if Ts>0:
            a        = self.a
            b        = self.b
            self.Ac  = np.asarray([[0,1],[-a,-b]])
            self.Bc  = np.asarray([0,a])

            # bilinear:
            aux      = np.linalg.inv(np.eye(2) - (0.5*Ts*self.Ac))
            self.Ad  = dot(np.eye(2) + (0.5*Ts*self.Ac), aux)
            self.Bd  = dot(aux, self.Bc) * Ts

            # forward:
            #self.Ad  = np.eye(2) + (Ts*self.Ac)
            #self.Bd  = Ts * self.Bc

            # backward:
            #self.Ad  = np.linalg.inv(np.eye(2) - (Ts*self.Ac))
            #self.Bd  = dot(np.linalg.inv(np.eye(2) - (Ts*self.Ac)), self.Bc) * Ts


        if (False):
            print("x1 = ",)
            print(self.x1)
            print("x2 = ",)
            print(self.x2)
            print("x  = ",)
            print(self.x)
            print("self.a = ",)
            print(self.a)
            print("self.b = ",)
            print(self.b)
            print("self.min_dxdt = ",)
            print(self.min_dxdt)
            print("self.max_dxdt = ",)
            print(self.max_dxdt)
            print("self.min_x = ",)
            print(self.min_x)
            print("self.max_x = ",)
            print(self.max_x)


    def __str__(self):
        return "state = %10.3e, derivative = %10.3e" % (self.x1, self.x2)


    def _saturate(self, _in, _min, _max):
        if (_in < _min):
            _out = _min
        elif (_in > _max):
            _out = _max
        else:
            _out = _in

        return _out


    def dstate_dt(self, x, t, u):
        """
            The state vector is

                x = [x1, x2]^T

            and the system is:

                dot(x1) = x2
                dot(x2) = (....)
        """

        # states:
        v1 = x[0]
        v2 = x[1]

        # saturation:
        f1 = self._saturate(u, self.min_x, self.max_x)
        f2 = v1 + (v2*self.b/self.a)

        vp1 = v2

        if ((v2 <= self.min_dxdt) and (f1 <= f2)) or ((v2 >= self.max_dxdt) and (f1 >= f2)):
            vp2 = 0
        else:
            vp2 = (-self.b*v2) + (self.a*(f1-v1))

        return np.asarray([vp1, vp2])

    def c_update(self, t, x):
        """
        Continuous time update.
        x: state at time t
        """

        self.x  = np.asarray(x)
        self.x1 = x[0]
        self.x2 = x[1]


    def d_update(self, t, u):
        """
        Discrete time update.
        u: input at time t
        """

        assert(self.Ts > 0)

        u_sat = self._saturate(u, self.min_x, self.max_x)

        w2k1   = dot(self.Ad, self.x) + (self.Bd * u_sat) # vector
        v2k    = self.x[1] # escalar v2[k]
        v2k1   = w2k1[1]   # escalar v2[k+1]

        w2k1_c = dot(self.Ac, self.x) + (self.Bc * u_sat)

        # lower constrained at [k+1]
        if (v2k1 < self.min_dxdt):

            # and trying to sink:
            if (v2k > v2k1):

                # forward:
                ti = (1./self.Ts) * (self.min_dxdt-v2k) / w2k1_c[1]

                if (0.<ti<1.): # descendo, com saturacao no caminho;
                    # forward:
                    Adti = np.eye(2) + (self.Ts*ti*self.Ac)
                    Bdti = self.Bc*self.Ts*ti

                    # from 0 until ti*Ts:
                    w2k1_ = dot(Adti, self.x) + (Bdti*u_sat)

                    # from ti*Ts until Ts:
                    v1k1 = w2k1_[0] + ((1.0 - ti) * self.Ts * w2k1_[1])

                    # update:
                    w2k1 = np.asarray([v1k1, w2k1_[1]])

                elif (ti<=0.): # comecou [k] em saturacao;
                    # integrate with d(v2k)/dt = 0:
                    v1k1 = self.x[0] + (self.Ts * self.min_dxdt)
                    w2k1 = np.asarray([v1k1, self.min_dxdt])

                else: # nao satura ateh [k+1]; impossible!
                    # w2k1 <- w2k1
                    pass

            else: # (v2k1 > v2k) # na regiao proibida de saturacao em [k] e em [k+1];
                # integrate with d(v2k)/dt = 0:
                v1k1 = self.x[0] + (self.Ts * self.min_dxdt)
                w2k1 = np.asarray([v1k1, self.min_dxdt])

        # upper constrained at [k+1]:
        elif (v2k1 > self.max_dxdt):

            # and trying to climb:
            if (v2k1 > v2k):

                # forward:
                ti = (1./self.Ts) * (self.max_dxdt-v2k) / w2k1_c[1]

                if (0.<ti<1.): # subindo com saturacao no caminho:
                    # forward:
                    Adti = np.eye(2) + (self.Ts*ti*self.Ac)
                    Bdti = self.Bc*self.Ts*ti

                    # from 0 until ti*Ts:
                    w2k1_ = dot(Adti, self.x) + (Bdti*u_sat)

                    # from ti*Ts until Ts:
                    v1k1 = w2k1_[0] + ((1.0 - ti) * self.Ts * w2k1_[1])

                    # update:
                    w2k1 = np.asarray([v1k1, w2k1_[1]])

                elif (ti<=0.): # comecou [k] em saturacao;
                    # integrate with d(v2k)/dt = 0:
                    v1k1 = self.x[0] + (self.Ts * v2k)
                    w2k1 = np.asarray([v1k1, v2k])

                else: # nao satura ateh [k+1]; impossible!
                    # w2k1 <- w2k1
                    pass

            else: # (v2k > v2k1) # na regiao proibida de saturacao em [k] e em [k+1];
                # integrate with d(v2k)/dt = 0:
                v1k1 = self.x[0] + (self.Ts * self.max_dxdt)
                w2k1 = np.asarray([v1k1, self.max_dxdt])

        else: # easy, no constraint:
            pass


        self.x = w2k1
        self.x1 = self.x[0]
        self.x2 = self.x[1]


    def get_state(self):
        return (self.x)



#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
#                                                                                 #
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#

class CLTISATSEC_MIMO:

    def __init__(self, qsi, wn, x0, min_dxdt, max_dxdt, min_x, max_x, Ts=0):
        """
            MIMO second order LTI filter.

            For continuous simulation, use these methods:
                .dstate_dt() to calculate the derivative, and
                .c_update() to update the current state.

            For discrete simulation (Ts>0), use this method:
                .d_update() to update the current state for a given reference input value.

            The parameters 'qsi,wn,min_dxdt, max_dxdt, min_x, max_x' shall be
            either scalars, vectors or lists. The parameter 'Ts' shall be a scalar, if any.

            The dimension of 'x0' defines the number of internal SISO systems.

            The MIMO state vector represents each two values one singlo SISO state vector:

            state =   [   x1    ]
                      [ dot{x1} ]
                      [   x2    ]
                      [ dot{x2} ]
                      [   ...   ]
                      [   xn    ]
                      [ dot{xn} ]

            The methods interleave() and deinterleave() change the representation of state.
        """

        self.n        = len(x0)
        self.x0       = x0
        self.Ts       = Ts
        self.qsi      = self._fn_fill_config_list(qsi)
        self.wn       = self._fn_fill_config_list(wn)
        self.min_x    = self._fn_fill_config_list(min_x)
        self.max_x    = self._fn_fill_config_list(max_x)
        self.min_dxdt = self._fn_fill_config_list(min_dxdt)
        self.max_dxdt = self._fn_fill_config_list(max_dxdt)

        # instances of CLTISATSEC_SISO:
        self.siso     = [CLTISATSEC_SISO(
                            self.qsi[i],
                            self.wn[i],
                            self.x0[i],
                            self.min_dxdt[i],
                            self.max_dxdt[i],
                            self.min_x[i],
                            self.max_x[i],
                            Ts=Ts)
                            for i in range(self.n)
                        ]


    def _fn_fill_config_list(self, In):
        """
            Converts scalar, array or list into list.
        """

        if (isinstance(In, int) or isinstance(In, float)):
            Out = [In for i in range(self.n)]

        elif (isinstance(In, np.ndarray)):
            Out = In.tolist()

        elif (isinstance(In, list)):
            Out = In

        else:
            print("hummm... are you sure you know what you are doing? I'm not!")
            assert(False)

        return Out



    def dstate_dt(self, x, t, u):
        """
            Returns a vector with all derivatives for continuous integration.
        """

        # state:
        dxdt = []

        # protecting 'u':
        u = self._fn_fill_config_list(u)

        # all derivatives:
        for i in range(self.n):
            xi = [x[2*i], x[(2*i)+1]]
            dxdt += self.siso[i].dstate_dt(xi, t, u[i]).tolist()

        return dxdt


    def c_update(self, t, x):
        """
        Continuous time update.
        x: state at time t
        """

        for i in range(self.n):
            xi = [x[2*i], x[(2*i)+1]]
            self.siso[i].c_update(t, xi)


    def d_update(self, t, u):
        """
        Discrete time update.
        u: input at time t
        """

        assert(self.Ts > 0)

        # protecting 'u':
        u = self._fn_fill_config_list(u)

        for i in range(self.n):
            self.siso[i].d_update(t, u[i])

    def get_state(self,*args):
        """
            Returns state of MIMO object or individual SISO instances.

            .get_state() returns state from MIMO object.
            .get_state(i) returns state from SISO instance 'i'.
        """

        if (len(args) == 0):
            x = []
            for i in range(self.n):
                x += self.siso[i].get_state().tolist()
        elif (len(args) == 1):
            x = self.siso[args[0]].get_state()

        return np.asarray(x) # list or np.array

    def interleave(self, x):
        """
            This method changes
                x = [x1, x2, ... , x1p, x2p, ...]
            to
                x = [x1, x1p, x2, x2p, ... ]
        """

        if (isinstance(x, np.ndarray)):
            In = x.squeeze().tolist()

        # interleave:
        y = []
        for i in range(self.n):
            y += [In[i], In[self.n+i]]

        if (isinstance(x, np.ndarray)):
            y = np.asarray(y)

        return y

    def deinterleave(self, x):
        """
            This method changes
                x = [x1, x1p, x2, x2p, ... ]
            to
                x = [x1, x2, ... , x1p, x2p, ...]
        """

        if (isinstance(x, np.ndarray)):
            In = x.squeeze().tolist()

        # deinterleave:
        y = [ In[2*i] for i in range(self.n) ] + [ In[(2*i)+1] for i in range(self.n) ]

        if (isinstance(x, np.ndarray)):
            y = np.asarray(y)

        return y


#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):

    from   scipy.integrate    import odeint
    import matplotlib.pyplot  as plt

    qsi = 0.5
    wn  = 2*3.14*10
    Fs  = 200 # [Hz]
    Ts  = 1./Fs
    T   = np.arange(0,2,Ts)
    U   = (T > 0.5) * 1.0

    lti_free     = CLTISATSEC_SISO(qsi, wn, 0, -inf, inf, -inf, inf)
    lti_ratelim  = CLTISATSEC_SISO(qsi, wn, 0, -1.5, 1.5, -inf, inf)
    lti_statelim = CLTISATSEC_SISO(qsi, wn, 0, -inf, inf, -0.5, 0.5)
    lti_2lim     = CLTISATSEC_SISO(qsi, wn, 0, -1.7, 1.7, -0.5, 0.5)

    lti_free_buf     = list()
    lti_ratelim_buf  = list()
    lti_statelim_buf = list()
    lti_2lim_buf     = list()

    for (idx, txt, lti, lti_buf) in zip(
            range(4),
            ['lti_free',    'lti_ratelim',    'lti_statelim',    'lti_2lim',    ],
            [ lti_free,      lti_ratelim,      lti_statelim,      lti_2lim,     ],
            [ lti_free_buf,  lti_ratelim_buf,  lti_statelim_buf,  lti_2lim_buf, ]
        ):

        print("--------  %s..".format(txt))
        t0 = 0

        for t,u in zip(T,U):

            # integration:
            # continuous:
            _,y = odeint(lti.dstate_dt, lti.get_state(), [t0, t], (u,))

            # pos:
            t0 = t
            lti.c_update(t, y)

            # buffer:
            lti_buf.append(lti.get_state())

            print(lti)


    #UmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUm#
    # new test with real sample signal:
    # [[time,value]...]
    Fs     = 100. # do not change; this is the sample rate of the next test signal:
    signal = [0.0100,-6.3808,0.0200,-7.5034,0.0300,-8.5765,0.0400,-9.5647,0.0500,-10.4716,0.0600,-11.3007,0.0700,-12.0555,0.0800,-12.7395,0.0900,-13.3563,0.1000,-13.9092,0.1100,-14.4017,0.1200,-14.8371,0.1300,-15.2186,0.1400,-15.5495,0.1500,-15.8327,0.1600,-16.0715,0.1700,-16.2686,0.1800,-16.4269,0.1900,-16.5492,0.2000,-16.6380,0.2100,-16.6960,0.2200,-16.7256,0.2300,-16.7291,0.2400,-16.7088,0.2500,-16.6667,0.2600,-16.6049,0.2700,-16.5255,0.2800,-16.4300,0.2900,-16.3205,0.3000,-16.1983,0.3100,-16.0652,0.3200,-15.9225,0.3300,-15.7716,0.3400,-15.6139,0.3500,-15.4504,0.3600,-15.2822,0.3700,-15.1105,0.3800,-14.9362,0.3900,-14.7601,0.4000,-14.5831,0.4100,-14.4058,0.4200,-14.2290,0.4300,-14.0533,0.4400,-13.8791,0.4500,-13.7071,0.4600,-13.5377,0.4700,-13.3712,0.4800,-13.2080,0.4900,-13.0484,0.5000,-12.8926,0.5100,-12.7409,0.5200,-12.5934,0.5300,-12.4503,0.5400,-12.3117,0.5500,-12.1777,0.5600,-12.0483,0.5700,-11.9235,0.5800,-11.8034,0.5900,-11.6880,0.6000,-11.5771,0.6100,-11.4708,0.6200,-11.3690,0.6300,-11.2716,0.6400,-11.1784,0.6500,-11.0895,0.6600,-11.0046,0.6700,-10.9236,0.6800,-10.8464,0.6900,-10.7729,0.7000,-10.7028,0.7100,-10.6361,0.7200,-10.5726,0.7300,-10.5121,0.7400,-10.4544,0.7500,-10.3995,0.7600,-10.3470,0.7700,-10.2970,0.7800,-10.2492,0.7900,-10.2034,0.8000,-10.1595,0.8100,-10.1174,0.8200,-10.0768,0.8300,-10.0377,0.8400,-9.9998,0.8500,-9.9631,0.8600,-9.9274,0.8700,-9.8926,0.8800,-9.8585,0.8900,-9.8250,0.9000,-9.7920,0.9100,-9.7594,0.9200,-9.7271,0.9300,-9.6949,0.9400,-9.6628,0.9500,-9.6306,0.9600,-9.5982,0.9700,-9.5662,0.9800,-9.5351,0.9900,-9.5050,1.0000,-9.4757,1.0100,-9.4472,1.0200,-9.4193,1.0300,-9.3920,1.0400,-9.3653,1.0500,-9.3391,1.0600,-9.3132,1.0700,-9.2878,1.0800,-9.2627,1.0900,-9.2379,1.1000,-9.2133,1.1100,-9.1890,1.1200,-9.1648,1.1300,-9.1407,1.1400,-9.1168,1.1500,-9.0930,1.1600,-9.0692,1.1700,-9.0456,1.1800,-9.0219,1.1900,-8.9983,1.2000,-8.9747,1.2100,-8.9511,1.2200,-8.9274,1.2300,-8.9038,1.2400,-8.8802,1.2500,-8.8565,1.2600,-8.8328,1.2700,-8.8091,1.2800,-8.7854,1.2900,-8.7617,1.3000,-8.7379,1.3100,-8.7141,1.3200,-8.6903,1.3300,-8.6665,1.3400,-8.6427,1.3500,-8.6189,1.3600,-8.5950,1.3700,-8.5712,1.3800,-8.5475,1.3900,-8.5237,1.4000,-8.5000,1.4100,-8.4763,1.4200,-8.4526,1.4300,-8.4290,1.4400,-8.4055,1.4500,-8.3820,1.4600,-8.3586,1.4700,-8.3353,1.4800,-8.3121,1.4900,-8.2889,1.5000,-8.2659,1.5100,-8.2430,1.5200,-8.2201,1.5300,-8.1974,1.5400,-8.1748,1.5500,-8.1524,1.5600,-8.1300,1.5700,-8.1078,1.5800,-8.0858,1.5900,-8.0639,1.6000,-8.0421,1.6100,-8.0205,1.6200,-7.9990,1.6300,-7.9778,1.6400,-7.9566,1.6500,-7.9357,1.6600,-7.9149,1.6700,-7.8943,1.6800,-7.8738,1.6900,-7.8536,1.7000,-7.8335,1.7100,-7.8136,1.7200,-7.7938,1.7300,-7.7743,1.7400,-7.7549,1.7500,-7.7358,1.7600,-7.7168,1.7700,-7.6979,1.7800,-7.6793,1.7900,-7.6609,1.8000,-7.6426,1.8100,-7.6245,1.8200,-7.6066,1.8300,-7.5889,1.8400,-7.5714,1.8500,-7.5540,1.8600,-7.5369,1.8700,-7.5199,1.8800,-7.5031,1.8900,-7.4864,1.9000,-7.4700,1.9100,-7.4537,1.9200,-7.4376,1.9300,-7.4216,1.9400,-7.4059,1.9500,-7.3903,1.9600,-7.3749,1.9700,-7.3596,1.9800,-7.3445,1.9900,-7.3295,2.0000,-7.3148,2.0100,-7.3002,2.0200,-7.2857,2.0300,-7.2714,2.0400,-7.2572,2.0500,-7.2432,2.0600,-7.2294,2.0700,-7.2157,2.0800,-7.2021,2.0900,-7.1887,2.1000,-7.1755,2.1100,-7.1623,2.1200,-7.1493,2.1300,-7.1365,2.1400,-7.1238,2.1500,-7.1112,2.1600,-7.0988,2.1700,-7.0864,2.1800,-7.0743,2.1900,-7.0622,2.2000,-7.0503,2.2100,-7.0385,2.2200,-7.0268,2.2300,-7.0152,2.2400,-7.0038,2.2500,-6.9925,2.2600,-6.9813,2.2700,-6.9702,2.2800,-6.9592,2.2900,-6.9483,2.3000,-6.9376,2.3100,-6.9269,2.3200,-6.9164,2.3300,-6.9060,2.3400,-6.8957,2.3500,-6.8855,2.3600,-6.8753,2.3700,-6.8653,2.3800,-6.8554,2.3900,-6.8456,2.4000,-6.8359,2.4100,-6.8263,2.4200,-6.8168,2.4300,-6.8074,2.4400,-6.7981,2.4500,-6.7888,2.4600,-6.7797,2.4700,-6.7706,2.4800,-6.7617,2.4900,-6.7528,2.5000,-6.7440,2.5100,-6.7353,2.5200,-6.7267,2.5300,-6.7182,2.5400,-6.7098,2.5500,-6.7014,2.5600,-6.6932,2.5700,-6.6850,2.5800,-6.6769,2.5900,-6.6689,2.6000,-6.6609,2.6100,-6.6530,2.6200,-6.6453,2.6300,-6.6375,2.6400,-6.6299,2.6500,-6.6223,2.6600,-6.6149,2.6700,-6.6075,2.6800,-6.6001,2.6900,-6.5929,2.7000,-6.5857,2.7100,-6.5785,2.7200,-6.5715,2.7300,-6.5645,2.7400,-6.5576,2.7500,-6.5507,2.7600,-6.5440,2.7700,-6.5373,2.7800,-6.5306,2.7900,-6.5240,2.8000,-6.5175,2.8100,-6.5111,2.8200,-6.5047,2.8300,-6.4984,2.8400,-6.4921,2.8500,-6.4859,2.8600,-6.4798,2.8700,-6.4737,2.8800,-6.4677,2.8900,-6.4617,2.9000,-6.4558,2.9100,-6.4500,2.9200,-6.4442,2.9300,-6.4385,2.9400,-6.4328,2.9500,-6.4272,2.9600,-6.4217,2.9700,-6.4162,2.9800,-6.4107,2.9900,-6.4053,3.0000,-6.4000,3.0100,-6.3947,3.0200,-6.3895,3.0300,-6.3843,3.0400,-6.3792,3.0500,-6.3741,3.0600,-6.3691,3.0700,-6.3641,3.0800,-6.3592,3.0900,-6.3543,3.1000,-6.3495,3.1100,-6.3447,3.1200,-6.3400,3.1300,-6.3353,3.1400,-6.3306,3.1500,-6.3260,3.1600,-6.3215,3.1700,-6.3170,3.1800,-6.3125,3.1900,-6.3081,3.2000,-6.3037,3.2100,-6.2994,3.2200,-6.2951,3.2300,-6.2909,3.2400,-6.2867,3.2500,-6.2825,3.2600,-6.2784,3.2700,-6.2744,3.2800,-6.2703,3.2900,-6.2663,3.3000,-6.2624,3.3100,-6.2585,3.3200,-6.2546,3.3300,-6.2507,3.3400,-6.2470,3.3500,-6.2432,3.3600,-6.2395,3.3700,-6.2358,3.3800,-6.2321,3.3900,-6.2285,3.4000,-6.2249,3.4100,-6.2214,3.4200,-6.2179,3.4300,-6.2144,3.4400,-6.2110,3.4500,-6.2076,3.4600,-6.2042,3.4700,-6.2009,3.4800,-6.1976,3.4900,-6.1943,3.5000,-6.1911,3.5100,-6.1879,3.5200,-6.1847,3.5300,-6.1816,3.5400,-6.1785,3.5500,-6.1754,3.5600,-6.1723,3.5700,-6.1693,3.5800,-6.1663,3.5900,-6.1634,3.6000,-6.1605,3.6100,-6.1576,3.6200,-6.1547,3.6300,-6.1518,3.6400,-6.1490,3.6500,-6.1462,3.6600,-6.1435,3.6700,-6.1408,3.6800,-6.1381,3.6900,-6.1354,3.7000,-6.1327,3.7100,-6.1301,3.7200,-6.1275,3.7300,-6.1249,3.7400,-6.1213,3.7500,-6.1082,3.7600,-6.0847,3.7700,-6.0517,3.7800,-6.0103,3.7900,-5.9611,3.8000,-5.9051,3.8100,-5.8431,3.8200,-5.7757,3.8300,-5.7037,3.8400,-5.6278,3.8500,-5.5485,3.8600,-5.4665,3.8700,-5.3823,3.8800,-5.2964,3.8900,-5.2094,3.9000,-5.1216,3.9100,-5.0335,3.9200,-4.9455,3.9300,-4.8580,3.9400,-4.7712,3.9500,-4.6855,3.9600,-4.6012,3.9700,-4.5184,3.9800,-4.4376,3.9900,-4.3587,4.0000,-4.2821,4.0100,-4.2079,4.0200,-4.1363,4.0300,-4.0673,4.0400,-4.0011,4.0500,-3.9377,4.0600,-3.8773,4.0700,-3.8199,4.0800,-3.7655,4.0900,-3.7142,4.1000,-3.6660,4.1100,-3.6208,4.1200,-3.5788,4.1300,-3.5399,4.1400,-3.5040,4.1500,-3.4712,4.1600,-3.4413,4.1700,-3.4145,4.1800,-3.3905,4.1900,-3.3693,4.2000,-3.3510,4.2100,-3.3353,4.2200,-3.3223,4.2300,-3.3119,4.2400,-3.3040,4.2500,-3.2985,4.2600,-3.2953,4.2700,-3.2944,4.2800,-3.2956,4.2900,-3.2989,4.3000,-3.3042,4.3100,-3.3114,4.3200,-3.3204,4.3300,-3.3312,4.3400,-3.3436,4.3500,-3.3575,4.3600,-3.3729,4.3700,-3.3897,4.3800,-3.4078,4.3900,-3.4271,4.4000,-3.4475,4.4100,-3.4690,4.4200,-3.4914,4.4300,-3.5148,4.4400,-3.5390,4.4500,-3.5640,4.4600,-3.5897,4.4700,-3.6159,4.4800,-3.6428,4.4900,-3.6701,4.5000,-3.6979,4.5100,-3.7260,4.5200,-3.7545,4.5300,-3.7833,4.5400,-3.8122,4.5500,-3.8414,4.5600,-3.8706,4.5700,-3.9000,4.5800,-3.9294,4.5900,-3.9587,4.6000,-3.9881,4.6100,-4.0174,4.6200,-4.0465,4.6300,-4.0756,4.6400,-4.1044,4.6500,-4.1331,4.6600,-4.1615,4.6700,-4.1897,4.6800,-4.2177,4.6900,-4.2454,4.7000,-4.2727,4.7100,-4.2998,4.7200,-4.3265,4.7300,-4.3528,4.7400,-4.3788,4.7500,-4.4045,4.7600,-4.4297,4.7700,-4.4546,4.7800,-4.4791,4.7900,-4.5032,4.8000,-4.5268,4.8100,-4.5501,4.8200,-4.5729,4.8300,-4.5954,4.8400,-4.6174,4.8500,-4.6390,4.8600,-4.6602,4.8700,-4.6810,4.8800,-4.7014,4.8900,-4.7213,4.9000,-4.7409,4.9100,-4.7600,4.9200,-4.7788,4.9300,-4.7972,4.9400,-4.8151,4.9500,-4.8327,4.9600,-4.8499,4.9700,-4.8668,4.9800,-4.8833,4.9900,-4.8994]
    c      = CLTISATSEC_SISO(0.7, 2.*np.pi*20, -16, -11, 5, -15, 0)
    d      = CLTISATSEC_SISO(0.7, 2.*np.pi*10, -16, -11, 5, -15, 0, Ts=1./Fs)
    t0     = 0
    log_t2 = []
    for idx in range(len(signal)/2):
        t   = signal[idx*2]
        u   = signal[(idx*2)+1]

        # continuous:
        _,y = odeint(c.dstate_dt, c.get_state(), [t0, t], (u,))
        c.c_update(t,y)

        # discrete:
        d.d_update(t,u)

        # log:
        log_t2.append([t,u] + c.get_state().tolist() + d.get_state().tolist())
        #log_t2.append([t,u] + c.get_state().tolist())

        t0  = t
        print("continuous = ",)
        print(c)
        print("discrete   = ",)
        print(d)

    log_t2 = np.asarray(log_t2)


    #UmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUmUm#
    # test of MIMO systems:
    mimo_c = CLTISATSEC_MIMO(0.7, 2.*np.pi*20, np.asarray([0,1]), -5, 5, -2, [0.8,1.1])
    mimo_d = CLTISATSEC_MIMO(0.7, 2.*np.pi*20, np.asarray([0,1]), -5, 5, -2, [0.8,1.1], Ts=Ts)
    mimo_c_buf = [];
    mimo_d_buf = [];

    t0 = 0
    for t,u in zip(T,U):

        # integration:
        # continuous:
        _,y = odeint(mimo_c.dstate_dt, mimo_c.get_state(), [t0, t], (u,))

        # pos:
        t0 = t
        mimo_c.c_update(t, y)

        # discrete:
        mimo_d.d_update(t,u)

        # buffer:
        mimo_c_buf.append([t,u] + mimo_c.get_state().tolist())
        mimo_d_buf.append([t,u] + mimo_d.get_state().tolist())

    mimo_c_buf = np.asarray(mimo_c_buf)
    mimo_d_buf = np.asarray(mimo_d_buf)


    #  - #  - #  - #  - #  - #  - #  - #  - # - #
    # figures:
    #  - #  - #  - #  - #  - #  - #  - #  - # - #

    plt.figure(1), plt.clf()
    plt.subplot(2,1,1)
    plt.plot(
            T, np.asarray(lti_free_buf)[:,0],
            T, np.asarray(lti_ratelim_buf)[:,0],
            T, np.asarray(lti_statelim_buf)[:,0],
            T, np.asarray(lti_2lim_buf)[:,0],
    )
    plt.grid()
    plt.legend(("lti_free_buf", "lti_ratelim_buf", "lti_statelim_buf", "lti_2lim_buf"))

    plt.subplot(2,1,2)
    plt.plot(
            T, np.asarray(lti_free_buf)[:,1],
            T, np.asarray(lti_ratelim_buf)[:,1],
            T, np.asarray(lti_statelim_buf)[:,1],
            T, np.asarray(lti_2lim_buf)[:,1],
    )
    plt.grid()
    plt.ylabel("rates")

    #----- new figure -----#
    plt.figure(2), plt.clf()
    plt.plot(log_t2[:,0], log_t2[:,1:])
    plt.grid(True)
    plt.legend(('signal', 'c', 'dot{c}', 'd', 'dot{d}'))

    #----- new figure -----#
    plt.figure(3), plt.clf()
    plt.plot(mimo_c_buf[:,0], mimo_c_buf[:,1:])
    plt.grid(True)
    plt.legend(('signal', 'c1', 'dot{c1}', 'c2', 'dot{c2}'))

    #----- new figure -----#
    plt.figure(4), plt.clf()
    plt.plot(mimo_d_buf[:,0], mimo_d_buf[:,1:])
    plt.grid(True)
    plt.legend(('signal', 'd1', 'dot{d1}', 'd2', 'dot{d2}'))

    plt.show(block=False)

#====================================#
