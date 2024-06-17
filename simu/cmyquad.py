#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This class appends the attributes of a unique quadcopter,
#               and simulates it as well. The attributes are related to
#               physical description (length, mass, angles, etc..) and
#               calculation of acting forces and moments.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy               as np;
from   cquadcopter     import CQUADCOPTER
from   navfunc         import CNAVFUNC
from   cmuxdemux       import CMUXDEMUX
from   cprop           import CPROP
from   numpy           import dot

#====================================#
## \brief class cmyquad ##
## \author luciano kruk     ##
#
## \description:
#====================================#

class CMYQUAD(CNAVFUNC):
    # permanent:
    zeros = np.zeros((3,1))

    def __init__(self):

        # some physical characteristics:
        mass_body = 0.40                              # [kg]
        mass_2    = 0.05                              # [kg]
        mass_3    = 0.05                              # [kg]
        mass_4    = 0.05                              # [kg]
        mass_5    = 0.05                              # [kg]
        J_body    = np.diag([0.0086, 0.0068, 0.0172]) # [kg.m2]
        J_2       = 1e-3 * np.eye(3)                  # [kg.m2]
        J_3       = 1e-3 * np.eye(3)                  # [kg.m2]
        J_4       = 1e-3 * np.eye(3)                  # [kg.m2]
        J_5       = 1e-3 * np.eye(3)                  # [kg.m2]
        l2        = 0.2                               # [m]
        l3        = 0.2                               # [m]
        l4        = 0.2                               # [m]
        l5        = 0.2                               # [m]

        # angle between body frame and copter arms:
        delta_2_rad = np.radians(0  +45)
        delta_3_rad = np.radians(90 +45)
        delta_4_rad = np.radians(180+45)
        delta_5_rad = np.radians(270+45)

        if False:
        #if True:
            delta_2_rad = np.radians(0)
            delta_3_rad = np.radians(40)
            delta_4_rad = np.radians(-40)
            delta_5_rad = np.radians(145)

        # misalignment between body's arm and rotor axis:
        alfa_2_rad = np.radians(0)
        alfa_3_rad = np.radians(0)
        alfa_4_rad = np.radians(0)
        alfa_5_rad = np.radians(0)

        #alfa_2_rad = np.radians(+90)
        #alfa_3_rad = np.radians(+90)
        #alfa_4_rad = np.radians(3)
        #alfa_5_rad = np.radians(-3)

        #if True:
        if False:
            alfa_2_rad = np.radians(20)
            alfa_3_rad = np.radians(0)
            alfa_4_rad = np.radians(0)
            alfa_5_rad = np.radians(0)

        #if True:
        if False:
            alfa_2_rad = np.radians(90)
            alfa_3_rad = np.radians(90)
            alfa_4_rad = np.radians(90)
            alfa_5_rad = np.radians(90)

        # description of joint connections:
        Tm = np.asarray([ # T^-
            [0,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0]
        ])

        TM = np.asarray([ # T^+ = T
            [1,0,0,0,0],
            [1,1,0,0,0],
            [1,0,1,0,0],
            [1,0,0,1,0],
            [1,0,0,0,1]
        ])


        # transformation matrices:
        # ( R_from_rotor(i)_to_bodyframe  =  R_from_arm(i)_to_bodyframe . R_from_rotor(i)_to_arm(i) )

        q221 = self.q1_prod_q2(
                self.euler2Q((0,0,-delta_2_rad)), # from arm(i) to body frame
                self.euler2Q((-alfa_2_rad,0,0)))  # from rotor(i) to arm(i)
        self.R221 = self.Q2C(q221)

        q321 = self.q1_prod_q2(
                self.euler2Q((0,0,-delta_3_rad)), # from arm(i) to body frame
                self.euler2Q((-alfa_3_rad,0,0)))  # from rotor(i) to arm(i)
        self.R321 = self.Q2C(q321)

        q421 = self.q1_prod_q2(
                self.euler2Q((0,0,-delta_4_rad)), # from arm(i) to body frame
                self.euler2Q((-alfa_4_rad,0,0)))  # from rotor(i) to arm(i)
        self.R421 = self.Q2C(q421)

        q521 = self.q1_prod_q2(
                self.euler2Q((0,0,-delta_5_rad)), # from arm(i) to body frame
                self.euler2Q((-alfa_5_rad,0,0)))  # from rotor(i) to arm(i)
        self.R521 = self.Q2C(q521)

        # description of joint constraints:
        # \omega^- = P.\dot(q)
        P = np.hstack((np.zeros((3,3)), np.eye(3), np.zeros((3,4))))

        P = np.vstack((P, np.hstack((np.zeros((3,6)), self.R221[:,2].reshape((3,1)), np.zeros((3,3)))) ))
        P = np.vstack((P, np.hstack((np.zeros((3,7)), self.R321[:,2].reshape((3,1)), np.zeros((3,2)) )) ))
        P = np.vstack((P, np.hstack((np.zeros((3,8)), self.R421[:,2].reshape((3,1)), np.zeros((3,1)) )) ))
        P = np.vstack((P, np.hstack((np.zeros((3,9)), self.R521[:,2].reshape((3,1)) )) ))

        # description of joint constraints:
        # \dot(c)^+ = k.\dot(q)
        k           = np.zeros((15,10))
        k[0:3, 0:3] = -np.eye(3)

        # parameters:
        self.PARAM = {
            'Tm'         : Tm,
            'TM'         : TM,
            'mass_body'  : mass_body,
            'mass_2'     : mass_2,
            'mass_3'     : mass_3,
            'mass_4'     : mass_4,
            'mass_5'     : mass_5,
            'J_body'     : J_body,
            'J_2'        : J_2,
            'J_3'        : J_3,
            'J_4'        : J_4,
            'J_5'        : J_5,
            'l2'         : l2,
            'l3'         : l3,
            'l4'         : l4,
            'l5'         : l5,
            'P'          : P,
            'k'          : k,
            'force_fn'   : (self.force_mainbody, self.force_prop_2, self.force_prop_3, self.force_prop_4, self.force_prop_5),
            'momt_fn'    : (self.momt_mainbody, self.momt_prop_2, self.momt_prop_3, self.momt_prop_4, self.momt_prop_5),
            'q221'       : q221,
            'q321'       : q321,
            'q421'       : q421,
            'q521'       : q521,
            'delta_2_rad': delta_2_rad,
            'delta_3_rad': delta_3_rad,
            'delta_4_rad': delta_4_rad,
            'delta_5_rad': delta_5_rad
        }

        # multibody dynamics of the quadcopter:
        self.dynquad = CQUADCOPTER(self.PARAM)

        # a mux/demux object:
        # (dynquad, cprop(4x))
        self.md      = CMUXDEMUX([len(self.dynquad.state), 2, 2, 2, 2])

        # some additional components:
        self.prop2   = CPROP(isCounterClockWise=False)
        self.prop3   = CPROP(isCounterClockWise=True)
        self.prop4   = CPROP(isCounterClockWise=False)
        self.prop5   = CPROP(isCounterClockWise=True)

        # system state vector:
        self.state = self.sys_mux(
            self.dynquad.state,
            self.prop2.state,
            self.prop3.state,
            self.prop4.state,
            self.prop5.state
        )
        self.t     = 0
        self.u     = np.zeros(4)

    ##WWww=--  mux states from all subsystems into one vector --=wwWW##
    def sys_mux(self, *states):
        return self.md.mux(*states)


    ##WWww=--  demux global state into individual state vectors  --=wwWW##
    def sys_demux(self, state):
        return self.md.demux(state)


    ##WWww=--  system derivative vector  --=wwWW##
    def dsys_dt(self, state, t):
        """
        Calculates the derivative of all continuous equations in the system
        """

        s1,s2,s3,s4,s5 = self.sys_demux(state)

        d_dt = self.sys_mux(
            self.dynquad.dstate_dt(s1, t),
            self.prop2.dstate_dt(s2, t),
            self.prop3.dstate_dt(s3, t),
            self.prop4.dstate_dt(s4, t),
            self.prop5.dstate_dt(s5, t)
        )

        return d_dt


    ##WWww=--  actions taken before time update  --=wwWW##
    def pre_update(self, t):
        assert(t >= self.t)

        # TODO: object control should calculate 'u'
        if t<1.:
            u = 0.1 * np.ones(4)
        elif t<1.5:
            u = 0.2 * np.ones(4)
        else:
            u = 0 * np.ones(4)

        if True:
            if (t<10e-3):
                u = self.u
            elif (np.remainder(t,3.0) < 1e-4):
                u = 0.04*np.random.rand(4)
                #u = self.u + (0.10 * np.random.rand(4))
            else:
                u = self.u

        #u = 0.05 * np.asarray([0,0,1,1]) # roll
        #u = 0.05 * np.asarray([1,0,0,1]) # pitch
        #u = 0.05 * np.asarray([0,1,0,1]) # yaw

        #u = 0.05 * np.ones(4)

        # get generalized coordinate of propeller rotation:
        s1,_,_,_,_  = self.sys_demux(self.state)
        _,qp,_ = self.dynquad.state_demux(s1)
        w2    = qp[6]
        w3    = qp[7]
        w4    = qp[8]
        w5    = qp[9]

        self.dynquad.pre_update(t)
        #print "w = %20.15e; %20.15e; %20.15e; %20.15e" % (w2, w3, w4, w5)

        self.u = u # backup
        self.prop2.pre_update(t, u[0], w2)
        self.prop3.pre_update(t, u[1], w3)
        self.prop4.pre_update(t, u[2], w4)
        self.prop5.pre_update(t, u[3], w5)


    ##WWww=--  actions taken after time update  --=wwWW##
    def pos_update(self, t, state):
        self.state = state
        self.t     = t

        s1,s2,s3,s4,s5 = self.sys_demux(state)

        self.dynquad.pos_update(t,s1)
        self.prop2.pos_update(t,s2)
        self.prop3.pos_update(t,s3)
        self.prop4.pos_update(t,s4)
        self.prop5.pos_update(t,s5)

        #######################
        ##  identification:  ##
        #######################
        #
        #   $ python simuquad.py | grep "^si," | sed 's/^.\{4\}//' > log
        #   $ ipython
        #   >>>  run sident.py
        #
        #######################
        if False:

            RI21    = self.dynquad.get_RI21()
            T2,tau2 = self.prop2.get_FT()
            T3,tau3 = self.prop3.get_FT()
            T4,tau4 = self.prop4.get_FT()
            T5,tau5 = self.prop5.get_FT()

            print("[T2...T5]     = [{:10f} {:10f} {:10f} {:10f}]".format(T2, T3, T4, T5))
            print("[tau2...tau5] = [{:10f} {:10f} {:10f} {:10f}]".format(tau2, tau3, tau4, tau5))

            # forces (T) and moments (tau) described in body 1:
            # (the minus signal is due to reaction on the body 1)
            T2   = dot(self.R221, np.asarray([[0],[0],[-T2]]))
            tau2 = dot(self.R221, np.asarray([[0],[0],[-tau2]]))
            T3   = dot(self.R321, np.asarray([[0],[0],[-T3]]))
            tau3 = dot(self.R321, np.asarray([[0],[0],[-tau3]]))
            T4   = dot(self.R421, np.asarray([[0],[0],[-T4]]))
            tau4 = dot(self.R421, np.asarray([[0],[0],[-tau4]]))
            T5   = dot(self.R521, np.asarray([[0],[0],[-T5]]))
            tau5 = dot(self.R521, np.asarray([[0],[0],[-tau5]]))

            #RI21 = np.eye(3)
            # 0..7 [t g_xyz u_1:4]
            i  = [t] + list(self.gravity(RI21).squeeze()) + list(self.u)
            # 8..10 [pqr]
            i += list(dot(RI21, self.dynquad.get_omega_01_0()).squeeze())
            # 11..13 [uvw]
            i += list(dot(RI21, self.dynquad.calc_rp(t)[0:3,0]).squeeze())
            # 14..16 [d(uvw)/dt]
            i += list(dot(RI21, self.dynquad.calc_rpp(t)[0:3,0]).squeeze())
            # 17..19 [dot(pqr)]
            i += list(dot(RI21, self.dynquad.get_omegap_01_0(t)).squeeze())
            # 20 [uT]
            i.append(float(np.sum([j[2] for j in [T2, T3, T4, T5]])))

            # 21 [u_phi]
            idx_quadrant = np.asarray([int(j/(3.14159/2))+1 for j in [
                self.PARAM['delta_2_rad'],
                self.PARAM['delta_3_rad'],
                self.PARAM['delta_4_rad'],
                self.PARAM['delta_5_rad']
            ]])

            i.append(float(
                np.sum(np.asarray([(((j in [3,4])*2)-1) for j in idx_quadrant]) * np.concatenate([j[0] for j in [tau2, tau3, tau4, tau5]])) +
                np.sum([j[2]*k*np.sin(l) for (j,k,l) in zip(
                    [T2,                        T3,                        T4,                        T5                       ],
                    [self.PARAM['l2'],          self.PARAM['l3'],          self.PARAM['l4'],          self.PARAM['l5']         ],
                    [self.PARAM['delta_2_rad'], self.PARAM['delta_3_rad'], self.PARAM['delta_4_rad'], self.PARAM['delta_5_rad']]
                )])
            ))

            print("------------------")
            print("u_phi = ",)
            print(i[-1])
            print("   1) ",)
            print(np.sum(np.asarray([(((j in [3,4])*2)-1) for j in idx_quadrant]) * np.concatenate([j[0] for j in [tau2, tau3, tau4, tau5]])))
            print("       tau2[0]) ",)
            print(tau2[0])
            print("       tau3[0]) ",)
            print(tau3[0])
            print("       tau4[0]) ",)
            print(tau4[0])
            print("       tau5[0]) ",)
            print(tau5[0])
            print("   2) ",)
            print(np.sum([j[2]*k*np.sin(l) for (j,k,l) in zip(
                    [T2, T3, T4, T5],
                    [self.PARAM['l2'], self.PARAM['l3'], self.PARAM['l4'], self.PARAM['l5']],
                    [self.PARAM['delta_2_rad'], self.PARAM['delta_3_rad'], self.PARAM['delta_4_rad'], self.PARAM['delta_5_rad']]
                )]))
            print("       T2[2]) ",)
            print(T2[2])
            print("       T3[2]) ",)
            print(T3[2])
            print("       T4[2]) ",)
            print(T4[2])
            print("       T5[2]) ",)
            print(T5[2])

            # 22 [u_tta]
            i.append(float(
                np.sum(np.asarray([(((j in [1,4])*2)-1) for j in idx_quadrant]) * np.concatenate([j[0] for j in [tau2, tau3, tau4, tau5]])) +
                np.sum([-j[2]*k*np.cos(l) for (j,k,l) in zip(
                    [T2, T3, T4, T5],
                    [self.PARAM['l2'], self.PARAM['l3'], self.PARAM['l4'], self.PARAM['l5']],
                    [self.PARAM['delta_2_rad'], self.PARAM['delta_3_rad'], self.PARAM['delta_4_rad'], self.PARAM['delta_5_rad']]
                )])
            ))

            print("------------------")
            print("u_tta = ",)
            print(i[-1])
            print("   1) ",)
            print(np.sum(np.asarray([(((j in [1,4])*2)-1) for j in idx_quadrant]) * np.concatenate([j[0] for j in [tau2, tau3, tau4, tau5]])))
            print("       tau2[1]) ",)
            print(tau2[1])
            print("       tau3[1]) ",)
            print(tau3[1])
            print("       tau4[1]) ",)
            print(tau4[1])
            print("       tau5[1]) ",)
            print(tau5[1])
            print("   2) ",)
            print(np.sum([-j[2]*k*np.cos(l) for (j,k,l) in zip(
                    [T2, T3, T4, T5],
                    [self.PARAM['l2'], self.PARAM['l3'], self.PARAM['l4'], self.PARAM['l5']],
                    [self.PARAM['delta_2_rad'], self.PARAM['delta_3_rad'], self.PARAM['delta_4_rad'], self.PARAM['delta_5_rad']]
                )]))
            print("       T2[2]) ",)
            print(T2[2])
            print("       T3[2]) ",)
            print(T3[2])
            print("       T4[2]) ",)
            print(T4[2])
            print("       T5[2]) ",)
            print(T5[2])

            # 23 [u_psi]
            i.append(float(
                np.sum([(-j[0]*k*np.sin(l))+(j[1]*k*np.cos(l))+ll[2] for (j,k,l,ll) in zip(
                    [T2,                        T3,                        T4,                        T5                       ],
                    [self.PARAM['l2'],          self.PARAM['l3'],          self.PARAM['l4'],          self.PARAM['l5']         ],
                    [self.PARAM['delta_2_rad'], self.PARAM['delta_3_rad'], self.PARAM['delta_4_rad'], self.PARAM['delta_5_rad']],
                    [tau2,                      tau3,                      tau4,                      tau5                     ]
                )])
            ))

            print("------------------")
            print("u_psi = ",)
            print(i[-1])
            print("   1) ")
            print("       tau2[2]) ",)
            print(tau2[2])
            print("       tau3[2]) ",)
            print(tau3[2])
            print("       tau4[2]) ",)
            print(tau4[2])
            print("       tau5[2]) ",)
            print(tau5[2])

            print("   2) ")
            print("       T2[0]) ",)
            print(T2[0])
            print("       T3[0]) ",)
            print(T3[0])
            print("       T4[0]) ",)
            print(T4[0])
            print("       T5[0]) ",)
            print(T5[0])

            print("   3) ")
            print("       T2[1]) ",)
            print(T2[1])
            print("       T3[1]) ",)
            print(T3[1])
            print("       T4[1]) ",)
            print(T4[1])
            print("       T5[1]) ",)
            print(T5[1])

            # 24..26 [rp_I]
            i += self.dynquad.calc_rp(t)[0:3,0].squeeze().tolist()
            # 27..29 [rpp_I]
            i += self.dynquad.calc_rpp(t)[0:3,0].squeeze().tolist()
            # 30..33 [euler I->1]
            i += list(self.dynquad.get_euler_body1())

            # print one line with 'i' content:
            print("si,",)
            print(' '.join(["%18.09e" % j for j in i]))


    def gravity(self, RI2i):
        g_i = np.asarray([[0],[0],[9.81]])
        g_b = dot(RI2i, g_i)
        #g_b = np.asarray([[0],[0],[0]])
        return g_b

    #---------------------------#
    def force_mainbody(self, t, RI2i):
        return self.gravity(RI2i) * self.dynquad.mass_body

    def force_prop_2(self, t, RI2i):
        F,_ = self.prop2.get_FT()
        F = np.asarray([[0],[0],[F]])
        #F = np.asarray([[0],[0],[0]])
        return (self.gravity(RI2i)*self.dynquad.mass_2) - F

    def force_prop_3(self, t, RI2i):
        F,_ = self.prop3.get_FT()
        F = np.asarray([[0],[0],[F]])
        #F = np.asarray([[0],[0],[0]])
        return (self.gravity(RI2i)*self.dynquad.mass_3) - F

    def force_prop_4(self, t, RI2i):
        F,_ = self.prop4.get_FT()
        F = np.asarray([[0],[0],[F]])
        #F = np.asarray([[0],[0],[0]])
        return (self.gravity(RI2i)*self.dynquad.mass_4) - F

    def force_prop_5(self, t, RI2i):
        F,_ = self.prop5.get_FT()
        F = np.asarray([[0],[0],[F]])
        #F = np.asarray([[0],[0],[0]])
        return (self.gravity(RI2i)*self.dynquad.mass_5) - F


    #---------------------------#
    def momt_mainbody(self, t, RI2i):
        Mmain = self.zeros.copy()
        _,M = self.prop2.get_FT()
        Mmain -= dot(self.R221, np.asarray([[0],[0],[M]]))
        _,M = self.prop3.get_FT()
        Mmain -= dot(self.R321, np.asarray([[0],[0],[M]]))
        _,M = self.prop4.get_FT()
        Mmain -= dot(self.R421, np.asarray([[0],[0],[M]]))
        _,M = self.prop5.get_FT()
        Mmain -= dot(self.R521, np.asarray([[0],[0],[M]]))
        return Mmain

    def momt_prop_2(self, t, RI2i):
        _,M = self.prop2.get_FT()
        M = np.asarray([[0],[0],[M]])
        #print "momt_prop_2 = %20.15e %20.15e %20.15e" % (M[0], M[1], M[2])
        return M

    def momt_prop_3(self, t, RI2i):
        _,M = self.prop3.get_FT()
        M = np.asarray([[0],[0],[M]])
        #print "momt_prop_3 = %20.15e %20.15e %20.15e" % (M[0], M[1], M[2])
        return M

    def momt_prop_4(self, t, RI2i):
        _,M = self.prop4.get_FT()
        M = np.asarray([[0],[0],[M]])
        #print "momt_prop_4 = %20.15e %20.15e %20.15e" % (M[0], M[1], M[2])
        return M

    def momt_prop_5(self, t, RI2i):
        _,M = self.prop5.get_FT()
        M = np.asarray([[0],[0],[M]])
        #print "momt_prop_5 = %20.15e %20.15e %20.15e" % (M[0], M[1], M[2])
        return M


#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):
    myquad = CMYQUAD()

#====================================#
