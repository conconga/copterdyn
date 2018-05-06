#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This class simulates the dynamical behaviour of a quadcopter.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.


#====================================#
##WWww=--  import section: --=wwWW##
import numpy               as np
import cshowquad3d         as sq
from   numpy           import dot
from   numpy           import linalg
from   kdebug          import CKDEBUG
from   navfunc         import CNAVFUNC
from   cmuxdemux       import CMUXDEMUX
from   somemath        import *

#====================================#
## \brief class CQUADCOPTER ##
## \author luciano kruk     ##
#
## \description:
#
# The body is described by 5 bodies, as:
#
# [body 1] = central body, with 4 different arms, and 4 motors at each arm.
# [body 2] = a rotor and propeller placed at the end of arm 2.
# [body 3] = a rotor and propeller placed at the end of arm 3.
# [body 4] = a rotor and propeller placed at the end of arm 4.
# [body 5] = a rotor and propeller placed at the end of arm 5.
#
# This model is described by the following variables:
# '     l2..l5     ' : length of each arm
# ' alfa_2..alfa_5 ' : misalignment between body's arm and rotor axis
# 'delta_2..delta_5' : angle between body frame and copter arms
#
#====================================#

class CQUADCOPTER(CNAVFUNC, CKDEBUG, CMUXDEMUX):

    def __init__(self, PARAM):
        # bases:
        CKDEBUG.__init__(self,0)
        CMUXDEMUX.__init__(self, [10,10,20]) # [q=qp=10, quat=4x5]

        # inner variables:
        self.T_minus  = fn_elwisemult(PARAM['Tm'],   np.eye(3))
        self.T_plus   = fn_elwisemult(PARAM['TM'],   np.eye(3))
        self.T        = fn_elwisemult(PARAM['TM'],   np.eye(3))
        self.P        = PARAM['P'].copy()
        self.k        = PARAM['k'].copy()
        self.force_fn = PARAM['force_fn']
        self.momt_fn  = PARAM['momt_fn']
        self.l2       = PARAM['l2']
        self.l3       = PARAM['l3']
        self.l4       = PARAM['l4']
        self.l5       = PARAM['l5']
        self.delta_2_rad = PARAM['delta_2_rad']
        self.delta_3_rad = PARAM['delta_3_rad']
        self.delta_4_rad = PARAM['delta_4_rad']
        self.delta_5_rad = PARAM['delta_5_rad']
        self.mass_body   = PARAM['mass_body']
        self.mass_2      = PARAM['mass_2']
        self.mass_3      = PARAM['mass_3']
        self.mass_4      = PARAM['mass_4']
        self.mass_5      = PARAM['mass_5']

        # state vector (generalized coordinates):
        self.q = np.asarray([
                0,0,0,  # initial position
                0,0,0,  # body initial angles
                0,0,0,0 # rotor angles
            ])
        self.qp = np.zeros(self.q.shape) # first derivative

        # [qI21, qI22, qI23, qI23, qI25]
        # ( Ri2I =  R12I  . Ri21 )
        # (      = eye(3) . Ri21 )
        self.quat = [
                list(self.euler2Q((0,0,0))),
                list(self.C2Q(self.Q2C(PARAM['q221']).T)),
                list(self.C2Q(self.Q2C(PARAM['q321']).T)),
                list(self.C2Q(self.Q2C(PARAM['q421']).T)),
                list(self.C2Q(self.Q2C(PARAM['q521']).T))
        ]

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
        self.m = m

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
        self.J = J
        self.show_matrix_details("J", J, level=2)

        # long term simulation:
        self.t     = 0;
        self.state = self.state_mux((self.q,self.qp,self.quat)) # initial conditions

    
    def c_plus(self, q):
        x      = q[0]
        y      = q[1]
        z      = q[2]

        c11_1  = [-x, -y, -z]
        zeros  = [0,0,0]
        c_plus = c11_1 + zeros + zeros + zeros + zeros

        return np.asarray(c_plus).reshape((len(c_plus),1))


    def c_minus(self):
        zeros  = np.asarray([0,0,0]).reshape((3,1))
        c12_1  = dot(self.Q2C(self.euler2Q((0,0,-self.delta_2_rad))), np.asarray([self.l2,0,0]).reshape((3,1)))
        c13_1  = dot(self.Q2C(self.euler2Q((0,0,-self.delta_3_rad))), np.asarray([self.l3,0,0]).reshape((3,1)))
        c14_1  = dot(self.Q2C(self.euler2Q((0,0,-self.delta_4_rad))), np.asarray([self.l4,0,0]).reshape((3,1)))
        c15_1  = dot(self.Q2C(self.euler2Q((0,0,-self.delta_5_rad))), np.asarray([self.l5,0,0]).reshape((3,1)))

        return np.vstack((zeros, c12_1, c13_1, c14_1, c15_1))


    def cp_plus(self, qp):
        xp      = qp[0]
        yp      = qp[1]
        zp      = qp[2]

        zeros   = [0,0,0]
        cp_plus = [-xp, -yp, -zp] + zeros + zeros + zeros + zeros

        return np.asarray(cp_plus).reshape((len(cp_plus),1))


    def A_minus(self, quat):
        qI21  = quat[0]

        A              = np.zeros((15,15))
        A[0:3,0:3]     = np.eye(3)
        A[3:6,3:6]     = self.Q2C(qI21).T
        A[6:9,6:9]     = self.Q2C(qI21).T
        A[9:12,9:12]   = self.Q2C(qI21).T
        A[12:15,12:15] = self.Q2C(qI21).T

        return A


    def A_plus(self, quat):
        qI21  = quat[0]
        qI22  = quat[1]
        qI23  = quat[2]
        qI24  = quat[3]
        qI25  = quat[4]

        A              = np.zeros((15,15))
        A[0:3,0:3]     = self.Q2C(qI21).T
        A[3:6,3:6]     = self.Q2C(qI22).T
        A[6:9,6:9]     = self.Q2C(qI23).T
        A[9:12,9:12]   = self.Q2C(qI24).T
        A[12:15,12:15] = self.Q2C(qI25).T

        return A


    def Omega_minus(self, qp):
        Omega_minus = dot(self.P, qp.reshape((qp.shape[0],1)))
        return Omega_minus


    ##WWww=--  mux state vector  --=wwWW##
    def state_mux(self, T):
        q    = T[0]
        qp   = T[1]
        quat = T[2]

        state = self.mux(q,qp,np.concatenate(quat))

        #self.show_matrix_details("state", state, level=2)
        self.assert_infnan(state, 'state')
        return state


    ##WWww=--  demux state vector  --=wwWW##
    def state_demux(self, state):
        self.assert_infnan(state, 'state')

        q,qp,aux = self.demux(state)
        quat     = [list(aux[(i*4):4*(i+1)]) for i in range(5)]

        # self.show_matrix_details("q", q, level=2)
        # self.show_matrix_details("qp", qp, level=2)
        # self.show_matrix_details("quat", quat, level=2)
        return q, qp, quat


    ##WWww=--  actions taken before time update  --=wwWW##
    def pre_update(self, t):
        # save input forces and momments:
        self.forcemoments_save(t)


    ##WWww=--  actions taken after time update  --=wwWW##
    def pos_update(self, t, state):
        # internal backup:
        self.state = state
        self.t     = t
        self.q, self.qp, self.quat = self.state_demux(state)

        # save current result:
        self.state_save()

    ##WWww=--  get current RI21 transformation matrix  --=wwWW##
    def get_RI21(self):
        return self.Q2C(self.quat[0])

    ##WWww=--  get euler angles [rad] for I->1 frames  --=wwWW##
    def get_euler_body1(self):
        return self.Q2euler(self.quat[0])

    ##WWww=--  get current \omega_{01}^0  --=wwWW##
    def get_omega_01_0(self):
        return self.Omega_minus(self.qp)[0:3]

    ##WWww=--  get current \omega_{01}^0  --=wwWW##
    def get_omegap_01_0(self, t):
        q       = self.q
        qp      = self.qp
        quat    = self.quat
        _,qpp,_ = self.state_demux( self.dstate_dt( self.state_mux((q, qp, quat)), t ) )
        return dot(self.P, qpp.reshape((qpp.shape[0],1)))[0:3]

    ##WWww=--  derivative  --=wwWW##
    def dstate_dt(self, state, t):
        """ Calculates the derivative of the multibody model. """ 
        #print "time = %f" % t

        T_minus  = self.T_minus
        T_plus   = self.T_plus
        T        = self.T
        P        = self.P
        k        = self.k
        m        = self.m
        J        = self.J

        ##WWww=--  state rescue:  --=wwWW##
        q, qp, quat = self.state_demux(state)

        ##WWww=--  calc a1:  --=wwWW##
        A_plus      = self.A_plus(quat)
        A_minus     = self.A_minus(quat)
        c_plus      = self.c_plus(q)
        c_plus_X    = fn_blockskew(c_plus)
        c_minus     = self.c_minus()
        c_minus_X   = fn_blockskew(c_minus)
        Omega_minus = self.Omega_minus(qp)

        self.show_matrix_details("T", T, level=2)
        self.show_matrix_details("A_plus", A_plus, level=2)
        self.show_matrix_details("c_plus_X", c_plus_X, level=2)
        self.show_matrix_details("T_plus", T_plus, level=2)
        self.show_matrix_details("A_minus", A_minus, level=2)
        self.show_matrix_details("P", P, level=2)

        a1_aux = dot(A_plus, dot(c_plus_X,   dot(A_plus.T,  self.T_plus))) - dot(A_minus, dot(c_minus_X, dot(A_minus.T, self.T_minus)))
        a1 = dot(self.T, dot(a1_aux, dot(A_minus, self.P)) - dot(A_plus, self.k))

        self.show_matrix_details("a1", a1, level=2)

        ##WWww=--  calc a20:  --=wwWW##
        omega_minus     = dot(T_minus, dot(A_minus, Omega_minus))
        omega_miminus   = dot(A_minus.T, omega_minus)
        omega_miminus_X = fn_blockskew(omega_miminus)

        omega_plus      = dot(T_plus,  dot(A_minus, Omega_minus))
        omega_pluplus   = dot(A_plus.T, omega_plus)
        omega_pluplus_X = fn_blockskew(omega_pluplus)

        cp_plus = self.cp_plus(qp)

        a20_1 =  dot(A_plus, dot(omega_pluplus_X, dot(c_plus_X, omega_pluplus)))
        a20_2 = -dot(A_minus, dot(omega_miminus_X, dot(c_minus_X, omega_miminus)))
        a20_3 =  dot(a1_aux, dot(A_minus, dot(omega_miminus_X, Omega_minus)))
        a20_4 = -2.0 * dot(A_plus, dot(omega_pluplus_X, cp_plus))

        a20   = dot(T, a20_1 + a20_2 + a20_3 + a20_4)
        self.show_matrix_details("a20", a20, level=2)

        ##WWww=--  calc b1:  --=wwWW##
        b1 = dot(T, dot(A_minus, P))
        self.show_matrix_details("b1", b1, level=2)

        ##WWww=--  calc b20:  --=wwWW##
        b20 = dot(T, dot(A_minus, dot(omega_miminus_X, Omega_minus)))
        self.show_matrix_details("b20", b20, level=2)

        ##WWww=--  calc body matrixes:  --=wwWW##
        a_b1   = dot(A_plus.T, a1)
        b_b1   = dot(A_plus.T, b1)
        a_b20  = dot(A_plus.T, a20)
        b_b20  = dot(A_plus.T, b20)
        self.show_matrix_details("a_b1", a_b1, level=3)
        self.show_matrix_details("b_b1", b_b1, level=3)
        self.show_matrix_details("a_b20", a_b20, level=3)
        self.show_matrix_details("b_b20", b_b20, level=3)
        # self.assert_infnan(a_b1, "a_b1")
        # self.assert_infnan(b_b1, "b_b1")
        # self.assert_infnan(a_b20, "a_b20")
        # self.assert_infnan(b_b20, "b_b20")

        ##WWww=--  forces and moments at bodies' frames  --=wwWW##
        F,M = self.calc_forcesmoments_body(t, quat)

        ##WWww=--  calc G:  --=wwWW##
        G = dot(a_b1.T, dot(m, a_b1)) + dot(b_b1.T, dot(J, b_b1))
        self.show_matrix_details("G", G, level=2)
        # self.assert_infnan(G, "G")

        ##WWww=--  calc H:  --=wwWW##
        omega_b = dot(A_plus.T, omega_plus)
        H_1     = dot(a_b1.T, F - dot(m, a_b20))
        H_2     = dot(b_b1.T, M - dot(J, b_b20) - dot(fn_blockskew(omega_b), dot(J, omega_b)))
        H       = H_1 + H_2
        self.show_matrix_details("H", H, level=2)
        # self.assert_infnan(H, 'H')

        ##WWww=--  calc qpp:  --=wwWW##
        qpp = linalg.solve(G,H)
        self.show_matrix_details("qpp", qpp, level=2)
        # self.assert_infnan(qpp, 'qpp')

        ##WWww=--  only for debug:  --=wwWW##
        if False:
            rp   = dot(a1, qp)
            rpp  = a20 + dot(a1, qpp)
            w    = dot(b1, qp)
            wp   = b20 + dot(b1, qpp)

            self.show_matrix_details("rp",  rp,  level=2)
            self.show_matrix_details("rpp", rpp, level=2)
            self.show_matrix_details("w",   w,   level=2)
            self.show_matrix_details("wp",  wp,  level=2)

        ##WWww=--  derivative of the quaternions  --=wwWW##
        quatp = list()
        for i in range(5):
            bomega = omega_plus[range(3*i, 3*(i+1)), :]
            bA     = A_plus[range(3*i, 3*(i+1))][:,range(3*i, 3*(i+1))]
            self.show_matrix_details("bomega", bomega, level=3)
            self.show_matrix_details("bA", bA, level=3)

            quatp.append(self.dqdt(quat[i], dot(bA.T, bomega)))

        state = self.state_mux((qp, qpp.squeeze(), quatp))

        self.show_matrix_details("[qp,qpp,quatp]", state, level=2)

        return state

    ##WWww=--  calculate linear vel in inertial frame  --=wwWW##
    def calc_rp(self,t):
        q       = self.q
        qp      = self.qp
        quat    = self.quat

        A_plus      = self.A_plus(quat)
        A_minus     = self.A_minus(quat)
        c_plus      = self.c_plus(q)
        c_plus_X    = fn_blockskew(c_plus)
        c_minus     = self.c_minus()
        c_minus_X   = fn_blockskew(c_minus)

        cp_plus = self.cp_plus(qp)

        a1_aux = dot(A_plus, dot(c_plus_X,   dot(A_plus.T,  self.T_plus))) - dot(A_minus, dot(c_minus_X, dot(A_minus.T, self.T_minus)))
        a1     = dot(self.T, dot(a1_aux, dot(A_minus, self.P)) - dot(A_plus, self.k))

        return dot(a1, qp).reshape((a1.shape[0],1))

    ##WWww=--  calculate acceleration in inertial frame  --=wwWW##
    def calc_rpp(self, t):
        q       = self.q
        qp      = self.qp
        quat    = self.quat
        _,qpp,_ = self.state_demux( self.dstate_dt( self.state_mux((q, qp, quat)), t ) )

        A_plus      = self.A_plus(quat)
        A_minus     = self.A_minus(quat)
        c_plus      = self.c_plus(q)
        c_plus_X    = fn_blockskew(c_plus)
        c_minus     = self.c_minus()
        c_minus_X   = fn_blockskew(c_minus)
        Omega_minus = self.Omega_minus(qp)

        omega_minus     = dot(self.T_minus, dot(A_minus, Omega_minus))
        omega_miminus   = dot(A_minus.T, omega_minus)
        omega_miminus_X = fn_blockskew(omega_miminus)

        omega_plus      = dot(self.T_plus,  dot(A_minus, Omega_minus))
        omega_pluplus   = dot(A_plus.T, omega_plus)
        omega_pluplus_X = fn_blockskew(omega_pluplus)

        cp_plus = self.cp_plus(qp)

        a1_aux = dot(A_plus, dot(c_plus_X,   dot(A_plus.T,  self.T_plus))) - dot(A_minus, dot(c_minus_X, dot(A_minus.T, self.T_minus)))
        a1     = dot(self.T, dot(a1_aux, dot(A_minus, self.P)) - dot(A_plus, self.k))

        a20_1  =  dot(A_plus, dot(omega_pluplus_X, dot(c_plus_X, omega_pluplus)))
        a20_2  = -dot(A_minus, dot(omega_miminus_X, dot(c_minus_X, omega_miminus)))
        a20_3  =  dot(a1_aux, dot(A_minus, dot(omega_miminus_X, Omega_minus)))
        a20_4  = -2.0 * dot(A_plus, dot(omega_pluplus_X, cp_plus))

        a20    = dot(self.T, a20_1 + a20_2 + a20_3 + a20_4)

        return a20 + dot(a1, qpp.reshape((qpp.shape[0],1)))

    ##WWww=--  store current state in a state buffer --=wwWW##
    def state_save(self):
        try:
            self.state_buf.append((self.t, self.state))
        except AttributeError:
            self.state_buf = [(self.t,self.state)]

        return

    ##WWww=--  restore state buffer  --=wwWW##
    def state_fetch(self):
        time  = [i[0] for i in self.state_buf]
        state = [i[1] for i in self.state_buf]
        return (time, state)

    ##WWww=--  clear current state buffer  --=wwWW##
    def state_clear(self):
        self.state_buf = list()
        return

    ##WWww=--  calculate position and velocity from state buffer  --=wwWW##
    def calc_pos_vel(self):
        t,state_list = self.state_fetch()

        r  = list()
        rp = list()
        for i in state_list:
            q,qp,quat   = self.state_demux(i)
            A_plus      = self.A_plus(quat)
            A_minus     = self.A_minus(quat)
            c_plus      = self.c_plus(q)
            c_plus_X    = fn_blockskew(c_plus)
            c_minus     = self.c_minus()
            c_minus_X   = fn_blockskew(c_minus)
            Omega_minus = self.Omega_minus(qp)

            a1_aux = dot(A_plus, dot(c_plus_X,   dot(A_plus.T,  self.T_plus))) - dot(A_minus, dot(c_minus_X, dot(A_minus.T, self.T_minus)))
            a1 = dot(self.T, dot(a1_aux, dot(A_minus, self.P)) - dot(A_plus, self.k))

            r.append(dot(self.T, dot(A_minus, c_minus) - dot(A_plus, c_plus)))
            # TODO verificar as equacoes que usam o T^T a partir do calculo de r!
            rp.append(dot(self.T, dot(a1, qp)))

        return (
            np.asarray(r).squeeze(), 
            np.asarray(rp).squeeze()
        )

    ##WWww=--  extract the generalized coordinates from the state buffer  --=wwWW##
    def state_to_q_qp_quat(self):
        t,state_list = self.state_fetch()

        q    = list()
        qp   = list()
        quat = list()
        for i in state_list:
            q_aux, qp_aux, quat_aux = self.state_demux(i)
            q.append(q_aux)
            qp.append(qp_aux)
            quat.append(quat_aux)

        return (
            np.asarray(q).squeeze(),
            np.asarray(qp).squeeze(),
            np.asarray(quat).reshape((len(state_list),-1))
        )

    ##WWww=--  calculation of forces and moments described at bodies' frames  --=wwWW##
    def calc_forcesmoments_body(self, t, quat):
        # transformation matrixes
        RI21  = self.Q2C(quat[0])
        RI22  = self.Q2C(quat[1])
        RI23  = self.Q2C(quat[2])
        RI24  = self.Q2C(quat[3])
        RI25  = self.Q2C(quat[4])

        # force vector:
        F1 = self.force_fn[0](t, RI21)
        F2 = self.force_fn[1](t, RI22)
        F3 = self.force_fn[2](t, RI23)
        F4 = self.force_fn[3](t, RI24)
        F5 = self.force_fn[4](t, RI25)

        F = np.vstack((F1, F2, F3, F4, F5)) # body's frames
        self.show_matrix_details("F", F, level=1)

        # moment vector:
        M1 = self.momt_fn[0](t, RI21)
        M2 = self.momt_fn[1](t, RI22)
        M3 = self.momt_fn[2](t, RI23)
        M4 = self.momt_fn[3](t, RI24)
        M5 = self.momt_fn[4](t, RI25)

        M = np.vstack((M1, M2, M3, M4, M5)) # body's frames
        self.show_matrix_details("M", M, level=1)

        return (F,M)


    ##WWww=--  store forces and moments (body frame) in a buffer  --=wwWW##
    def forcemoments_save(self, t):
        q,qp,quat   = self.state_demux(self.state)
        F,M         = self.calc_forcesmoments_body(t, quat)

        try:
            self.forcemoments_buf.append((self.t, F, M))
        except AttributeError:
            self.forcemoments_buf = [(self.t, F, M)]


    ##WWww=--  restore forces and moments values  --=wwWW##
    def forcemoments_fetch(self):
        time   = [i[0] for i in self.forcemoments_buf]
        forces = [i[1] for i in self.forcemoments_buf]
        momts  = [i[2] for i in self.forcemoments_buf]
        return (time, forces, momts)

    

#################################
## ##WWww=--  main:  --=wwWW## ##
#################################
if (__name__ == "__main__"):
    pass
#====================================#
