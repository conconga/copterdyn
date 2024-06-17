#!/usr/bin/python
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
# author:      Luciano Augusto Kruk
# website:     www.kruk.eng.br
#
# description: Package of functions for quaternions and
#               geodetic coordinates handling.
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>

import numpy as np
import math  as mt
from numpy import zeros,sin,cos,empty,sqrt;

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
class CGEO:
    def __init__(self, lat, lon, h):
        self.lat = lat;
        self.lon = lon;
        self.h   = h;

    def __repr__(self):
        K = 180./mt.pi;
        return "<CGEO: lat=%1.2f[deg]  lon=%1.2f[deg]  h=%1.1f[m]>" % (self.lat*K, self.lon*K, self.h)

class CRECT:
    def __init__(self, r_e):
        if type(r_e) is np.ndarray:
            self.x = r_e.squeeze()[0];
            self.y = r_e.squeeze()[1];
            self.z = r_e.squeeze()[2];
        elif type(r_e) in (tuple, list):
            self.x = r_e[0]
            self.y = r_e[1]
            self.z = r_e[2]
        else:
            print("ainda nao suportado!")

    def __repr__(self):
        K = 180./mt.pi;
        return "<CRECT: x=%1.2f[m]  y=%1.2f[m]  z=%1.2f[m]>" % (self.x, self.y, self.z)

    def aslist(self):
        return [
            self.x,
            self.y,
            self.z
        ];

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
class CNAVFUNC:
    """
    Geodetic Funcions
    """

    # Earth Elliptic Model #
    earth_a  = 6378137.0; # [m]
    earth_b  = 6356752.3142; # [m]
    wie      = 1.0 * 7.2921151467e-5;
    wie_e    = np.asarray([0,0,wie]).reshape((3,1));
    earth_f  = (earth_a-earth_b)/earth_a;
    earth_e  = sqrt(earth_f*(2.0-earth_f));
    earth_e2 = (earth_e**2.0);

    def __init__(self):
        pass

    def Rlambda(self, lat_rad):
        """
        : parameter : lat_rad [rad] latitude
        : output    : R_lbd
        """
        return (self.earth_a*(1.-self.earth_e2)) / ((1.-(self.earth_e2*(sin(lat_rad)**2)))**1.5);

    def Rphi(self, lat_rad):
        """
        : parameter : lat_rad [rad] latitude
        : output    : R_phi
        """
        return self.earth_a / sqrt(1.-(self.earth_e2*(sin(lat_rad)**2.0)));

    def euler2Q(self, euler):
        """
        Navigation -- from euler to Q.

        : parameter : phi   [rad]
        : parameter : theta [rad]
        : parameter : psi   [rad]
        : output    : Q4
        """
        phi   = euler[0]
        theta = euler[1]
        psi   = euler[2]

        half_phi   = 0.5*phi
        half_theta = 0.5*theta
        half_psi   = 0.5*psi;

        return np.asarray([
            (cos(half_phi)*cos(half_theta)*cos(half_psi)) + (sin(half_phi)*sin(half_theta)*sin(half_psi)),
            (sin(half_phi)*cos(half_theta)*cos(half_psi)) - (cos(half_phi)*sin(half_theta)*sin(half_psi)),
            (cos(half_phi)*sin(half_theta)*cos(half_psi)) + (sin(half_phi)*cos(half_theta)*sin(half_psi)),
            (cos(half_phi)*cos(half_theta)*sin(half_psi)) - (sin(half_phi)*sin(half_theta)*cos(half_psi))
        ]);

    def Q2euler(self, q):
        """
        Navigation -- from Q to euler.

        : input    : q
        : output   : phi   [rad]
        : output   : theta [rad]
        : output   : psi   [rad]
        """

        phi   = mt.atan2(2.0*((q[2]*q[3])+(q[0]*q[1])), (q[0]**2.0)-(q[1]**2.0)-(q[2]**2.0)+(q[3]**2.0));
        psi   = mt.atan2(2.0*((q[1]*q[2])+(q[0]*q[3])), (q[0]**2.0)+(q[1]**2.0)-(q[2]**2.0)-(q[3]**2.0));
        try:
            theta = mt.asin(2.0*((q[0]*q[2])-(q[1]*q[3])));
        except ValueError:
            print("ERRO: norm(Q) = {:f}".format(np.sqrt(np.sum(q**2))))
            theta = 0;

        return (phi, theta, psi)

    def Q2C(self, q):
        """
        Navigation -- from Q to C.

        If Q represents the transformation from 'a' to 'b', the matrix
        'C' represents 'Ca2b'.

        : input    : q
        : output   : C
        """

        #q = q.squeeze();
        C = np.empty((3,3));
        C[0,0] = (q[0]**2.0) + (q[1]**2.0) - (q[2]**2.0) - (q[3]**2.0);
        C[0,1] = 2.0 * ((q[1]*q[2]) + (q[0]*q[3]));
        C[0,2] = 2.0 * ((q[1]*q[3]) - (q[0]*q[2]));

        C[1,0] = 2.0 * ((q[1]*q[2]) - (q[0]*q[3]));
        C[1,1] = (q[0]**2.0) - (q[1]**2.0) + (q[2]**2.0) - (q[3]**2.0);
        C[1,2] = 2.0 * ((q[2]*q[3]) + (q[0]*q[1]));

        C[2,0] = 2.0 * ((q[1]*q[3]) + (q[0]*q[2]));
        C[2,1] = 2.0 * ((q[2]*q[3]) - (q[0]*q[1]));
        C[2,2] = (q[0]**2.0) - (q[1]**2.0) - (q[2]**2.0) + (q[3]**2.0);

        return C

    def C2Q(self, C):
        """
        Navigation -- from C to Q

        output: nparray() with Q
        """

        return self.euler2Q(self.C2euler(C))

    def C2euler(self, C):
        """
        Navigation -- from C to (phi,theta,psi)[rad]

        output: tuple with angles in [rad]
        """

        assert(C[2,2] != 0)
        assert(C[0,0] != 0)
        assert(C[0,2]>=-1 and C[0,2]<=1)

        phi   = np.arctan2(C[1,2], C[2,2])
        theta = np.arcsin(-C[0,2])
        psi   = np.arctan2(C[0,1], C[0,0])

        return (phi, theta, psi)

    def q1_prod_q2(self, q1, q2):
        """
        Navigation -- multiplies two quaternions

        Let q1 represent C_a2b, and q2 represent C_b2c.
        The product C_a2c = C_b2c.C_a2b might be represented
        by q3 = q1.q2

        output: np.array quaternion q3=q1.q2
        """

        q3 = np.array([
            (q1[0]*q2[0])-(q2[1]*q1[1])-(q2[2]*q1[2])-(q2[3]*q1[3]),
            (q2[0]*q1[1])+(q2[1]*q1[0])+(q2[2]*q1[3])-(q2[3]*q1[2]),
            (q2[0]*q1[2])+(q2[2]*q1[0])-(q2[1]*q1[3])+(q2[3]*q1[1]),
            (q2[0]*q1[3])+(q2[3]*q1[0])+(q2[1]*q1[2])-(q2[2]*q1[1])
        ])

        return q3

    def matrix_Q2euler(self, q):
        """
        Converts a matrix with quaternions (N x 4) to euler angles (N x 3).
        """

        N   = q.shape[0]
        dcm = np.zeros((N, 3))
        for i in range(N):
            phi,theta,psi = self.Q2euler(q[i,:])
            dcm[i,:] = [phi, theta, psi]

        return dcm

    def Re2n(self, lat, lon):
        """
        Navigation -- calculates Re2n(lat,lon)

        : input    : lat   [rad]
        : input    : lon   [rad]
        : output   : Re2n
        """

        Re2n = np.empty((3,3));
        Re2n[0,0] = -sin(lat)*cos(lon);
        Re2n[0,1] = -sin(lat)*sin(lon);
        Re2n[0,2] = cos(lat);
        Re2n[1,0] = -sin(lon);
        Re2n[1,1] = cos(lon);
        Re2n[1,2] = 0;
        Re2n[2,0] = -cos(lat)*cos(lon);
        Re2n[2,1] = -cos(lat)*sin(lon);
        Re2n[2,2] = -sin(lat);

        return Re2n

    def geo2rect(self, geo):
        """
        Converter coordenadas ECEF geodeticas para retangulares.
        pgeo [in] Coordenadas geodeticas.
        prect [out] Coordenadas retangulares.
        """

        s  = sin(geo.lat);
        RN = self.earth_a / sqrt(1.0 - (self.earth_e2 * s * s));

        return CRECT((
            (RN + geo.h) * cos(geo.lat) * cos(geo.lon),
            (RN + geo.h) * cos(geo.lat) * sin(geo.lon),
            ((RN * (1.0 - self.earth_e2)) + geo.h) * sin(geo.lat)
        ))

    def rect2geo(self, rect):
        """
        Converter coordenadas ECEF retangulares para geodeticas.
        pgeo [out] Coordenadas geodeticas.
        prect [in] Coordenadas retangulares.
        """

        p = sqrt((rect.x * rect.x) + (rect.y * rect.y));
        geo = CGEO(0,0,0);
        geo.h     = 0;
        RN          = self.earth_a;
        for i in range(100): # timeout
            #print "[lat h] = [%1.09f %1.03f]" % (geo.lat, geo.h)
            lastlat = geo.lat;
            lasth   = geo.h;

            # algoritmo de conversao:
            s               = rect.z / (((1.0 - self.earth_e2) * RN) + geo.h);
            geo.lat       = mt.atan((rect.z + (self.earth_e2 * RN * s)) / p);
            RN              = self.earth_a / sqrt(1.0 - (self.earth_e2 * s * s));
            geo.h         = (p / cos(geo.lat)) - RN;

            # erro:
            d = ((lastlat - geo.lat) * (lastlat - geo.lat)) + ((lasth - geo.h) * (lasth - geo.h));
            if (d < 1e-9):
               break;

        geo.lon = mt.atan2(rect.y, rect.x);

        return geo

    def dqdt(self, q, w):
        """
        The derivative of the quaternions is $\dot{q} = 1/2 .B(w).q$
        This funtion returns $\dot{q}$.
        """

        K      = 1e1
        cq     = np.asarray(q).reshape((4,1))
        epslon = 1.0 - np.sum(cq**2.0)

        B = np.asarray([
            [   0, -w[0], -w[1], -w[2]],
            [w[0],     0,  w[2], -w[1]],
            [w[1], -w[2],     0,  w[0]],
            [w[2],  w[1], -w[0],     0]
        ])

        dq = (0.5 * np.dot(B,cq)) + (K*epslon*cq)

        return list(dq.squeeze())

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>

if (__name__ == "__main__"):
    g = CNAVFUNC();

    print(g.Rlambda(20./57))
    print(g.Rphi(30./57))

    q = g.euler2Q((10./57, 20./57, -30./57))
    print(np.asarray(g.Q2euler(q))*57)

    print(g.Q2C(q))

    print(g.Re2n(0,0))
    print(g.Re2n(1,0.9))

    geo = CGEO(10./57, -30./57, 33);
    print(geo)
    rec = g.geo2rect(geo)
    print(rec)
    geo = g.rect2geo(rec)
    print(geo)


    #----------------------#
    # some dynamic tests:
    #----------------------#
    from   scipy.integrate import odeint;
    from   numpy           import dot;
    print()

    #  I: inertial frame
    #  b: body frame
    qI2b = g.euler2Q((0,0,0))

    # angular rotation between I and b:
    # \omega_{Ib}^I
    w = np.asarray([2./57,  0,   0]).reshape((3,1))

    def eqdiff(q,t,w):
        RI2b = g.Q2C(q)
        dqdt = g.dqdt(q, dot(RI2b,w))
        return dqdt

    # a vector described at I:
    F = np.asarray([0,0,1]).reshape((3,1))
    print("F = ")
    print(F.T)

    for t in [1,5,20,90]:
        # after t seconds, the quaternions should be:
        y = odeint(eqdiff, list(qI2b), [0,t], (w,))[1,:]
        # with these euler angles:
        euler = g.Q2euler(y)

        # and described at b:
        F_b = dot(g.Q2C(y), F)
        print("F_b(phi = {:1.03f}) = [{:1.03f} {:1.03f} {:1.03f}]".format(
            57.*euler[0], F_b[0], F_b[1], F_b[2]))

    #----------------------#
    # some convertion tests:
    #----------------------#

    euler = (10./57, -40./57, 163./57)
    Q     = g.euler2Q(euler)

    print("euler = ")
    print(np.degrees(euler))
    print(np.degrees(g.Q2euler(g.euler2Q(euler))))
    euler_2 = g.Q2euler(g.C2Q(g.Q2C(g.euler2Q(euler))))
    print(np.degrees(np.asarray(euler_2)))

    #----------------------#
    # quaternion product:
    #----------------------#

    print()
    print("quaternion product")
    q_a2b = g.euler2Q((-10., 33., -55.))
    q_b2c = g.euler2Q((44., -38., 77.))
    print("C_a2c = C_b2c . Ci_a2b =")
    print(dot(g.Q2C(q_b2c), g.Q2C(q_a2b)))
    print("C(q_b2c . qa2b) =")
    print(g.Q2C(g.q1_prod_q2(q_b2c, q_a2b)))

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
