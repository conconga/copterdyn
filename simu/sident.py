#====================================#
# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: System Identification.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.

#====================================#
##WWww=--  import section: --=wwWW##
import numpy                  as np
import matplotlib.pyplot      as plt
import ffrls                  as rls
import copy
from   matrixbeautify  import CMATRIXBEAUTIFY
from   numpy           import dot

M    = np.genfromtxt('log')
Tmax = M[-1,0]
print "time_max = %f" % Tmax

# all the data, from cmyquad identification "print":
t           = M[:,0]
g_b         = M[:,range(1,4)]
u           = M[:,range(4,8)]
omega_01_1  = M[:,range(8,11)]
omegap_01_1 = M[:,range(17,20)]
rp_1        = M[:,range(11,14)]
rpp_1       = M[:,range(14,17)]
uT          = M[:,20]
u_phi       = M[:,21]
u_tta       = M[:,22]
u_psi       = M[:,23]
rp_0        = M[:,range(24,27)]
rpp_0       = M[:,range(27,30)]
euler       = M[:,range(30,33)]


fig = 1;
##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('g_b')
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(t, g_b[:,i], hold=False)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('euler I-1')
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(t, euler[:,i], hold=False)
    plt.ylabel(["$\phi$", "$\\theta$", "$\\psi$"][i])
    plt.grid(True)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('u')
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(t, u[:,i], hold=False)
    plt.ylim(0,1)
    plt.grid(True)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('uvw x ddt(uvw)')
for i in range(3):
    plt.subplot(2,3,1+i)
    plt.plot(t, rp_1[:,i], hold=False)
    plt.grid(True)
    plt.subplot(2,3,4+i)
    plt.plot(t, rpp_1[:,i], hold=False)
    plt.grid(True)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('rp_0  x  rpp_0')
for i in range(3):
    plt.subplot(2,3,1+i)
    plt.plot(t, rp_0[:,i], hold=False)
    plt.grid(True)
    plt.subplot(2,3,4+i)
    plt.plot(t, rpp_0[:,i], hold=False)
    plt.grid(True)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('pqr x ddt(pqr)')
for i in range(3):
    plt.subplot(2,3,1+i)
    plt.plot(t, omega_01_1[:,i], hold=False)
    plt.grid(True)
    plt.subplot(2,3,4+i)
    plt.plot(t, omegap_01_1[:,i], hold=False)
    plt.grid(True)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('uT, u_phi, u_tta, u_psi')
data = (uT, u_phi, u_tta, u_psi)
leg  = ('uT', 'u_phi', 'u_tta', 'u_psi')
for i in range(4):
    plt.subplot(2,2,1+i)
    plt.plot(t, data[i], hold=False)
    plt.ylabel(leg[i])
    plt.grid(True)

####################
## Adopted Model: ##
####################
#
#    xp[k+1] = xp[k] + T.a1.uT[k] + T.gx[k]
#    yp[k+1] = yp[k] + T.a2.uT[k] + T.gy[k]
#    zp[k+1] = zp[k] + T.a3.uT[k] + T.gz[k]
#
#    p[k+1]  = p[k]  + T.a4.q[k].r[k] + T.a5.q[k].w[k] + T.a6.u_phi[k]
#    q[k+1]  = q[k]  + T.a7.p[k].r[k] + T.a8.p[k].w[k] + T.a9.u_tta[k]
#    r[k+1]  = r[k]  + T.a10.p[k].q[k] + T.a11.u_psi[k]
#
####################

a1    = rls.CFFRLS([0],     lbd=0.99)
a2    = rls.CFFRLS([0],     lbd=0.99)
a3    = rls.CFFRLS([0],     lbd=0.99)
a456  = rls.CFFRLS([0,0,0], lbd=0.99)
a789  = rls.CFFRLS([0,0,0], lbd=0.99)
a1011 = rls.CFFRLS([0,0],   lbd=0.99)

log_a  = list()

for i in range(len(t)):
    j = []

    a1.update(rpp_1[i,0]-g_b[i,0], uT[i])
    j.append(float(a1.theta.squeeze()))

    a2.update(rpp_1[i,1]-g_b[i,1], uT[i])
    j.append(float(a2.theta.squeeze()))

    a3.update(rpp_1[i,2]-g_b[i,2], uT[i])
    j.append(float(a3.theta.squeeze()))

    a456.update(omegap_01_1[i,0], np.asarray([
        omega_01_1[i,1] * omega_01_1[i,2],
        omega_01_1[i,1] * rp_1[i,2],
        u_phi[i]]))
    j += list(a456.theta.squeeze())

    a789.update(omegap_01_1[i,1], np.asarray([
        omega_01_1[i,0] * omega_01_1[i,2],
        omega_01_1[i,0] * rp_1[i,2],
        u_tta[i]]))
    j += list(a789.theta.squeeze())

    a1011.update(omegap_01_1[i,2], np.asarray([
        omega_01_1[i,0] * omega_01_1[i,1],
        u_psi[i]]))
    j += list(a1011.theta.squeeze())

    log_a.append(j)

# show results:
T = 1./100

print "------------------------------------------------------------------------------------------------------------"
print "for a sample rate of %1.1f[ms], the discrete system is:" % (T*1e3)
print
print "xp[k+1] = xp[k] + (%1.2e).uT[k] + T.gx[k]" % (T*a1.theta)
print "yp[k+1] = yp[k] + (%1.2e).uT[k] + T.gy[k]" % (T*a2.theta)
print "zp[k+1] = zp[k] + (%1.2e).uT[k] + T.gz[k]" % (T*a3.theta)
print
print "p[k+1]  = p[k]  + (%1.2e).q[k].r[k] + (%1.2e).q[k].w[k] + (%1.2e).u_phi[k]" % tuple(T*a456.theta)
print "q[k+1]  = q[k]  + (%1.2e).p[k].r[k] + (%1.2e).p[k].w[k] + (%1.2e).u_tta[k]" % tuple(T*a789.theta)
print "r[k+1]  = r[k]  + (%1.2e).p[k].q[k] + (%1.2e).u_psi[k]" % tuple(T*a1011.theta)
print "------------------------------------------------------------------------------------------------------------"

# convert to array:
log_a = np.asarray(log_a)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('identification @alfa')
for i in range(11):
    plt.subplot(3,4,i+1)
    plt.plot(log_a[:,i], hold=False)
    plt.grid(True)
    txt = 'a_%d' % (i+1)
    plt.legend((txt,))

####################
## Adopted Model: ##
####################
#
#   
#   [ uT    ]    = [ b11  b12  b13  b14 ]   [ u2 ]
#   [ u_phi ]    = [ b21  b22  b23  b24 ] . [ u3 ]
#   [ u_tta ]    = [ b31  b32  b33  b34 ]   [ u4 ]
#   [ u_psi ][k] = [ b41  b42  b43  b44 ]   [ u5 ][k]
#   
#
####################

line1 = rls.CFFRLS([0,0,0,0], lbd=0.95+0.032)
line2 = rls.CFFRLS([0,0,0,0], lbd=0.95+0.032)
line3 = rls.CFFRLS([0,0,0,0], lbd=0.95+0.032)
line4 = rls.CFFRLS([0,0,0,0], lbd=0.95+0.032)

log_b = list()
exp   = 1.

for i in range(len(t)):

    line1.update(uT[i], u[i,:]**exp)
    j = list(line1.theta.squeeze())

    line2.update(u_phi[i], u[i,:]**exp)
    j += list(line2.theta.squeeze())

    line3.update(u_tta[i], u[i,:]**exp)
    j += list(line3.theta.squeeze())

    line4.update(u_psi[i], u[i,:]**exp)
    j += list(line4.theta.squeeze())

    log_b.append(j)

# convert to array:
log_b = np.asarray(log_b)

# matrix B:
B = np.vstack((
    line1.theta.T,
    line2.theta.T,
    line3.theta.T,
    line4.theta.T
))

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('identification @beta')
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.plot(log_b[:,i], hold=False)
    plt.grid(True)
    txt = 'b_%d' % (i+1)
    plt.legend((txt,))


####################
## Adopted Model: ##
####################
#
#    xp[k+1] = xp[k] + T.gx[k] + T.a1^T.u[k]
#    yp[k+1] = yp[k] + T.gy[k] + T.a2^T.u[k]
#    zp[k+1] = zp[k] + T.gz[k] + T.a3^T.u[k]
#
#    p[k+1]  = p[k]  + T.a4.q[k].r[k] + T.a5.q[k].w[k] + T.a6^T.u[k]
#    q[k+1]  = q[k]  + T.a7.p[k].r[k] + T.a8.p[k].w[k] + T.a9^T.u[k]
#    r[k+1]  = r[k]  + T.a10.p[k].q[k] + T.a11^T.u[k]
#
####################

line1 = rls.CFFRLS([0,0,0,0], lbd=0.99)
line2 = rls.CFFRLS([0,0,0,0], lbd=0.99)
line3 = rls.CFFRLS([0,0,0,0], lbd=0.99)
line4 = rls.CFFRLS([0,0,0,0,0,0], lbd=0.99)
line5 = rls.CFFRLS([0,0,0,0,0,0], lbd=0.99)
line6 = rls.CFFRLS([0,0,0,0,0], lbd=0.99)

log_c = list()

for i in range(len(t)):
    line1.update(rpp_1[i,0] - g_b[i,0], u[i,:])
    j = list(line1.theta.squeeze())

    line2.update(rpp_1[i,1] - g_b[i,1], u[i,:])
    j += list(line2.theta.squeeze())

    line3.update(rpp_1[i,2] - g_b[i,2], u[i,:])
    j += list(line3.theta.squeeze())

    line4.update(omegap_01_1[i,0], np.asarray(
        [
            omega_01_1[i,1] * omega_01_1[i,2],
            omega_01_1[i,1] * rp_1[i,2]
        ] + 
        list(u[i,:].squeeze())
    ))
    j += list(line4.theta.squeeze())

    line5.update(omegap_01_1[i,1], np.asarray(
        [
            omega_01_1[i,0] * omega_01_1[i,2],
            omega_01_1[i,0] * rp_1[i,2]
        ] + 
        list(u[i,:].squeeze())
    ))
    j += list(line5.theta.squeeze())

    line6.update(omegap_01_1[i,2], np.asarray(
        [ omega_01_1[i,0] * omega_01_1[i,1] ] +
        list(u[i,:].squeeze())
    ))
    j += list(line6.theta.squeeze())

    log_c.append(j)


##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('identification @all')
plt.plot(t, log_c, hold=False)
plt.grid(True)

#######################################################
##  Adopted Model:                                   ##
#######################################################
# in navigation frame:                               ##
#                                                    ##
#  z_pp_n = cos(phi).cos(tta).uT/m + gz              ##
#         = a1.cos(phi).cos(tta).uT + gz             ##
#                                                    ##
#######################################################

log_zp = []
a1    = rls.CFFRLS([0], lbd=0.99+0.01)

for i in range(len(t)):
    j = []

    cphictta = np.cos(euler[i,0]) * np.cos(euler[i,1])
    a1.update((rpp_0[i,2]-9.8)/cphictta, uT[i])
    j += [float(a1.theta.squeeze())]

    log_zp += j

print
print "------------------------------------------------------------------------------------------------------------"
print "z_pp_n(t) = (%1.2e).cos(phi).cos(tta).uT + gz" % a1.theta
print "(the quad mass shall be close to %1.2f[kg])" % (1./a1.theta)
print "------------------------------------------------------------------------------------------------------------"
print

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('z_n model')
plt.plot(t, log_zp, hold=False)
plt.grid(True)

######################################################
## Adopted Model:                                   ##
######################################################
#  dot(p) = [q.r  -q.Om  u_phi] . [c1 c2 1/Ix]^T    ##
#  dot(q) = [p.r   p.Om  u_tta] . [c3 c4 1/Iy]^T    ##
#  dot(r) = [p.q  u_psi] . [c5 1/Iz]^T              ##
######################################################

# [c1 c2 1/Ix]
v1     = rls.CFFRLS([0,0,0], lbd=0.99)
v2     = rls.CFFRLS([0,0,0], lbd=0.99)
v3     = rls.CFFRLS([0,0], lbd=0.99)
log_v1 = []
log_v2 = []
log_v3 = []

# change the resolution of data:
Fs      = 40. # [Hz]
Ts      = 1./Fs
c_t     = np.arange(0, t.max(), Ts)
c_dot_p = np.interp(c_t, t, omegap_01_1[:,0])
c_dot_q = np.interp(c_t, t, omegap_01_1[:,1])
c_dot_r = np.interp(c_t, t, omegap_01_1[:,2])
c_p     = np.interp(c_t, t, omega_01_1[:,0])
c_q     = np.interp(c_t, t, omega_01_1[:,1])
c_r     = np.interp(c_t, t, omega_01_1[:,2])
c_Om    = np.interp(c_t, t, uT)
c_uphi  = np.interp(c_t, t, u_phi)
c_utta  = np.interp(c_t, t, u_tta)
c_upsi  = np.interp(c_t, t, u_psi)

for i in range(len(c_t)):
    v1.update(c_dot_p[i], np.asarray([ c_q[i]*c_r[i], -c_q[i]*c_Om[i], c_uphi[i] ]))
    v2.update(c_dot_q[i], np.asarray([ c_p[i]*c_r[i],  c_p[i]*c_Om[i], c_utta[i] ]))
    v3.update(c_dot_r[i], np.asarray([ c_p[i]*c_q[i],  c_upsi[i] ]))
    log_v1.append(list(v1.theta.squeeze()))
    log_v2.append(list(v2.theta.squeeze()))
    log_v3.append(list(v3.theta.squeeze()))

log_v1 = np.asarray(log_v1)
log_v2 = np.asarray(log_v2)
log_v3 = np.asarray(log_v3)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('RLS for attitude model')

plt.subplot(3,3,1)
plt.plot(c_t, log_v1[:,0], hold=False)
plt.ylabel('c1')
plt.grid(True)

plt.subplot(3,3,2)
plt.plot(c_t, log_v1[:,1], hold=False)
plt.ylabel('c2')
plt.grid(True)

plt.subplot(3,3,3)
plt.plot(c_t, log_v1[:,2], hold=False)
plt.ylabel('1/Ix')
plt.grid(True)

plt.subplot(3,3,4)
plt.plot(c_t, log_v2[:,0], hold=False)
plt.ylabel('c3')
plt.grid(True)

plt.subplot(3,3,5)
plt.plot(c_t, log_v2[:,1], hold=False)
plt.ylabel('c4')
plt.grid(True)

plt.subplot(3,3,6)
plt.plot(c_t, log_v2[:,2], hold=False)
plt.ylabel('1/Iy')
plt.grid(True)

plt.subplot(3,3,7)
plt.plot(c_t, log_v3[:,0], hold=False)
plt.ylabel('c5')
plt.grid(True)

plt.subplot(3,3,8)
plt.plot(c_t, log_v3[:,1], hold=False)
plt.ylabel('1/Iz')
plt.grid(True)


print
print "------------------------------------------------------------------------------------------------------------"
print "dot(p) = [q.r  -q.Om  u_phi] . [(%1.2e)  (%1.2e)  (%1.2e)]^T" % (log_v1[-1,0], log_v1[-1,1], log_v1[-1,2])
print "dot(q) = [p.r   p.Om  u_tta] . [(%1.2e)  (%1.2e)  (%1.2e)]^T" % (log_v2[-1,0], log_v2[-1,1], log_v2[-1,2])
print "dot(r) = [p.q  u_psi] . [(%1.2e)  (%1.2e)]^T" % (log_v3[-1,0], log_v3[-1,1])
print "------------------------------------------------------------------------------------------------------------"

###########################################
#  Adopted Model:                         #
###########################################
#                                         #
#              [ uT    ]                  #
#   u_G[k]   = [ u_phi ]                  #
#              [ u_tta ]                  #
#              [ u_psi ][k]               #
#                                         #
#              [ u2 ]                     #
#   u_2:5[k] = [ u3 ]                     #
#              [ u4 ]                     #
#              [ u5 ][k]                  #
#                                         #
#   continuous:                           #
#   dot(u_G) = -L'1.u_G + L'2.u_2:5       #
#                                         #
#   discrete:                             #
#   u_G[k+1] = (I-L1).u_G[k] + L2.u_2:5   #
#            = M.[u_G u_2:5]^T            #
#                                         #
###########################################

# each line of M:
m1 = rls.CFFRLS([0 for i in range(5)], lbd=0.99+0.01)
m2 = rls.CFFRLS([0 for i in range(5)], lbd=0.99+0.01)
m3 = rls.CFFRLS([0 for i in range(5)], lbd=0.99+0.01)
m4 = rls.CFFRLS([0 for i in range(5)], lbd=0.99+0.01)
log_m1 = []
log_m2 = []
log_m3 = []
log_m4 = []

# change the resolution of data:
Fs      = 50. # [Hz]
Ts      = 1./Fs
c_t     = np.arange(0, t.max(), Ts)
c_u2    = np.interp(c_t, t, u[:,0])
c_u3    = np.interp(c_t, t, u[:,1])
c_u4    = np.interp(c_t, t, u[:,2])
c_u5    = np.interp(c_t, t, u[:,3])
c_uT    = np.interp(c_t, t, uT)
c_uphi  = np.interp(c_t, t, u_phi)
c_utta  = np.interp(c_t, t, u_tta)
c_upsi  = np.interp(c_t, t, u_psi)

for i in range(len(c_t)-1):
    m1.update(  c_uT[i+1], np.asarray([   c_uT[i], c_u2[i], c_u3[i], c_u4[i], c_u5[i]]))
    m2.update(c_uphi[i+1], np.asarray([ c_uphi[i], c_u2[i], c_u3[i], c_u4[i], c_u5[i]]))
    m3.update(c_utta[i+1], np.asarray([ c_utta[i], c_u2[i], c_u3[i], c_u4[i], c_u5[i]]))
    m4.update(c_upsi[i+1], np.asarray([ c_upsi[i], c_u2[i], c_u3[i], c_u4[i], c_u5[i]]))
    log_m1.append(list(m1.theta.squeeze()))
    log_m2.append(list(m2.theta.squeeze()))
    log_m3.append(list(m3.theta.squeeze()))
    log_m4.append(list(m4.theta.squeeze()))

M = np.vstack((
        np.asarray(log_m1[-1]),
        np.asarray(log_m2[-1]),
        np.asarray(log_m3[-1]),
        np.asarray(log_m4[-1])
    ))
L1  = np.eye(4) - np.diag(M[:,0])
#L1[L1<0] = 1e-2
L2  = M[:,1:5]

log_m1 = np.asarray(log_m1)
log_m2 = np.asarray(log_m2)
log_m3 = np.asarray(log_m3)
log_m4 = np.asarray(log_m4)


##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('dot(u_G) = -L\'1.u_G + L\'2.u_2:5  (I)')

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(c_t[1:], eval("log_m%d"%(i+1)))
    plt.title("M[%d,:]"%(i+1))
    plt.grid(True)

# the discrete system:
# x[k+1]  = P.x[k]  +  G.u[k]
# the continuous system:
# \dot{x} = A.x(t)  +  B.u(t)
P = np.diag(M[:,0])
G = M[:,1:5]
A = 2.0*dot(np.linalg.inv(np.eye(4)+P), P-np.eye(4))/Ts
B = dot(np.eye(4)-(A*Ts/2.0), G)/np.sqrt(Ts)

# save current settings
npo = copy.deepcopy(np.get_printoptions()['formatter'])
# change formatter
np.set_printoptions(formatter={'float_kind': lambda x: "(%9.2e)"%x if x!=0 else "(         )"})
# vvv       print with new format      vvv
print "P ="
print P
print "G ="
print G
print "A ="
print A
print "B ="
print B
# ^^^  end of printing with new format ^^^
# restore current settings
np.set_printoptions(formatter=npo)

print
print "------------------------------------------------------------------------------------------------------------"
print "for a sample rate of %d[Hz], the discrete system is:\n" % (Fs)
print "[ uT   ]          [(%9.2e)                                    ] [ uT   ]        [ (%9.2e) (%9.2e) (%9.2e) (%9.2e) ] [ u2 ]   " % tuple(M[0,:])
print "[ uphi ]       =  [           (%9.2e)                         ]x[ uphi ]     +  [ (%9.2e) (%9.2e) (%9.2e) (%9.2e) ]x[ u3 ]   " % tuple(M[1,:])
print "[ utta ]          [                       (%9.2e)             ] [ utta ]        [ (%9.2e) (%9.2e) (%9.2e) (%9.2e) ] [ u4 ]   " % tuple(M[2,:])
print "[ upsi ][k+1]     [                                   (%9.2e) ] [ upsi ][k]     [ (%9.2e) (%9.2e) (%9.2e) (%9.2e) ] [ u5 ][k]" % tuple(M[3,:])
print "------------------------------------------------------------------------------------------------------------"
print

#########################
## test of the results ##
#########################
at_t = np.asarray([0, 0.4, 0.8, 1.2, 1.6])
at_t = np.linspace(0, Tmax, 6)
do_u = np.asarray([
    #uT  uphi utta upsi
    [1,  0,   1,   0,     0],
    [1,  0,   0,   1,     0],
    [1,  1,   0,   0,     0],
    [1,  1,   1,   1,     0],
])

uG     = np.asarray([[0],[0],[0],[0]])
log_uG = []
for i in range(len(c_t)):
    j   = np.max((len(at_t[at_t < c_t[i]])-1, 0))
    u25 = 0.2*do_u[:,j].reshape((4,1))
    #print c_t[i], "  ", j, "  ", u25.T
    uG  = dot(np.eye(4)-L1, uG) + dot(L2, u25)
    #print "%10.2f   [[ %12.2e %12.2e %12.2e %12.2e   ]]   " % tuple([c_t[i]] + uG.squeeze().tolist())
    log_uG.append(uG.squeeze().tolist())

log_uG = np.asarray(log_uG)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('test u2:5 --> uG  (I)')
leg  = ('uT', 'u_phi', 'u_tta', 'u_psi')

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(c_t, log_uG[:,i])
    plt.title(leg[i])
    plt.grid(True)


###########################################
#  Adopted Model:                         #
###########################################
#                                         #
#              [ uT    ]                  #
#   u_G[k]   = [ u_phi ]                  #
#              [ u_tta ]                  #
#              [ u_psi ][k]               #
#                                         #
#              [ u2 ]                     #
#   u_2:5[k] = [ u3 ]                     #
#              [ u4 ]                     #
#              [ u5 ][k]                  #
#                                         #
#   continuous:                           #
#                                         #
#   dot(u_G) =  A.u_G + B.u_2:5           #
#                                         #
#   with                                  #
#                                         #
#   A = diag([f,g,h,k])                   #
#                                         #
#   B = [  a  a  a  a ]                   #
#       [ -b -b  b  b ]                   #
#       [  c -c -c  c ]                   #
#       [  d -d  d -d ]                   #
#                                         #
###########################################

# each line of M:
m1 = rls.CFFRLS([0 for i in range(2)], lbd=0.99+0.01)
m2 = rls.CFFRLS([0 for i in range(2)], lbd=0.99+0.01)
m3 = rls.CFFRLS([0 for i in range(2)], lbd=0.99+0.01)
m4 = rls.CFFRLS([0 for i in range(2)], lbd=0.99+0.01)
log_m1 = []
log_m2 = []
log_m3 = []
log_m4 = []

# change the resolution of data:
Fs      = 50. # [Hz]
Ts      = 1./Fs
c_t     = np.arange(0, t.max(), Ts)
c_u2    = np.interp(c_t, t, u[:,0])
c_u3    = np.interp(c_t, t, u[:,1])
c_u4    = np.interp(c_t, t, u[:,2])
c_u5    = np.interp(c_t, t, u[:,3])
c_uT    = np.interp(c_t, t, uT)
c_uphi  = np.interp(c_t, t, u_phi)
c_utta  = np.interp(c_t, t, u_tta)
c_upsi  = np.interp(c_t, t, u_psi)

for i in range(len(c_t)-1):
    m1.update(  c_uT[i+1], np.asarray([   c_uT[i],  c_u2[i] + c_u3[i] + c_u4[i] + c_u5[i]]))
    m2.update(c_uphi[i+1], np.asarray([ c_uphi[i], -c_u2[i] - c_u3[i] + c_u4[i] + c_u5[i]]))
    m3.update(c_utta[i+1], np.asarray([ c_utta[i],  c_u2[i] - c_u3[i] - c_u4[i] + c_u5[i]]))
    m4.update(c_upsi[i+1], np.asarray([ c_upsi[i],  c_u2[i] - c_u3[i] + c_u4[i] - c_u5[i]]))
    log_m1.append(list(m1.theta.squeeze()))
    log_m2.append(list(m2.theta.squeeze()))
    log_m3.append(list(m3.theta.squeeze()))
    log_m4.append(list(m4.theta.squeeze()))

P = np.diag(
    np.hstack([
        m1.theta[0],
        m2.theta[0],
        m3.theta[0],
        m4.theta[0]
    ])
)

G = np.asarray([
    np.hstack((  m1.theta[1],  m1.theta[1],  m1.theta[1],  m1.theta[1] )),
    np.hstack(( -m2.theta[1], -m2.theta[1],  m2.theta[1],  m2.theta[1] )),
    np.hstack((  m3.theta[1], -m3.theta[1], -m3.theta[1],  m3.theta[1] )),
    np.hstack((  m4.theta[1], -m4.theta[1],  m4.theta[1], -m4.theta[1] ))
])

log_m1 = np.asarray(log_m1)
log_m2 = np.asarray(log_m2)
log_m3 = np.asarray(log_m3)
log_m4 = np.asarray(log_m4)


##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('dot(u_G) = -L\'1.u_G + L\'2.u_2:5  (II)')

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(c_t[1:], eval("log_m%d"%(i+1)))
    plt.title("M[%d,:]"%(i+1))
    plt.grid(True)

# the discrete system:
# x[k+1]  = P.x[k]  +  G.u[k]
# the continuous system:
# \dot{x} = A.x(t)  +  B.u(t)
A = 2.0*dot(np.linalg.inv(np.eye(4)+P), P-np.eye(4))/Ts
B = dot(np.eye(4)-(A*Ts/2.0), G)/np.sqrt(Ts)

mbeauty = CMATRIXBEAUTIFY("%10.3e")
print "---- model ----------------------"
print "\\dot{uG} = A.uG(t)  +  B.u2:5(t)"
print "\nA ="
mbeauty(A)
print "\nB ="
mbeauty(B)
print "---------------------------------"

#########################
## test of the results ##
#########################
at_t = np.asarray([0, 0.4, 0.8, 1.2, 1.6])
at_t = np.linspace(0, Tmax, 6)
do_u = np.asarray([
    #uT  uphi utta upsi
    [1,  0,   1,   0,     0],
    [1,  0,   0,   1,     0],
    [1,  1,   0,   0,     0],
    [1,  1,   1,   1,     0],
])

uG     = np.asarray([[0],[0],[0],[0]])
log_uG = []
for i in range(len(c_t)):
    j   = np.max((len(at_t[at_t < c_t[i]])-1, 0))
    u25 = 0.2*do_u[:,j].reshape((4,1))
    #print c_t[i], "  ", j, "  ", u25.T
    uG  = dot(P, uG) + dot(G, u25)
    #print "%10.2f   [[ %12.2e %12.2e %12.2e %12.2e   ]]   " % tuple([c_t[i]] + uG.squeeze().tolist())
    log_uG.append(uG.squeeze().tolist())

log_uG = np.asarray(log_uG)

##-- figure --##
fig = fig + 1; pfig = plt.figure(fig); plt.clf();
pfig.canvas.set_window_title('test u2:5 --> uG  (II)')
leg  = ('uT', 'u_phi', 'u_tta', 'u_psi')

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(c_t, log_uG[:,i])
    plt.title(leg[i])
    plt.grid(True)







##----##
plt.show(block=False)
##----##

