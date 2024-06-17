# \author    Luciano Augusto Kruk
# \website   www.kruk.eng.br
# \date      2017.0
#
# \description: This class displays animation of quadcopters.
#
# \license: Please feel free to use and modify this, but keep this header as
#           part of yours. Thanks.


#====================================#
##WWww=--  import section: --=wwWW##
import numpy                       as np;
import matplotlib.animation        as animation
import mpl_toolkits.mplot3d.axes3d as p3
from   numpy                   import dot;
from   numpy                   import linalg;
from   kdebug                  import CKDEBUG;
from   scipy.integrate         import odeint;
from   somemath                import *
from   navfunc                 import CNAVFUNC

#====================================#
## \brief class cshowquad3d ##
## \author luciano kruk   ##
#
## \description:
#
#====================================#
class cshowquad3d(CNAVFUNC):
    def __init__(self, pfig, lpos_I_buf, quat):
        """
        lpos_I_buf: matrix [N x (5*3)] [m]
        quat      : matrix [N x (4*5)] [5 x qI2i]
        """

        self.pfig          = pfig
        self.ax            = p3.Axes3D(pfig);
        self.lpos_I_buf    = lpos_I_buf
        self.quat          = quat
        self.deg           = 0.
        self.bExportFrames = False

        # clean axis:
        self.ax.cla()

        # lines:
        r = [0,0]
        self.arms  = [ self.ax.plot(r,r,r)[0] for i in range(4) ] 
        self.props = [ self.ax.plot(r,r,r)[0] for i in range(4) ]

        # body positions:
        self.pts  = [ self.ax.scatter(0,0,0, marker='o', animated=False) for i in range(5) ]

        # body tracker line:
        self.tracker,     =  self.ax.plot(r,r,r)
        self.tr_buffer    = list()
        self.tr_maxbuflen = 50


    def init(self):
        """ initialize animation """

        self.ax.set_xlabel('X');
        self.ax.set_ylabel('Y');
        self.ax.set_zlabel('Z');

        return

    def set_line_3d(self, obj, x, y, z):
        """
        NOTE: there is no .set_data() for 3 dim data...

        type(obj) = < matplotlib.pyplot.Axes3D(pfig) >
        """

        obj.set_data(x,y)
        obj.set_3d_properties(z)

    def set_scatter_3d(self, obj, x, y, z):
        """
        NOTE: there is no .set_data() for 3 dim data...

        type(obj) = < matplotlib.pyplot.Axes3D(pfig) >
        """

        obj.set_offsets(np.hstack((x,y)))
        obj.set_3d_properties(z, 'z')

    def tr_add2buffer(self, r):
        self.tr_buffer.append(r)
        if (len(self.tr_buffer) > self.tr_maxbuflen):
            del(self.tr_buffer[0])

    def tr_getxyz(self):
        buf = np.asarray(self.tr_buffer)
        return (buf[:,i] for i in range(3))

    def scale_calclims(self, vmin, data):
        delta = np.max(data) - np.min(data)
        if (delta < vmin):
            vlim = [
                np.min(data) - ((vmin-delta)/2.),
                np.max(data) + ((vmin-delta)/2.)
            ]
        else:
            vlim = [np.min(data), np.max(data)]
        return vlim

    def set_scale(self):
        xmin = 0.3
        ymin = 0.3
        zmin = 0.3
        x,y,z = self.tr_getxyz()

        self.ax.set_xlim(self.scale_calclims(xmin, x))
        self.ax.set_ylim(self.scale_calclims(ymin, y))
        self.ax.set_zlim(self.scale_calclims(zmin, z))
        self.ax.invert_zaxis();

    def update_frame(self, idx):
        """ perform animation step """

        # only one item:
        lpos_I = fn_separate3x1(self.lpos_I_buf[idx,:])

        print("=======ani frame {:d}=========".format(idx))
        print("[{:10.4f} {:10.4f} {:10.4f} | {:10.4f} {:10.4f} {:10.4f} | {:10.4f} {:10.4f} {:10.4f} | {:10.4f} {:10.4f} {:10.4f} | {:10.4f} {:10.4f} {:10.4f}]".format( \
                self.lpos_I_buf[idx,0],
                self.lpos_I_buf[idx,1],
                self.lpos_I_buf[idx,2],
                self.lpos_I_buf[idx,3],
                self.lpos_I_buf[idx,4],
                self.lpos_I_buf[idx,5],
                self.lpos_I_buf[idx,6],
                self.lpos_I_buf[idx,7],
                self.lpos_I_buf[idx,8],
                self.lpos_I_buf[idx,9],
                self.lpos_I_buf[idx,10],
                self.lpos_I_buf[idx,11],
                self.lpos_I_buf[idx,12],
                self.lpos_I_buf[idx,13],
                self.lpos_I_buf[idx,14]
            ))

        # "center" body:
        r1 = lpos_I[0].squeeze()

        # rotor coordinates described in 'I':
        r2 = lpos_I[1].squeeze()
        r3 = lpos_I[2].squeeze()
        r4 = lpos_I[3].squeeze()
        r5 = lpos_I[4].squeeze()

        # arms:
        self.set_line_3d( self.arms[0], [r1[0], r2[0]], [r1[1], r2[1]], [r1[2], r2[2]])
        self.set_line_3d( self.arms[1], [r1[0], r3[0]], [r1[1], r3[1]], [r1[2], r3[2]])
        self.set_line_3d( self.arms[2], [r1[0], r4[0]], [r1[1], r4[1]], [r1[2], r4[2]])
        self.set_line_3d( self.arms[3], [r1[0], r5[0]], [r1[1], r5[1]], [r1[2], r5[2]])

        # bodies:
        self.set_scatter_3d(self.pts[0], r1[0], r1[1], r1[2])
        self.set_scatter_3d(self.pts[1], r2[0], r2[1], r2[2])
        self.set_scatter_3d(self.pts[2], r3[0], r3[1], r3[2])
        self.set_scatter_3d(self.pts[3], r4[0], r4[1], r4[2])
        self.set_scatter_3d(self.pts[4], r5[0], r5[1], r5[2])

        def calc_endpoint_prop(i, RI2i, r_prop_I, length, height):
            """
            Calculates the endpoints of a two-blade-propeller.
            i        : index of propeller (1..4)
            RI2i     : list with matrixes RI2i (i \in 0..4)
            r_prop_I : position of propeller described in 'I'
            length   : length of each blade
            height   : distance between arms and propellers
            """
            
            p1 = r_prop_I.reshape((3,1)) + \
                dot(RI2i[0].T, np.asarray([[0],[0],[-height]])) + \
                dot(RI2i[i].T, np.asarray([[length/2.],[0],[0]]))

            p2 = r_prop_I.reshape((3,1)) + \
                dot(RI2i[0].T, np.asarray([[0],[0],[-height]])) + \
                dot(RI2i[i].T, np.asarray([[-length/2.],[0],[0]]))

            return p1,p2


        # propellers:
        RI2i = [ self.Q2C(self.quat[idx,(4*i):(4*(i+1))]) for i in range(5) ]

        p1,p2 = calc_endpoint_prop(1, RI2i, r2, fn_norm(r2-r1)/4., fn_norm(r2-r1)/10.) # endpoints of propeller
        self.set_line_3d( self.props[0], [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]] )
        self.props[0].set_color('black')

        p1,p2 = calc_endpoint_prop(2, RI2i, r3, fn_norm(r3-r1)/4., fn_norm(r3-r1)/10.)
        self.set_line_3d( self.props[1], [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]] )
        self.props[1].set_color('black')

        p1,p2 = calc_endpoint_prop(3, RI2i, r4, fn_norm(r4-r1)/4., fn_norm(r4-r1)/10.)
        self.set_line_3d( self.props[2], [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]] )
        self.props[2].set_color('black')

        p1,p2 = calc_endpoint_prop(4, RI2i, r5, fn_norm(r5-r1)/4., fn_norm(r5-r1)/10.)
        self.set_line_3d( self.props[3], [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]] )
        self.props[3].set_color('black')

        if self.bExportFrames:
            self.pfig.savefig("cshowquad3d_%05d.jpg" % idx)


        # tracker:
        self.tr_add2buffer(r1)
        x,y,z = self.tr_getxyz()
        self.set_line_3d( self.tracker, x,y,z)
        self.set_scale()


        # return:
        return self.arms + self.pts + self.props



    def do_it(self):

        if False:
            from time import time
            t0 = time()
            self.update_frame(0)
            t1 = time()
            print("time to update/generate a new frame = {:1.1f}[ms]".format(1000.*(t1-t0)))

        """
        Makes an animation by repeatedly calling a function *func*, passing in (optional) arguments in *fargs*.

        *frames* can be a generator, an iterable, or a number of frames.

        *init_func* is a function used to draw a clear frame. If not given, the
        results of drawing from the first item in the frames sequence will be
        used. This function will be called once before the first frame.

        If blit=True, *func* and *init_func* should return an iterable of drawables to clear.

        *kwargs* include *repeat*, *repeat_delay*, and *interval*:
        *interval*       draws a new frame every *interval* milliseconds.
        *repeat*         controls whether the animation should repeat when the sequence of frames is completed.
        *repeat_delay*   optionally adds a delay in milliseconds before repeating the animation.
        """

        ani = animation.FuncAnimation(
            self.pfig, 
            self.update_frame, 
            frames=self.lpos_I_buf.shape[0],
            fargs=(), 
            interval=5, 
            blit=False, 
            init_func=self.init
        )

        print("creating animation in a .gif file...")
        ani.save('cshowquad3d.gif', fps=10, dpi=60, writer='imagemagick')


#====================================#
