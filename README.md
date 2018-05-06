# copterdyn

**copterdyn** is a sophisticated simulator for a quadcopter modeled using
multibody simulation concepts. The quadcopter is partitioned in five bodies,
each with individual physical characteristics, like inertia, geometry, mass,
and the multibody mathematical core solves all differential equations
describing the dynamical behavior. 

Additional differential equations might easily be plugged and solved together.
The propellers, for example, are modeled by a second order transfer function to
have their transient dynamical effects also simulated.

The simulation is written in Python, object oriented, with an object called
*cmyquad*, which might be a reference when one wants to model a different quad
but use the same numerical kernel of this simulation. A model includes:
- masses
- inertial characteristics
- length of each copter arm
- angles between body frame and each arm
- tilt angle between vertical and thrust (misalignment)

Most of quad simulations considers that each arm is separated by 90[deg] angles
from each other, has same lengths, all motors are equal, the inertia tensors
are diagonal, the thrust vectors are all parallel to each other, and the
gyroscopic effect of the propellers are negligible. With this assumptions, one
can develop a control project. The remaining question is: how would this
control project work with the tolerances, imperfections and misalignments of a
real quad? Such a simulation can be run hundreds of times with randomly
selected model values, and a statistical analysis of the results can lead to
better solutions, either concerning quad parts, or software project.

# bodies
It is time to clarify the meaning of "a quadcopter with five bodies". This
picture depicts the bodies. 

![bodies](./images/bodies.png?raw=true "Bodies")

- central body supporting arms, electronics, batteries, **and** the static part of each motor;
- four sets of propellers+rotors.

# geometry
As mentioned above, this simulator allows some additional degrees of
configuration. The traditional setup of orthogonal arms with same length is
not necessary anymore. An unusual configuration like this is allowed:

![non-orthogonal arms](./images/nonortho.png?raw=true "Non-Orthogonal Arms")

Moreover, a tilt in the thrust vector is also allowed (and necessary for a good
simulation!), like this: 

![alfa angles](./images/alfa.png?raw=true "Alpha Angles")

# frames
Internally the orientation of frames is adopted as: X forward, Y right, Z
down. 

![frames](./images/eixos.png?raw=true "main frames")
![frames non-orth](./images/delta.png?raw=true "non-orth main frame")

Each body has its own frame, therefore one might get the propeller angle
directly from its Euler angles.

# current status
The project of *cmyquad* still does not have a controller. That said,
please do not expect to see well-behaviored-flying quads. Here you can
check some current results:

1) perfectly symmetrical aligned quadcopter with four perfect equal
propellers.
![perfect](./images/report-no-deviation.gif?raw=true "perfect quad")

2) one of the perfect equal propellers is not so perfect and generates a
thrust vector with a misalignment of 3[deg].
![alfa=3deg](./images/report-alfa-3deg.gif?raw=true "misaligned thrust vector")

3) one of the arms is placed at 87[deg] instead of 90[deg].
![delta=3deg](./images/report-delta-3deg.gif?raw=true "misaligned arms")

It is worthless to say that a quadcopter without control is unstable, right?

# how to run

```sh
$ ipython
>>> %run simuquad.py
```

# how to change

Here you find some hints where to start from.

## change structural parameters

Go to the first lines of *cmyquad.\_\_init\_\_()*. There you can play with masses,
inertia, arms length, and angles. Make your changes and run it again.

## where to plug the control

Go to the first lines of *cmyquad.pre_update()*. The variable *u* is fed equally
to each propeller model. The controller will calculate the appropriate value of
each *u*.

## propeller model

Take a look at *cprop* methods.

## animated gif

In the file *simuquad.py*, look for a line with *sq.cshowquad3d()*. Enable it
to have a *.gif* file for each run.

# roadmap

Some ideas to continue this work are the modeling and implementation of:
- a minimal controller, in order to have some longer and more realistic simulations;
- a better propeller model;
- navigation, sensor models, estimators;
- effect of wind (disturbances) (torques and forces);
- guidance (optimal trajectory planning);
- and so on... 
