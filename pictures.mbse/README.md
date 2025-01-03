-   [<span class="toc-section-number">1</span> copterdyn](#copterdyn)
-   [<span class="toc-section-number">2</span> history](#history)
-   [<span class="toc-section-number">3</span>
    refactoring](#refactoring)
-   [<span class="toc-section-number">4</span> terms](#terms)
    -   [<span class="toc-section-number">4.1</span> Mission](#mission)
    -   [<span class="toc-section-number">4.2</span> Path and
        Trajectory](#path-and-trajectory)
        -   [<span class="toc-section-number">4.2.1</span> Path](#path)
        -   [<span class="toc-section-number">4.2.2</span>
            Trajectory](#trajectory)
-   [<span class="toc-section-number">5</span> System Of
    Interest](#system-of-interest)
    -   [<span class="toc-section-number">5.1</span> use case
        diagram](#use-case-diagram)
    -   [<span class="toc-section-number">5.2</span> context
        diagram](#context-diagram)
    -   [<span class="toc-section-number">5.3</span> Activity: Simulate
        Hover](#activity-simulate-hover)
    -   [<span class="toc-section-number">5.4</span> Activity: Perform
        Navigation](#activity-perform-navigation)
    -   [<span class="toc-section-number">5.5</span> Activity: Simulate
        Sensors](#activity-simulate-sensors)
    -   [<span class="toc-section-number">5.6</span> Functional
        Allocation](#functional-allocation)
    -   [<span class="toc-section-number">5.7</span> System
        Decomposition](#system-decomposition)

# copterdyn

**copterdyn** is a sophisticated simulator for a quadcopter modeled
using multibody simulation concepts.

# history

The quadcopter simulator is outdated, and with its first version
completed in 2017. It was developed using Python 2.7, without any
compatibility to any Python 3.0. This was critical in particular with
debugging information and frame generation (animations). Additionally,
the first implementation had other issues. For instance, while the
simulation of the dynamics impressive, the project lacked a proper
propeller model, and a piloting (controlling) algorithm. Moreover, the
interfaces to any guidance inadequate.

# refactoring

As my career banked to systems engineering, I made a decision to reshape
the project with some reasonable increments with the intention of
bringing it reviving the project also for python 3, populating the
simulation with the complete GNC (guidance-navigation-control) and
sensing, and enhancing the clarity at any interface level.

# terms

## Mission

First of all, a robot needs a mission, a goal, a target, a reason, or
several of them.

A fire-and-forget missile, for instance, is a robot lauched with two
missions:

-   phase 1: the mission is a waypoints, and the robot searches for a
    target along its cruise;
-   phase 2: the mission is to deliver a package (warhead!) as close as
    possible to the target taking in account, of course, the wish of the
    latter to refuse it.

when landing an aircraft autonomously, the mission is to reach a
touchpoint (position) with a minimum speed. When landing a quadcopter,
the mission is defined by a touchpoint, but the speed shall be as small
as possible to avoid a crash.

That said, the mission is a representation of the robot’s final state
and is independent of its current state.

## Path and Trajectory

Along my career I have had several discussions where these terms appear
sometimes interchangeably and not rarely with particular meanings.
Whenever one asks for clarification about any of the terms, a new
discussion starts and the original focus gets lost.

I will describe in the next lines the denotation I first learnt in
discussions with seniors and experts and technical articles. I learned
also in international collaborations that the terms were used abroad
with the same meaning.

### Path

Somewhere in the brain of a robot there is a piece of math that
fictiously imagine what the robot needs to do to satisfy an assigned
mission, and the solution has several dependencies, as the type of
mission, related constrains, among others.

The **path** is the sequence of intermediate states between the current
one and the mission, not necessarily linear due to constraints.

This is what our cellphones do for us when the mission is to go to a
particular address, maybe another city, from the current position. A
path planner will break down the straigh line connecting source and
destiny in several intermediate states compliant with constraits, maybe
regulations.

An aircraft leaving São Paulo targeting London cannot flight straight
due to the earth curvature, but along arcs over the earth surface. There
is an uncountable number of possible arcs to serve as the flight
**path**, and only one which is the shortest, and maybe another one
which uses the minimum ammount of querosene seizing atmospheric winds.

A vehicle leaving a mall and targeting home, 10 blocks north plus 10
blocks east, cannot move freely along the diagonal of the quadrilateral.
It must instead follow specific traffic rules. In virtue of the
mentioned rules, some **paths** will be longer than others, some faster,
some nicer, some cheaper.

Another nice example: recent hypersonic weapons “fly” at the borders of
earth atmosphere. The **path** calculation considers limited payload,
low drag, and fast reach.

The main take-away here is that, if the path is followed by the robot,
the mission is supposed to be fulfilled.

### Trajectory

The **trajectory**, on the other hand, is a sequence of states that
enable the robot to follow the path. When the robot follows the
trajectory, it remains on the path. When the state of a robot is
corrupted by noise or limited by constraints, it has at least two
alternatives to overcome the risk of not satisfing the mission, either
recalculating a path, or updating the **trajectory**.

A very simple example of this dilema comes from autonomous driving. In
this example, the path is a straigh line from the current position, over
downtown, up to the target position. But along the path, the robot will
face several crosswalks and STOP signs. It might recalculate the path
and maybe circunvent downtown for each crosswalk and pedestrian. But it
can also push decelerations and acceleration, preventing accidents,
following the regulations, protecting pedestrians, and following the
path. In this example, the **trajectory** is the sequence of
accelerations and decelerations that enable the vehicle to reach the
other side of the city overcoming noise over the path.

Rockets, missiles, aircrafts are sensitive to wind. Despite the random
behavior of the wind, the referred robots manage to follow the path by
constantly updating the guiding trajectory.

Technically, the trajectory is the sequence of selected characteristics
which are used as reference values for the dynamical behavior of the
robot.

![path_trajectory.png](path_trajectory.png?raw=true "A possible path for the helicopter, and the respective trajectory for a safe landing.")

# System Of Interest

The next paragraphs and respective diagrams will provide the details
about the new project, and what is expected to have achieved. The System
of Interest is modelled using the Model Based System Engineering (MBSE)
paradigm.

## use case diagram

The System-of-Interest will provide at least three services:

-   to simulate any hover maneuver of the quadcopter (configured by the
    user);
-   to simulate sensors as described by a comercial datasheet; GPS data
    is an input for the navigation, and will be generated by the sensor
    simulator;
-   to calculate any navigation data, position, velocity and
    acceleration.

![UseCases.JPG](UseCases.JPG?raw=true "Use-Case diagram, and details on stakeholders (I myself!) needs.")

## context diagram

The Context-Diagram for the System-of-Interest (SoI) does not depict any
surprises. The SoI interacts with the User at lauching of any
simulation, and provides data to a logger, file, or disply unit.

![SoI_Context.JPG](SoI_Context.JPG?raw=true "Context Diagram.")

## Activity: Simulate Hover

The UseCase “Simulate Hover” is decomposed into its subfunctions. Here
the role of Guidance and Piloting are represented by the functions
“calculate trajectory” and “calculate control commands”. Most of the
functions are allowed to perform in parallel.

This Activity satisfies some requirements:

-   to simulate the full kinematics of a quadcopter, also
    non-simmetrical quadcopter;
-   to enable the development of control laws for the plant;
-   to enable the development of guidance laws based on configured
    missions (waypoints, abrupt structural changes (fail))

![simulate_hover.JPG](simulate_hover.JPG?raw=true "UseCase: simulate hover")

## Activity: Perform Navigation

The UseCase “Perform Navigation” is decomposed into its subfunctions.
The main role of the Activity is to allow the development of navigation
algorithms fusing any source available. For instance, some possible
sources supporting the navigation algorithms are:

-   gyrometer measurements;
-   accelerometer measurements;
-   GPS or similar data;
-   altitude.

The modularity enables the evaluation of different concepts, like
gyroless-navigation, or navigation supported by static reference points
measured by radars or cameras, for instance.

![perform_navigation.JPG](perform_navigation.JPG?raw=true "UseCase: perform navigation")

## Activity: Simulate Sensors

The UseCase “Simulate Sensors” is decomposed into its subfunctions. The
main goal of this activity is to convert the state vector describing the
dynamic system into sensor measurements, if any sensors are installed on
the system. For example, a classical sensor set (CSS) for navigation
consists of 3 gyros, 3 accelerometers, and GPS. Several assumptions have
to be make, for instance:

-   the CSS is not installed exactly at the center of gravity of the
    quadcopter;
-   the GPS data is delayed with regard to the actual dynamics;
-   the sensors are not ideal, and provide noise measurements.

Due to the modularity of this model, the sensor simulation might be
replaced (or simply configured) to include magnetic sensors, radars,
cameras, and whatever is necessary for your (my) application.

The output of this activity is designed to be the input of the
navigation.

![simulate_sensors.JPG](simulate_sensors.JPG?raw=true "UseCase: simulate navigation")

## Functional Allocation

The SysML allows the allocation of \<\<Activity\>\> as any
\<\<Block\>\>, by any \<\<Block\>\> and \<\<Activity\>\>. This enables
the modeling of reuse, as a single \<\<Activity\>\> might be a \<\<Part
Property\>\> of multiple elements in the model.

The next diagram depicts the functional allocation of the UseCases to
the System of Interest. In addition, it shows the allocation of a new
\<\<Activity\>\>: “manage time”. This activity is intended to manage the
pace of the simulation, continuous and discrete, and handle time events,
for instance a change of configuration programed to happen at a
particular timeslot.

![Functional
Allocation](functional_allocation.JPG?raw=true "System of Interest: Functional Allocation")

## System Decomposition

Now, replacing the activities allocated to the System of Interest by
their sub-activities, and grouping the result into the first set of
subsystems of the System of Interest, the result is depicted in the next
diagram.

![FunctionalAllocation_subsystems_noname.JPG](FunctionalAllocation_subsystems_noname.JPG?raw=true "Sub-activities allocated to sub-elements")

The sybsystems are renamed harmonically based on the set of activities
allocated to each of them.

| old name | new name         |
|----------|------------------|
| Block 1  | copterdyn        |
| Block 2  | piloting         |
| Block 3  | guidance         |
| Block 4  | sensor simulator |
| Block 5  | manager          |
| Block 6  | navigation       |

Renaming Blocks based on the performed activites.

![subsystems.JPG](subsystems.JPG?raw=true "Sub-activities allocated to sub-elements")
