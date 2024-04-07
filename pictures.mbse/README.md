# copterdyn

**copterdyn** is a sophisticated simulator for a quadcopter modeled using
multibody simulation concepts.

# history

The quadcopter simulator is old, and its first version was concluded in 2017.
It was developed using Python 2.7, without any compatibility to any Python
3.0. This was critical in particular with debugging information and frames
generations (animations).  There were other issues with this first
implementation. For example, although the simulation of the dynamics was pretty
cool, the project never included a good model of propellers. It also lacked any
piloting (controlling). Moreover, the interfaces to any guidance was poor.

# refactoring

As my career banked to systems engineering, I decided to reshape the project
with some reasonable increments aiming at bringing it to life again also for
python 3, populating the simulation with the complete GNC
(guidance-navigation-control) and sensing, and improving the clarity at any
interface level.

