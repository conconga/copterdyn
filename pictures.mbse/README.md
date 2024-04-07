# copterdyn

**copterdyn** is a sophisticated simulator for a quadcopter modeled using
multibody simulation concepts.

# history

The quadcopter simulator is outdated, and with its first version completed in 2017.
It was developed using Python 2.7, without any compatibility to any Python
3.0. This was critical in particular with debugging information and frame
generation (animations). Additionally, the first implementation had other issues.
For instance, while the simulation of the dynamics impressive, the project lacked a proper propeller model, and a
piloting (controlling) algorithm. Moreover, the interfaces to any guidance inadequate.

# refactoring

As my career banked to systems engineering, I made a decision to reshape the project
with some reasonable increments aiming at bringing it reviving the project also for
python 3, populating the simulation with the complete GNC
(guidance-navigation-control) and sensing, and enhancing the clarity at any
interface level.

# mbse

The next paragraphs and respective diagrams will provide the details about the new project, and what is
expected to have achieved.

## use case diagram

![Package_00_UseCases_Use_Cases.SVG](Use-Case diagram, and details on stakeholders (I myself!) needs.)
![Package_02_UseCases_UseCases.SVG](Use-Case diagram, and details on stakeholders (I myself!) needs.)
