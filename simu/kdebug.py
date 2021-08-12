#==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==#
# author:      Luciano Augusto Kruk
# website:     www.kruk.eng.br
#
# description: Some debug functions.
#==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==#
import inspect
import numpy     as np
import sys
import traceback
#==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==#

class CKDEBUG:
    def __init__(self, maxlevel=0):
        self.maxlevel = maxlevel

    def show_matrix_details(self, name, D, level=10, linenb=0, fname=""):
        if (level <= self.maxlevel):
            print("---------------------------")
            if (linenb > 0):
                print(": (at line {:d})".format(linenb))

            if (len(fname) > 0):
                print(": ({:s})".format(fname))

            print(": {:s} <{:s}> =".format(name, type(D).__name__))
            print(D)

            if isinstance(D, np.ndarray):
                print(": shape =")
                print(D.shape)

    def linenb(self):
        """Returns the current line number in our program."""
        return inspect.currentframe().f_back.f_lineno

    def show_traceback(self):
        print(traceback.print_stack())

    def abort(self):
        print("============================")
        print("CKDEBUG:   aborting... ")
        print("============================")
        sys.exit(-1)

    def assert_infnan(self, np_array, txt):
        if (np.any(np.isnan(np_array)) or np.any(np.isinf(np_array))):
            print()
            self.show_matrix_details("array with nan ({:s}) =".format(txt), np_array, level=0)
            print()
            self.show_traceback()
            self.abort();

#==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==#

if (__name__ == "__main__"):
    kd = CKDEBUG(True)
    kd.show_matrix_details("zeros", np.zeros((2,2)))
    print("current line = {:d}".format(kd.linenb()))

    kd = CKDEBUG(False)
    kd.show_matrix_details("zeros", np.zeros((2,2)))
    print("current line = {:d}".format(kd.linenb()))

    kd = CKDEBUG(True)
    kd.assert_infnan(0, "line 1")
    kd.assert_infnan(np.nan + np.zeros((3,3)), "line 2")

    # it is not supposed to get here...
    print("ops...")

#==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==#
