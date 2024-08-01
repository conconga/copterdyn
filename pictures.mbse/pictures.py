#!/usr/bin/python3

import argparse
import os
import re
import sys

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser( description="handler to keep management of documentation pictures organized" )
    parser.add_argument( "-u", "--used",  help="file with list of pictures used by the documentation", default=None )
    parser.add_argument( "-p", "--pics",  help="file with list of pictures in the current directory", default=None )
    parser.add_argument( "-n", "--nogit", help="file with list of pictures out of the versioning control", default=None )
    cli = parser.parse_args()
    #print(cli)

    for i in ( cli.used, cli.pics, cli.nogit ):
        if i is None:
            parser.print_help()
            sys.exit(-1)

        if not os.path.isfile(i):
            print("File '{:s}' not found. Aborting..".format(i))
            sys.exit(-1)

    # load lists:
    def fn_loadlist(filename):
        lst = []
        with open(filename) as f:
            while True:
                line = f.readline()
                if not line: break
                lst.append(line.rstrip())
        return lst

    lst_used  = fn_loadlist(cli.used)
    lst_pics  = fn_loadlist(cli.pics)
    lst_nogit = fn_loadlist(cli.nogit)

    #print("lst_used = {:s}".format( lst_used.__str__() ))
    #print("lst_pics = {:s}".format( lst_pics.__str__() ))
    #print("lst_nogit = {:s}".format( lst_nogit.__str__() ))

    # check whether all necessary files are available:
    print("** required files, but not available: **")
    lst_notAvailable = []
    for i in lst_used:
        if i not in lst_pics:
            print("A)  {:s}".format(i))
            lst_notAvailable.append(i)

    # check whether all necessary files are versioned:
    print("** required files not under versioning control: **")
    lst_necessaryButNotGit = []
    for i in lst_used:
        if i in lst_nogit:
            print("B)  {:s}".format(i))
            lst_necessaryButNotGit.append(i)

    # check whether folder has unnecessary files:
    print("** not required files: **")
    lst_notNecessary = []
    for i in lst_pics:
        if i not in lst_used:
            print("C)  {:s}".format(i))
            lst_notNecessary.append(i)

    # check whether folder has unnecessary files under versioning control:
    print("** not required files, but versioned: **")
    lst_notNecessaryInGit = []
    for i in lst_pics:
        if (i not in lst_used) and (i not in lst_nogit):
            print("D)  {:s}".format(i))
            lst_notNecessaryInGit.append(i)

    #print("lst_notAvailable = {:s}".format( lst_notAvailable.__str__()))
    #print("lst_necessaryButNotGit = {:s}".format( lst_necessaryButNotGit.__str__()))
    #print("lst_notNecessary = {:s}".format( lst_notNecessary.__str__()))
    #print("lst_notNecessaryInGit = {:s}".format( lst_notNecessaryInGit.__str__()))
