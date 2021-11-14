import numpy as np
import ROOT
from array import array
import os, os.path
import numba

class process_data():
    @classmethod

    def read_tree_files(cls, dir_name):
        dir_loc = str(dir_name+"/")
        chain = ROOT.TChain("Delphes")
        for f in os.listdir(dir_loc):
            chain.Add(str(dir_loc+f))

        return chain

    @ROOT.Numba.Declare(['ROOT.RDataFrame'])
    def get_constituents(self, df):
        pass



