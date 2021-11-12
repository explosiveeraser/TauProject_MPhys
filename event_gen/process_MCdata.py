import numpy as np
import ROOT
from array import array
import os, os.path

class process_data():
    @classmethod

    def read_tree_files(cls, dir_name):
        dir_loc = str(dir_name+"/")
        chain = ROOT.TChain("Delphes")
        f_name=str(dir_loc+dir_name+"_merged_trees.root")
        for f in os.listdir(dir_loc):
            chain.Add(str(dir_loc+f))
        new_file = ROOT.TFile.Open(f_name, "RECREATE")
        chain.Merge(new_file)
        return os.DirEntry.is_file(str(dir_loc+dir_name+"_merged_trees.root"))



