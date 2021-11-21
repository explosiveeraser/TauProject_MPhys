import inspect
from inspect import getargspec
import ROOT
from collections import Iterable
import numba
import os, os.path

ROOT.gSystem.Load("install/lib/libDelphes")
ROOT.gSystem.Declare("using namespace RooFit;")

class Dataset:

    def __init__(self, name, title, categories, save_dir, dataset=None):
        self.name = name
        self.title = title
        self._categories = categories
