import numpy as np
import ROOT
from array import array
from ROOT import gROOT
import numba
from numba import jit, jit_module
from event_gen.process_MCdata import *

#ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

background_dir = "Delphes_Background"
signal_dir = "Delphes_Signal"

back_chain = read_tree_files(background_dir)
sig_chain = read_tree_files(signal_dir)

back_df = ROOT.RDataFrame(back_chain)
sig_df = ROOT.RDataFrame(sig_chain)

back_df = back_df.Define(str("Jet_Pntrs"), "Numba::find_tagged_jets(Jet.TauTag)")
back_df = back_df.Define(str("GenJet_Pntrs"), "Numba::find_tagged_jets(GenJet.TauTag)")

ROOT.gInterpreter.Declare("""
bool print_tref(ROOT::VecOps::RVec<TRefArray> x) {
    int size = x.size();
    for (int i = 0; i < size; i++) {
        int size_ = x[i].GetLast();
        for (int r = 0; r < size_; r++) {
            cout << x[i].At(r) << endl;
        }
    }
    return true;
}
""")




column_names = back_df.GetColumnNames()
columns = {}


for col in column_names:
    try:
        columns[str(col)] = str(back_df.GetColumnType(col))
        print("{}||{}".format(col, columns[col]))
    except:
        pass

back_hists = {}
sig_hists = {}

ROOT.gInterpreter.Declare('''
    TObjArray printb(ROOT::RDF::RNode df) {
        TObjArray pol = TObjArray(100);
        auto std = df.Take<ROOT::VecOps::RVec<TRefArray>>("Jet.Constituents");
        for (ROOT::VecOps::RVec<TRefArray> r : std) {
            int s_ = r.size();
            for (int i = 0; i < 1; i++) {
                int r_ = r[i].GetLast();
                for (int h = 0; h < r_; h++) {
                     if (r[i].GetLast() > 5) {
                         pol.Add(r[i].At(h));
                     }
                }
            }
        }
        return pol;
    }
''')

pol = ROOT.printb(ROOT.RDF.AsRNode(back_df))
print(pol)

for i in range(0, pol.GetLast()):
    pol.At(i).Dump()
    try:
        pol.At(i).Dump()
    except:
        pass

for co in column_names:
    col = str(co)
    try:
        if "Float_t" in columns[col] or "Double_t" in columns[col] or "Int_t" in columns[col]:
            if "Jet." in col and col[0] == "J":
                back_hists[col] = back_df.Histo1D(get_hist(str("Background "+col), float, 250), col)
                sig_hists[col] = sig_df.Histo1D(get_hist(str("tau "+col), float, 250), col)
            elif "GenJet." in col and col[0] == "G":
                back_hists[col] = back_df.Histo1D(get_hist(str("Background " + col), float, 250), col)
                sig_hists[col] = sig_df.Histo1D(get_hist(str("tau " + col), float, 250), col)
        else:
            pass
    except:
        pass



print(column_names)

#input("blob")
