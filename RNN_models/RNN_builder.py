from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open("RNN_1-Prong.root", "RECREATE")
factory = TMVA.Factory("TMVAClassification", output, '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=multiclass')




Signal_File = TFile.Open("../NewTTrees/signal_tree_1-Prong.root")
Background_File = TFile.Open("../NewTTrees/background_tree_1-Prong.root")

Signal_Tree = Signal_File.Get('signal_tree')
Background_Tree = Background_File.Get('background_tree;6')

dl_HLvars = TMVA.DataLoader('HL_vars')
dl_Track = TMVA.DataLoader('Tracks')
dl_Tower = TMVA.DataLoader('Towers')



HL_input = Input(shape=(13,))
Track_input = Input(shape=(10,))
Tower_input = Input(shape=(11,))


