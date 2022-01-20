import numpy as np
from ROOT import TMVA, TFile, TTree, TCut
import ROOT
from subprocess import call
from os.path import isfile


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open("RNN_1-Prong.root", "RECREATE")
factory = TMVA.Factory("TMVAClassification", output, '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=multiclass')




Signal_File = TFile.Open("../NewTTrees/signal_tree_1-Prong.root")
Background_File = TFile.Open("../NewTTrees/background_tree_1-Prong.root")

Signal_Tree = Signal_File.Get('signal_tree')
Background_Tree = Background_File.Get('background_tree;6')

SignalDF = ROOT.RDataFrame(Signal_Tree)
BackgroundDF = ROOT.RDataFrame(Background_Tree)

print(SignalDF.GetColumnNames())

#Do numpy approach for time being
signalArray = SignalDF.AsNumpy()
backArray = BackgroundDF.AsNumpy()

SortedsignalArray = np.array([])
SortedbackArray = np.array([])

print(signalArray)




#HL Layers
HL_input = Input(shape=(13,))
HLdense1 = Dense(128, activation='relu')(HL_input)
HLdense2 = Dense(128, activation='relu')(HLdense1)
HLdense3 = Dense(16, activation='relu')(HLdense2)

#Track Layers
Track_input1 = Input(shape=(10,))
Track_input2 = Input(shape=(10,))

trackDense1 = Dense(32, activation='relu')(Track_input1)
trackDense2 = Dense(32, activation='relu')(Track_input2)

mergeTrack = Concatenate()([trackDense1, trackDense2])

trackLSTM1 = LSTM(32, input_shape=(10,10))(mergeTrack)
trackLSTM2 = LSTM(32, input_shape=(10,10))(trackLSTM1)

#Tower Layers
Tower_input1 = Input(shape=(11,))
Tower_input2 = Input(shape=(11,))

towerDense1 = Dense(32, activation='relu')(Tower_input1)
towerDense2 = Dense(32, activation='relu')(Tower_input2)

mergeTower = Concatenate()([towerDense1, towerDense2])

towerLSTM1 = LSTM(24, input_shape=(6,11))(mergeTower)
towerLSTM2 = LSTM(24, input_shape=(6,11))(towerLSTM1)

#Layers Merged
mergedLayer = Concatenate([trackLSTM2, towerLSTM2, HLdense3])
fullDense1 = Dense(64, activation='relu')(mergedLayer)
fullDense2 = Dense(32, activation='relu')(fullDense1)
Output = Dense(1, activation='sigmoid')(fullDense2)

RNNmodel = Model(inputs=[Track_input1, Track_input2, Tower_input1, Tower_input2, HL_input], outputs=Output)
RNNmodel.save('tauRNN_1-prong.h5')
RNNmodel.summary()


