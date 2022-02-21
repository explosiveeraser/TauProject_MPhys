import inspect
from inspect import getargspec
import ROOT
from collections import Iterable
import numba
import os, os.path

class DataAnalysis(ROOT.TChain):

    def __init__(self, directory, maxEvents=0):
        ROOT.TChain.__init__(self, "Delphes", "Delphes")
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        for f in os.listdir(directory):
            self.AddFile(directory + f)
        self.BuildIndex("Event[0].Number")
        self.SetBranchStatus("*", 0)
        self._eventCounts = 0
        self._maxEvents = maxEvents
        ##
        self._weightCache = {}
        self._weightEngines = {}
        ##
        self._collections = {}
        self._branches = dict((b, False) for b in map(lambda  b:b.GetName(), self.GetListOfBranches()))
        ##
        self._producers = {}
        ##
        self.__dict__["vardict"] = {}

    def addWeight(self, name, weightClass):
        if name in self._weightEngines:
            raise KeyError("%s weight engine is already declared" % name)
        self._weightEngines[name] = weightClass
        self._weightCache.clear()

    def delWeight(self, name):
        del self._weightEngines[name]
        self._weightCache.clear()

    def weight(self, weightList=None, **kwargs):
        if weightList is None:
            weightList=self._weightEngines.keys()
        kwargs["weightList"] = weightList
        myhash = self._dicthash(kwargs)
        if not myhash in self._weightCache:
            w=1
            for weightElement in weightList:
                engine = self._weightEngines[weightElement]
                engineArgs = inspect.getfullargspec(engine.weight).args
                subargs = dict((k, v) for k, v in kwargs.iteritems() if k in engineArgs)
                w *= self._weightCache.setdefault("weightElement:%s # %s" %(weightElement, self._dicthash(subargs)), engine.weight(self, **subargs))
            self._weightCache[myhash] = w
        return self._weightCache[myhash]

    def addCollection(self, name, inputTag):
        print(name)
        if name in self._collections:
            raise KeyError("%r collection is already declared", name)
        if name in self._producers:
            raise KeyError("%r is already declared as a producer", name)
        if hasattr(self, name):
            raise AttributeError("%r object already has attribute %r" %(type(self).__name__, name))
        if inputTag not in self._branches:
            raise AttributeError("%r object has no branch %r" % (type(self).__name__, inputTag))
        self._collections[name] = inputTag
        self.SetBranchStatus(inputTag+"*", 1)
        self._branches[inputTag] = True

    def removeCollection(self, name):
        self.SetBranchStatus(self._collections[name]+"*", 0)
        self._branches[self._collections[name]] = False
        del self._collections[name]
        if name in self.vardict:
            delattr(self, name)

    def getCollection(self, name):
        if not name in self._collections:
            raise AttributeError("%r object has no attribute %r" %(type(self).__name__, name))
        if not name in self.vardict:
            print(self._collections[name])
            self.vardict[name] = ROOT.TChain.getattr(self, self._collections[name])
        return getattr(self, name)

    def addProducer(self, name, producer, **kwargs):
        if name in self._producers:
            raise KeyError("%r producer is already declared", name)
        if name in self._collections:
            raise KeyError("%r is already declared as a collection", name)
        if hasattr(self, name):
            raise AttributeError("%r object already has attribute %r" % (type(self).__name__, name))
        if "name" in kwargs: del kwargs["name"]
        if "producer" in kwargs: del kwargs["producer"]
        self._producers[name] = (producer, kwargs)

    def removeProducer(self, name):
        """Forget about the producer.
           This method will delete both the product from the cache (if any) and the producer.
           To simply clear the cache, use "del event.name" instead."""
        del self._producers[name]
        if name in self.vardict:
            delattr(self, name)

    def event(self):
        """Event number"""
        if self._branches["Event"]:
            return self.Event.At(0).Number
        else:
            return 0

    def to(self, event):
        """Jump to some event"""
        self.GetEntryWithIndex(event)

    def __getitem__(self, index):
        """Jump to some event"""
        self.GetEntryWithIndex(index)
        return self

    def __iter__(self):
        """Iterator"""
        self._eventCounts = 0
        while self.GetEntry(self._eventCounts):
            self.vardict.clear()
            self._weightCache.clear()
            yield self
            self._eventCounts += 1
            if self._maxEvents > 0 and self._eventCounts >= self._maxEvents:
                break

    def __getattr__(self, attr):
        """Overloaded getter to handle properly:
             - volatile analysis objects
             - event collections
             - data producers"""
        print(attr)
        if attr in self.__dict__["vardict"]:
            return self.vardict[attr]
        if attr in self._collections:
            return self.vardict.setdefault(attr, ROOT.TChain.getattr(self, self._collections[attr]))
        if attr in self._producers:
            return self.vardict.setdefault(attr, self._producers[attr][0](self, **self._producers[attr][1]))
        return ROOT.TChain.getattr(self, attr)

    def __setattr__(self, name, value):
        """Overloaded setter that puts any new attribute in the volatile dict."""
        if name in self.__dict__ or not "vardict" in self.__dict__ or name[0] == '_':
            self.__dict__[name] = value
        else:
            if name in self._collections or name in self._producers:
                raise AttributeError(
                    "%r object %r attribute is read-only (event collection)" % (type(self).__name__, name))
            self.vardict[name] = value

    def __delattr__(self, name):
        """Overloaded del method to handle the volatile internal dictionary."""
        if name == "vardict":
            raise AttributeError("%r object has no attribute %r" % (type(self).__name__, name))
        if name in self.__dict__:
            del self.__dict__[name]
        elif name in self.vardict:
            del self.vardict[name]
        else:
            raise AttributeError("%r object has no attribute %r" % (type(self).__name__, name))

    def _dicthash(self, dict):
        return (lambda d, j='=', s=';': s.join([j.join((str(k), str(v))) for k, v in d.iteritems()]))(dict)

    def decayTree(self, genparticles):
        db = ROOT.TDatabasePDG()
        theString = ""
        for part in genparticles:
            if part.M1 == -1 and part.M2 == -1:
                theString += part.printDecay(db, genparticles)
        return theString