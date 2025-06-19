import numpy as np


def sexa2deci(hh=0, mm=0, ss=0):
    return hh+mm/60.+ss/3600.

class EGM2008():
    def __init__(self, fn):
        self.egm=np.load(fn)
        self.lonmin=self.egm[0,0,1]
        self.latmin=self.egm[0,0,0]
        self.d=self.egm[0,1,1]-self.egm[0,0,1]
#        print(self.lonmin, self.latmin, self.d)

    def interpolate(self, lat, lon):
        x = np.round((lon-self.lonmin)/self.d).astype(np.int32)
        y = np.round((lat-self.latmin)/self.d).astype(np.int32)
#        print(x, y)
        return self.egm[y, x, 2]
