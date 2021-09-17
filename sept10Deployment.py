
import os
import numpy as np
import matplotlib.pyplot as plt
import pyproj
# need to point to your cmtb repo
from testbedutils import geoprocess
import scipy.io
import math
import datetime



def moving_average(a, n=21) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def smooth(x, window_len=11, window='hanning'):
    '''
    Smooth the data using a window with requested size.
    Adapted from:
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal

    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this:
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    '''

    if window_len < 3:  return x

    # if x.ndim != 1: raise (StandardError('smooth only accepts 1 dimension arrays.'))
    # if x.size < window_len:  raise (StandardError('Input vector needs to be bigger than window size.'))
    # win_type = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    # if window not in win_type: raise (StandardError('Window type is unknown'))

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    # saesha modify
    ds = y.shape[0] - x.shape[0]  # difference of shape
    dsb = ds // 2  # [almsot] half of the difference of shape for indexing at the begining
    dse = ds - dsb  # rest of the difference of shape for indexing at the end
    y = y[dsb:-dse]

    return y




def getAndroDrifter(drifterpath, filename, startTime, endTime):

    # Create the time vector based on the data inputs
    rawDate = [line.split(',')[-1].rstrip('\n') for line in open(os.path.join(drifterpath, filename))]
    rawDate = rawDate[2:]
    measTime = np.array([datetime.datetime.strptime(aa, '%Y-%m-%d %H:%M:%S:%f') for aa in rawDate])

    tindMin = np.argmin(np.abs(measTime - startTime))
    tindMax = np.argmin(np.abs(measTime - endTime))

    # load data
    data = np.genfromtxt(os.path.join(drifterpath, filename), delimiter=',', skip_header=2, usecols = range(21,27))
    # data = data[tindMin:tindMax, :]
    data = data[10:tindMax, :]

    # Crop time vector
    # measTime = measTime[tindMin:tindMax]
    measTime = measTime[10:tindMax]

    # lat = data[::100, 21]
    # lon = data[::100, 22]
    # accur = data[::100, 26]
    # measTime = measTime[::100]

    lat = data[::4, 0]
    lon = data[::4, 1]
    accur = data[::4, -1]
    measTime = measTime[::4]

    badvalues = np.nonzero(accur > 100)
    lat = np.delete(lat, badvalues)
    lon = np.delete(lon, badvalues)
    t = np.delete(measTime, badvalues)

    #myProj = pyproj.Proj("+proj=utm +zone=18S, +north +ellps=WGS84 +datum=WGS84 +untis=m +no_defs")
    ncSP = pyproj.Proj(init='epsg:3358')
    meshNCx, meshNCy = ncSP(lon, lat)

    FRF = geoprocess.FRFcoord(meshNCx, meshNCy)
    x = FRF['xFRF']
    y = FRF['yFRF']
    allSteps = t[-1]-t[0]
    intTime = []
    base = t[0]
    for i in range((allSteps.seconds)):
        intTime.append(base + datetime.timedelta(seconds=1 * i))

    intTime = np.asarray(intTime)

    tSeconds = np.asarray([(i - t[0]).total_seconds() for i in t])
    intSeconds = np.asarray([(i - intTime[0]).total_seconds() for i in intTime])

    xInt = np.interp(intSeconds,tSeconds,x)
    yInt = np.interp(intSeconds,tSeconds,y)

    x_moving = smooth(xInt, window_len=3, window='hanning')
    y_moving = smooth(yInt, window_len=3, window='hanning')

    v = np.zeros((len(x_moving),))
    vx = np.zeros((len(x_moving),))
    vy = np.zeros((len(x_moving),))

    for i in range(len(x_moving)-1):
        timediff = (intTime[i+1]-intTime[i]).total_seconds()
        v[i] = math.hypot(x_moving[i+1] - x_moving[i], y_moving[i+1] - y_moving[i])/timediff
        vx[i] = np.sign(x_moving[i+1] - x_moving[i]) * math.hypot(x_moving[i+1] - x_moving[i], 0)/timediff
        vy[i] = np.sign(y_moving[i+1] - y_moving[i]) * math.hypot(y_moving[i+1] - y_moving[i], 0)/timediff



    drifter = dict()
    drifter['lat'] = lat
    drifter['lon'] = lon
    drifter['t'] = intTime
    drifter['x'] = xInt
    drifter['y'] = yInt
    drifter['v'] = v
    drifter['vx'] = vx
    drifter['vy'] = vy
    drifter['tOrig'] = t
    drifter['xOrig'] = x
    drifter['yOrig'] = y

    return drifter





file_name = dict()
drifterpath = '/home/dylananderson/projects/drifterIO/data/Sept10'
files = os.listdir(drifterpath)
files.sort()
startTime = datetime.datetime(2021, 9, 8, 0, 0, 0)
endTime = datetime.datetime(2021, 9, 12, 0, 0, 0)

subset = files[0:5]

all = dict()
drifterlist = []

def addDrift(drifterlist, drifter, s, e):
    drift = {}
    drift['x'] = drifter['x'][s:e]
    drift['y'] = drifter['y'][s:e]
    drift['v'] = drifter['v'][s:e]
    drift['vx'] = drifter['vx'][s:e]
    drift['vy'] = drifter['vy'][s:e]
    drift['t'] = drifter['t'][s:e]
    drifterlist.append(drift.copy())
    return drifterlist



for i in range(len(subset)):

    if i == 0:
        print(subset[i])
        drifter = getAndroDrifter(drifterpath, subset[i], startTime, endTime)

        s = 0
        e = -1
        all['x'] = drifter['x'][s:e]
        all['y'] = drifter['y'][s:e]
        all['v'] = drifter['v'][s:e]
        all['vx'] = drifter['vx'][s:e]
        all['vy'] = drifter['vy'][s:e]
        all['t'] = drifter['t'][s:e]
        drifterlist = addDrift(drifterlist, drifter, s, e)

    elif i == 1 | i == 2:
        print('skipping these short files...')
        # drifter = getAndroDrifter(drifterpath, subset[i], startTime, endTime)
        # print(subset[i])


    elif i == 3:
        drifter = getAndroDrifter(drifterpath, subset[i], startTime, endTime)
        print(subset[i])
        s = 0
        e = -1

        all['x'] = np.append(all['x'], drifter['x'][s:e])
        all['y'] = np.append(all['y'], drifter['y'][s:e])
        all['v'] = np.append(all['v'], drifter['v'][s:e])
        all['vx'] = np.append(all['vx'], drifter['vx'][s:e])
        all['vy'] = np.append(all['vy'], drifter['vy'][s:e])
        all['t'] = np.append(all['t'], drifter['t'][s:e])
        drifterlist = addDrift(drifterlist, drifter, s, e)

    elif i == 4:
        drifter = getAndroDrifter(drifterpath, subset[i], startTime, endTime)
        print(subset[i])
        s = 0
        e = -1

        all['x'] = np.append(all['x'], drifter['x'][s:e])
        all['y'] = np.append(all['y'], drifter['y'][s:e])
        all['v'] = np.append(all['v'], drifter['v'][s:e])
        all['vx'] = np.append(all['vx'], drifter['vx'][s:e])
        all['vy'] = np.append(all['vy'], drifter['vy'][s:e])
        all['t'] = np.append(all['t'], drifter['t'][s:e])
        drifterlist = addDrift(drifterlist, drifter, s, e)




plt.figure(figsize=(14,6))
axDrift = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
im = scipy.io.loadmat('/home/dylananderson/projects/drifters/1572458401.Wed.Oct.30_18_00_01.GMT.2019.argus02b.cx.timex.merge')
ims = axDrift.imshow(im['Ip'], origin="upper",extent=(np.min(im['y']), np.max(im['y']), np.max(im['x']), np.min(im['x'])))
for i in drifterlist:
    sc = axDrift.scatter(i['y'], i['x'], s=10, c=i['v'], marker=u'.', vmin=0, vmax=2, edgecolor='None')
# sc = axDrift.scatter(drifterlist[-1]['y'], drifterlist[-1]['x'], s=10, c=drifterlist[-1]['v'], marker=u'.', vmin=0, vmax=2, edgecolor='None')
# axDrift.set_xlim([500, 750])
# axDrift.set_ylim([225,50])



