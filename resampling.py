from asynchat import simple_producer

import numpy as np
import pyqtgraph as pg
from low_pass_filter import createPlot, create2Plot_2x, low_pass_filter2


def detimation(time: np.ndarray, signal: np.ndarray, N):
    print("time", time)
    print("signal", signal)
    delArr1 = []
    delArr2 = []
    for i in range(len(signal)):
        if i % N != 0:
            delArr1.append(i)
            delArr2.append(i)
    print(delArr1)
    newTime = np.delete(time, delArr1)
    newSignal = np.delete(signal, delArr2)
    print("new time: ", time)
    print("new signal: ", signal)
    return newTime, newSignal

def downsampling(time: list, signal: list, dividor):
    #createPlot(time, np.absolute(np.fft.fft(signal)), Name="old spectr")
    filtredSignal = low_pass_filter2(signal, len(signal) // (2 * dividor))
    #createPlot(time, np.absolute(np.fft.fft(filtredSignal)), Name = "new spectr")
    dsTime, dsSignal = detimation(time, filtredSignal, dividor)
    createPlot(dsTime, np.absolute(np.fft.fft(dsSignal)), Name = "ds spectr")
    return dsTime, dsSignal

def insertZeros(time, signal, N):
    #np.insert(a, [0, 4, 7], 77)
    insertArr = []
    for i in range(len(signal)):
        if (i % 2) == 0:
            insertArr.append(i)

    #print(signal)
    for i in range(N - 1):
        insArr = np.arange(1, len(signal) + 1, i + 1)
        signal = np.insert(signal, insArr, 0)
        #insArr = np.arange(0, len(signal), i)
        print(i, "now signal len = ", len(signal))
    #print(signal)
    dt = time[1] - time[0]
    newtime = np.arange(time[0], time[len(time) - 1] + dt, dt / (N - 0))

    createPlot(newtime, signal)
    return newtime, signal

def upsampling(time, signal, M):
    newTime, newSignal = insertZeros(time, signal, M)
    clearSignal = low_pass_filter2(newSignal, len(newSignal) // (2 * M))
    for i in range(len(clearSignal)):
        clearSignal[i] = clearSignal[i] * M
    return newTime, clearSignal

def changeSampling(time, signal, mult, div):
    newTime, newSignal = upsampling(time, signal, mult)
    return downsampling(newTime, newSignal, div)

step = 0.05
left = 0
right = 8 * np.pi


time = np.arange(left, right, step)
signal = np.sin(1 * time)

#newTime, newSignal = downsampling(time, signal, 30)
#newTime, newSignal = downsampling(time, signal, 5)
#newTime, newSignal = upsampling(time, signal, 10)

newTime, newSignal = changeSampling(time, signal, 5, 1)

print("len new time = ", len(newTime))
print("len new signal = ", len(newSignal))
#createPlot(time, signal)
#createPlot(newTime, newSignal)

create2Plot_2x(time, newTime, signal, newSignal)

arr = np.arange(0, 1, 0.1)
delArr = [0, 1, 2]
newArr = np.delete(arr, delArr)
print(arr)
print(newArr)


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()