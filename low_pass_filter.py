import numpy as np
import pyqtgraph as pg
import scipy.signal

def createPlot(x: list, y: list, minX = 0, maxX = 10, minY = -5, maxY = 5, Name = "_"):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties
    plt.setLabel('left', 'Value', units='V')
    plt.setLabel('bottom', 'Time', units='s')
    plt.setXRange(minX, maxX)
    plt.setYRange(minY, maxY)
    plt.setWindowTitle('pyqtgraph plot')
    c1 = plt.plot(x, y, pen='b', symbol='x', symbolPen='b', symbolBrush=0.2, name = Name)

def create2Plot(x: list, y: list, y2: list, minX = 0, maxX = 10, minY = -5, maxY = 5, Name = "_"):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties
    plt.setLabel('left', 'Value', units='V')
    plt.setLabel('bottom', 'Time', units='s')
    plt.setXRange(minX, maxX)
    plt.setYRange(minY, maxY)
    plt.setWindowTitle('pyqtgraph plot')
    c1 = plt.plot(x, y, pen='b', symbol='x', symbolPen='b', symbolBrush=0.2, name = Name)
    c2 = plt.plot(x, y2, pen='r', symbol='x', symbolPen='r', symbolBrush=0.2, name=Name)

def create2Plot_2x(x1: list, x2: list, y: list, y2: list, minX = 0, maxX = 10, minY = -5, maxY = 5, Name1 = "old", Name2 = "new"):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties
    plt.setLabel('left', 'Value', units='V')
    plt.setLabel('bottom', 'Time', units='s')
    plt.setXRange(minX, maxX)
    plt.setYRange(minY, maxY)
    plt.setWindowTitle('pyqtgraph plot')
    c1 = plt.plot(x1, y, pen='b', symbol='x', symbolPen='b', symbolBrush=0.2, name = Name1)
    c2 = plt.plot(x2, y2, pen='r', symbol='o', symbolPen='r', symbolBrush=0.2, name=Name2)

def convolve(signal, filter):
    x1 = np.fft.fft(filter)
    x2 = np.fft.fft(signal)
    result = np.multiply(x1, x2)
    result = np.fft.ifft(result)
    return result

def gauss(x, a, sigma):
    return np.exp(-(x - a) ** 2 / (2 * sigma) ** 2)

def noised_sin(x):
  y = [0.]*(len(x))
  #np.random.seed(1000)
  for i in range(len(x)):
    y[i] = np.random.uniform(-1, 1)
  y2 = 1 * np.sin(x)
  return y + y2

def many_sin(x):
  y = np.zeros(len(x))
  y = np.sin(x) + np.sin(20 * x) / 10 + np.sin(80 * x) / 10 #+ np.sin(30 * x) + np.sin(40 * x) + np.sin(50 * x)
  #y = np.sin(10 * x) + np.sin(20 * x) + np.sin(30 * x)
  return y

def convolulution(xn1: list, xn2: list):
    N = len(xn1)
    M = len(xn2)
    result = [0.] * (N + M)
    for i in range(N + M):
        for j in range(i):
            if (i - j >= 0) and (i - j < N) and (j < M):
                result[i] += xn1[i - j] * xn2[j]
    return result

#==========================================================================================
def low_pass_filter2(data, max_freq):
    dataSize = len(data)
    x_ = [i for i in range(dataSize)]
    idle_arr = np.zeros(dataSize)
    for i in range(0, max_freq):
        idle_arr[i] = 1
        idle_arr[dataSize - 1 - i] = 1
    #create2Plot(x_, np.absolute(np.fft.fft(data)), idle_arr, minX=x_[0], maxX=60)
    afr_responce = np.fft.ifft(idle_arr)

    #create2Plot(x_, np.real(afr_responce), np.imag(afr_responce))
    count_zero_coef = 500
    for i in range(count_zero_coef, dataSize // 2):
        afr_responce[i] = 0
        afr_responce[dataSize - i - 1] = 0
    create2Plot(x_, np.absolute(np.fft.fft(data)), np.real(np.fft.fft(afr_responce)), minX=x_[0], maxX=60)
    res = convolve(data, afr_responce)
    result = []#np.real(res)
    for i in range(0, dataSize):
        result.append(np.real(res[i]))
    #create2Plot(x_, np.real(result), np.imag(result))
    print(len(result))
    print(len(data))
    return result

def high_pass_filter3(x, data, min_freq):
    clear = low_pass_filter2(x, data, min_freq)
    #createPlot(x, clear, Name="clear")
    result = []
    for i in range(len(clear)):
        print(data[i], ' ', clear[i], " = ", data[i] - clear[i])
        result.append(data[i] - clear[i])
    return result

def high_pass_filter2(data, min_freq):
    dataSize = len(data)
    x_ = [i for i in range(dataSize)]
    idle_arr = np.ones(dataSize)
    for i in range(0, min_freq):
        idle_arr[i] = 0
        idle_arr[dataSize - 1 - i] = 0
    #create2Plot(x_, np.absolute(np.fft.fft(data)), idle_arr, minX=x_[0], maxX=60)
    afr_responce = np.fft.ifft(idle_arr)
    count_zero_coef = 50
    for i in range(count_zero_coef, dataSize // 2):
        afr_responce[i] = 0
        afr_responce[dataSize - i - 1] = 0
    #createPlot(x_, np.real(afr_responce), Name="AFR zero")
    #create2Plot(x_, np.absolute(np.fft.fft(data)), np.real(np.fft.fft(afr_responce)), minX=x_[0], maxX=60)
    res = convolve(data, afr_responce)
    #res = np.convolve(data, afr_responce)
    result = []  # np.real(res)
    for i in range(0, dataSize):
        result.append(np.real(res[i]))
    print(len(result))
    print(len(data))
    return result

def low_pass_filter(x, data, max_freq):
    dataSize = len(data)
    idle_arr = np.zeros(dataSize)
    for i in range(0, max_freq):
        idle_arr[i] = 1
        idle_arr[dataSize - 1 - i] = 1
    createPlot(x, idle_arr, minX=x[0], maxX=x[len(x) - 1])
    spectr_data = np.fft.fft(data)
    create2Plot(x, np.real(spectr_data), np.imag(spectr_data))
    #createPlot(x, np.absolute(spectr_data), minX=x[0], maxX=x[len(x) - 1])
    res = np.multiply(spectr_data, idle_arr)
    #createPlot(x, np.absolute(res), minX=x[0], maxX=x[len(x) - 1])
    create2Plot(x, np.real(res), np.imag(res))
    res = np.fft.ifft(res)
    res = np.real(res)
    return res

def high_pass_filter(x, data, min_freq):
    dataSize = len(data)
    idle_arr = np.zeros(dataSize)
    for i in range(min_freq, dataSize // 2 + 1):
        idle_arr[i] = 1
        idle_arr[dataSize - 1 - i] = 1
    createPlot(x, np.absolute(idle_arr), minX=x[0], maxX=x[len(x) - 1], Name="idle arr")
    afr_responce = np.fft.ifft(idle_arr)
    afr_responce = np.real(afr_responce)
    createPlot(x, np.absolute(afr_responce), minX=x[0], maxX=x[len(x) - 1], Name="charact in time")
    for i in range(dataSize):
        if np.absolute(afr_responce[i]) < 0.005:
           afr_responce[i] = 0
    createPlot(x, np.real(afr_responce), minX=x[0], maxX=x[len(x) - 1], Name="charact in time not ideal")
    idle_arr = np.fft.fft(afr_responce)
    createPlot(x, np.absolute(idle_arr), minX=x[0], maxX=x[len(x) - 1], Name="new idle arr")
    spectr_data = np.fft.fft(data)
    #createPlot(x, np.absolute(spectr_data), minX=x[0], maxX=x[len(x) - 1])
    res = np.multiply(spectr_data, idle_arr)
    #createPlot(x, np.absolute(res), minX=x[0], maxX=x[len(x) - 1])
    res = np.fft.ifft(res)
    res = np.real(res)
    return res

def narrow_band_filter(x, data, min_freq, max_freq):
    dataSize = len(data)
    idle_arr = np.zeros(dataSize)
    for i in range(min_freq, max_freq):
        idle_arr[i] = 1
        idle_arr[dataSize - 1 - i] = 1
    #createPlot(x, idle_arr, minX=x[0], maxX=x[len(x) - 1])
    spectr_data = np.fft.fft(data)
    #createPlot(x, np.absolute(spectr_data), minX=x[0], maxX=x[len(x) - 1])
    res = np.multiply(spectr_data, idle_arr)
    #createPlot(x, np.absolute(res), minX=x[0], maxX=x[len(x) - 1])
    res = np.fft.ifft(res)
    res = np.real(res)
    return res

def cutting_filter(data, min_freq, max_freq):
    dataSize = len(data)
    idle_arr = np.ones(dataSize)
    for i in range(min_freq, max_freq):
        idle_arr[i] = 0
        idle_arr[dataSize - 1 - i] = 0
    #createPlot(x, idle_arr, minX=x[0], maxX=x[len(x) - 1])
    spectr_data = np.fft.fft(data)
    #createPlot(x, np.absolute(spectr_data), minX=x[0], maxX=x[len(x) - 1])
    res = np.multiply(spectr_data, idle_arr)
    #createPlot(x, np.absolute(res), minX=x[0], maxX=x[len(x) - 1])
    res = np.fft.ifft(res)
    res = np.real(res)
    return res

def narrow_band_filter2(x, data, Min_freq, Max_freq):
    low = low_pass_filter2(x, data, Min_freq)
    hight = high_pass_filter2(x, data, Max_freq)
    res = []
    for i in range(len(low)):
        res.append(low[i] + hight[i])
    return res

def cutting_filter2(x, data, Min_freq, Max_freq):
    low = low_pass_filter2(x, data, Max_freq)
    hight = high_pass_filter2(x, data, Min_freq)
    #low = low_pass_filter2(x, data, Min_freq)
    #hight = high_pass_filter2(x, data, Max_freq)
    res = data - low - hight
    return res
#==================================================================================================================
'''
x_left = -5 * np.pi
x_right = 5 * np.pi
step = 0.01

x = np.arange(x_left, x_right, step)
print(len(x))

#y = noised_sin(x)
y = many_sin(x)
y_spectr = np.fft.fft(y)
#y_low = low_pass_filter(x, y, 10)
#y_high = high_pass_filter(x, y, 10)
#y_narrow_band = narrow_band_filter(x, y, 25, 125)
#y_cutting = cutting_filter2(x, y, 25, 125)

#y_high2 = low_pass_filter2(x, y, 10)
y_high2 = high_pass_filter2(x, y, 60)
#y_high2 = cutting_filter2(x, y, Min_freq=60, Max_freq=130)
#y_high2 = narrow_band_filter2(x, y, Min_freq=60, Max_freq=130)

createPlot(x, y, Name="signal")
createPlot(x, np.absolute(y_spectr), Name="spectr")
createPlot(x, y_high2, Name="clear signal")
createPlot(x, np.absolute(np.fft.fft(y_high2)), Name="clear signal spectr")
#createPlot(x, np.absolute(y_spectr), Name="spectr")
#createPlot(x, y_low, Name="low pass")
#createPlot(x, y_high, Name="high pass")
#createPlot(x, y_narrow_band, Name="narrow band")
#createPlot(x, y_cutting, Name="cutting")

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
'''