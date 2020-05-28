import numpy as np
import pyqtgraph as pg

def getNum(arr: list, i: int):
    if 0 <= i < len(arr) :
        return arr[i]
    else:
        return 0

def convolution_Fourier(xn1: list, xn2: list):
    _xn1 = np.concatenate((xn1, [0.] * len(xn2)), axis=None)
    _xn2 = np.concatenate((xn2, [0.] * len(xn1)), axis=None)
    f_xn1 = np.fft.fft(_xn1)#classic_FT(xn1)
    f_xn2 = np.fft.fft(_xn2)#classic_FT(xn2)
    res = [0.] * len(f_xn1)
    for i in range(len(f_xn1)):
        res[i] = f_xn1[i] * f_xn2[i]
    return np.fft.ifft(res)

def convolulution(xn1: list, xn2: list):
    N = len(xn1)
    M = len(xn2)
    result = [0.] * (N + M)
    for i in range(N + M):
        for j in range(i):
            if (i - j >= 0) and (i - j < N) and (j < M):
                result[i] += xn1[i - j] * xn2[j]
    return result

def crossCorelation(xn1: list, xn2: list):
    N = len(xn1)
    M = len(xn2)
    result = [0.] * (N + M)
    for i in range(N + M):
        for j in range(i):
            if (i - j >= 0) and (i - j < N) and (j < M):
                result[i] += xn1[i - j] * np.conj(xn2[j])
    return result


def shelve(x, a):
  return float(abs(x) <= a)

def impuls(x, a):
  if x < 0.:
      return 0.
  else:
      return np.exp(-x)

def noised_sin(x):
  y = [0.]*(len(x))
  #np.random.seed(1000)
  for i in range(len(x)):
    y[i] = np.random.uniform(-2, 2)
  y2 = np.sin(x)
  return y + y2


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

step = 0.05
left = -10 * np.pi
right = 10 * np.pi
left2 = -1*np.pi
right2 = 1*np.pi

x = np.arange(left, right + step / 2, step)
print("len x = ", len(x))
x2 = np.arange(left2, right2 + step / 2, step)
print("len x2 = ", len(x2))

#y1, y2 = [shelve(i, 1.01) for i in x], [shelve(i, 1.01) for i in x2]

#y1, y2 = [np.sin(i) for i in x], [np.sin(i) for i in x2]
#y2, y1 = [np.sin(i) for i in x2], noised_sin(x)

y1, y2 = [(impuls(i, 1.01) + np.random.uniform(-0.1, 0.1)) for i in x], [shelve(i, 1.01) for i in x2]
#y1, y2 = [(impuls(i, 1.01)) for i in x], [shelve(i, 1.01) for i in x2]





conv_sum = convolulution(y1, y2)

conv_F = convolution_Fourier(y1, y2)
conv_F = [np.real(conv_F[i]) for i in range(len(conv_F))]

conv_lib = np.convolve(y1, y2)
print(len(conv_lib))
print(len(conv_lib))





cmp = [0.] * len(conv_F)
for i in range(len(conv_F)):
    cmp[i] = conv_F[i] - conv_sum[i]

print(cmp)

croscor = crossCorelation(y1, y2)



createPlot(x, y1, -2, 2, Name="1st func")
createPlot(x2, y2, -2, 2, Name="2nd func")

x_conv = np.arange(0, len(conv_sum) * step, step)
x_conv2 = np.arange(0, len(conv_F) * step, step)

print("len x_conv ", len(x_conv))
print("len conv ", len(conv_sum))


print("len x_conv 2 ", len(x_conv2))
print("len conv 2 ", len(conv_F))

createPlot(x_conv, conv_sum, -4, 4, 0, 40, Name="conv")
createPlot(x_conv2, conv_F, -4, 4, 0, 40, Name ="conv Fourier")
createPlot(x_conv2, cmp, -4, 4, 0, 40, Name="cmp")
createPlot(x_conv, croscor, -4, 4, 0, 40, Name="crossCor")

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
