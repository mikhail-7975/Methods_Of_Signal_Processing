import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np


def classic_FT(func_array) -> list:
    N = len(func_array)
    result_func = [0.] * (N)
    for k in range(N):
        for n in range(N):
            result_func[k] += complex(func_array[n]) * np.exp(2 * np.pi * 1j / N * k * n)
    return result_func


def inv_classic_FT(func_array) -> list:
    N = len(func_array)
    result_func = [0.] * (N)
    for n in range(N):
        for k in range(N):
            result_func[n] += complex(func_array[k]) * np.exp(-2 * np.pi * 1j / N * k * n) / N
    print(result_func)
    return result_func

def hevisaid(x):
  return float(x > 0)

def shelve(x, a):
  return float(abs(x) < a)

def Gauss(x):
    return 1/(2 * np.pi)**(1/2) * np.exp(- x**2 / 2)

def noised_sin(x):
  y = [0.]*(len(x))
  #np.random.seed(1000)
  for i in range(len(x)):
    y[i] = np.random.uniform(-0.5, 0.5)
  y2 = np.sin(40 * x)
  return y + y2

def two_sin(x):
  y = [0.]*(len(x))
  #np.random.seed(1000)
  for i in range(len(x) // 2 ):
    y[i] = np.sin(x[i] / 2)

  for i in range(len(x) // 2, len(x)):
    y[i] = np.cos(20*x[i])
  return y

# define the data
theTitle = "classic FT"
'''
N = 314

x = np.arange(0, 2 * np.pi, 0.0001)#[i/10 for i in range(N)]

y = [0.] * len(x)

y2 = [np.sin(i / 2) + np.cos(20 * i) for i in x]
#y2 = [np.sin(30 * i) for i in x]
#y2 = [np.sin(0.6 * i) for i in x]
#y2 = [np.sin(i) + np.sin(3 * i) for i in x]
#y2 = [shelve(i, 0.1) for i in x]
#y2 = [Gauss(i) for i in x]
#y2 = noised_sin(x)
y = two_sin(x)

y2_spectr = classic_FT(y2)
y_spectr = classic_FT(y)
def createPlot(x: list, y: list, minX = 0, maxX = 10, minY = -5, maxY = 5, _pen = 'b'):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties
    plt.setLabel('left', 'Value', units='V')
    plt.setLabel('bottom', 'Time', units='s')
    plt.setXRange(minX, maxX)
    plt.setYRange(minY, maxY)
    plt.setWindowTitle('pyqtgraph plot')
    c1 = plt.plot(x, y, _pen = 'b', symbol='x', symbolPen='b', symbolBrush=0.2, name='red')

createPlot(x, y, minX= 0, maxX = 7, maxY=5, _pen='b')
createPlot(x, y2, minX= -1, maxX = 7, maxY=5, _pen='b')
y2_spectr_abs = [np.absolute(i) for i in y2_spectr]
y_spectr_abs = [np.absolute(i) for i in y_spectr]
createPlot(x, y2_spectr_abs, minX= 0, maxX = 0, maxY=60, _pen='r')
createPlot(x, y_spectr_abs, minX= 0, maxX = 0, maxY=60, _pen='r')


#y2_inv = inv_classic_FT(y2_spectr)
#y2_inv_abs = [np.float(i) for i in y2_inv]
#createPlot(x, y2_inv_abs, minX= -4, maxX = 4, maxY=5,_pen='g')

lib_spectr = np.fft.fft(y2)
lib_spectr_abs = [np.absolute(i) for i in lib_spectr]
cmp = [np.absolute(y2_spectr[i]) - np.absolute(lib_spectr[i]) for i in range(len(lib_spectr))]
print(y2_spectr_abs)
print(lib_spectr_abs)
print(cmp)
#createPlot(x, cmp, minX= -4, maxX = 0, minY= -0.0000000001, maxY=0.0000000001)
'''

'''
# create plot
plt = pg.plot()
plt.showGrid(x=True,y=True)
plt.addLegend()

# set properties
plt.setLabel('left', 'Value', units='V')
plt.setLabel('bottom', 'Time', units='s')
plt.setXRange(0,10)
plt.setYRange(-5,5)
plt.setWindowTitle('pyqtgraph plot')

# plot
c2 = plt.plot(x, y2, pen='r', symbol='o', symbolPen='r', symbolBrush=0.2, name='blue')
c1 = plt.plot(x, y, pen='b', symbol='x', symbolPen='b', symbolBrush=0.2, name='red')


## Start Qt event loop.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()

'''