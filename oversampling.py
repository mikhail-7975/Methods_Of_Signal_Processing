import numpy as np
print("0")
import pyqtgraph as pg
print("1")
from low_pass_filter import low_pass_filter2, createPlot
print("2")

n_data = 1000
cut_freq = 80
accuracy = 0.005


def oversampling_FFT(signal, n, m): #n - начальное m - конечное число отсчетов
    fft = np.fft.fft(signal)
    furie = np.zeros(m, dtype = complex)
    for i in range(n):
        furie[i] = fft[i]
    if(n%2) :
        half_n = int((n +1) / 2)
        for i in range (half_n+m-n+1,m ):
            furie[i] = furie[i-m+n]
        for i in range (half_n, half_n+m-n ):
            furie[i] = 0
    else:
        half_n = int(n/2)
        for i in range (half_n+m-n+2,m ):
            furie[i] = furie[i-m+n]
        furie[half_n+m-n+1] = furie[half_n+1]/2
        furie[half_n + 1] = furie[half_n+1]/2
        for i in range (half_n + 2, half_n+m-n ):
            furie[i] = 0
    res = np.fft.ifft(furie)
    res = res.real
    res = res * m/n
    return res

def decimation (signal, n, m): #n - начальное m - во сколько раз уменьшаем
    cut = int(n/m/2)
    data = low_pass_filter2(signal, cut)
    #data = low_freq_filter(signal, cut)
    res = np.zeros(int(n/m))
    j=0
    for i in range(0,n-m+1, m):
        res[j] = data[i]
        j+=1
    return res

def samplerating(signal, n, k=1, m=1): #n - начальное m - во сколько раз уменьшаем k - во сколько увеличиваем
    if(k/m>1):
        res = oversampling_FFT(signal, n, int(n*k/m))
        return res
    else:
        res = oversampling_FFT(signal, n, int(n*k))
        print(res.size)
        res = decimation(res, n*k, m)
        return res

x = np.random.rand(n_data)
for i in range (n_data):
    x[i] += 10*np.sin(i/50)+5*np.sin(i*2)



res = oversampling_FFT(x, n_data, n_data*2)
res2 = decimation(x, n_data, 3)
res3 = samplerating(x, n_data, 4, 1)



x_ = [i for i in range(n_data)]
x1 = [i for i in range(2 * n_data)]
x2 = [i for i in range(n_data // 3)]
x4 = [i for i in range(4 * n_data)]

createPlot(x_, x, Name = "original")
createPlot(x_, np.abs(np.fft.fft(x)), Name = "1000")
createPlot(x1, np.abs(np.fft.fft(res)), Name="2000")
createPlot(x2, np.abs(np.fft.fft(res2)), Name="333")
createPlot(x4, np.abs(np.fft.fft(res3)), Name="4000")


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()

'''
axes[0,0].plot(x)
axes[1,0].plot(np.abs(np.fft.fft(x)))


axes[0,1].plot(res)
axes[1,1].plot(np.abs(np.fft.fft(res)))

axes[0,2].plot(res2)
axes[1,2].plot(np.abs(np.fft.fft(res2)))

axes[0,3].plot(res3)
axes[1,3].plot(np.abs(np.fft.fft(res3)))

plt.show()


axis = np.linspace(0, n_data, num = res3.size)
plt.plot(axis, res3, 'bo')
plt.plot(x, 'ro')
plt.show()
'''
