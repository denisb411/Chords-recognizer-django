import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('samples_nylonGuitar_1024_Mm_7.csv')#'samples_steelGuitar_8192_test.csv')
X = df.iloc[:,0:-1]
Y = df.iloc[:,-1]

XnpArray = np.array(X.iloc[:,:], dtype = np.float)

window = np.blackman(1024)

sample1 = XnpArray[0,:]
sample2 = XnpArray[30,:]

pltSample = sample2

# pltSample = sample2.reshape(-1,8).mean(axis=1)

print(pltSample)

pltSample = pltSample#*window
fftData=np.abs(np.fft.rfft(pltSample))

print fftData

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.cla()
ax1.plot(pltSample)
ax1.grid()

ax2.cla()
ax2.plot(fftData)
ax2.grid()


plt.show()