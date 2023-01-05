import numpy as np
import matplotlib.pyplot as plt

data = []
cf = []
corr = []
window = [3,5,10,20]
fano = []
count = []

n = 0
f = open('bint.txt', 'r')
for line in f:
    data1 = []
    for row in line:
        if (row==' '):
            continue
        elif (row=='\n'):
            continue
        else:
            data1.append(float(row))
    data.append(data1)
    n = n+1
f.close()

sum = 0
for i in range(n-1):
    for j in range(i,n):
        coef = np.corrcoef(data[i], data[j], "same")
        print(np.shape(coef))
        coef = coef[0][1] 
        cf.append(coef)
        if ((float(coef))**2 > 0.64):
            cf2 = []
            cf2 = np.correlate(data[i],data[j])
            corr.append(cf2)
            sum = sum+1
            count.append(sum)

plt.hist(cf, bins=200, range=(-1,1))
plt.title('Correlation coefficient distribution')
plt.show()

plt.plot(count, corr)
plt.title('Cross correlation curve')
plt.show()

for i in range(4):
    mean = np.mean(data[0:159][int(window[i])])
    std = np.std(data[0:159][int(window[i])])
    r = mean/std
    fano.append(r)

plt.scatter(window, fano)
plt.xlabel('Time window')
plt.ylabel('Fano factor')
plt.title('Fano factor in some time windows')
plt.show()