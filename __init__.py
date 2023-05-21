import sys
import numpy as np
import matplotlib.pyplot as plt
from srm import SpikeResponseModel

n_inputs = np.random.poisson(4)

SRM = SpikeResponseModel(n_inputs, 3, 0.01, 1, 0.01, 1, 1, 1)

input = []
for i in range(n_inputs):
    presynaptic = []
    for j in range(int(7/0.01)):
         p = np.random.binomial(n=1, p=0.003)
         if(p==1):
             presynaptic.append(j*0.01)
    input.append(presynaptic)

print(input)
  
timelist, voltage = SRM.simulate(7, input)
print(timelist)

time = []
for i in range(int(7/0.01)):
    t = i*0.01
    time.append(t)

plt.plot(time, voltage)
plt.title('Time Course of the Voltage u')
plt.xlabel('time/s')
plt.ylabel('voltage/V')
plt.show()