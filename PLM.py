import csv
import math
from datetime import datetime
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
import numpy as np
from matplotlib.dates import date2num
import random
from matplotlib.pylab import hist, show
import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np


f = open('OEM Event PLM Diciembre.csv')
csv_f = csv.reader(f)

#with open('OEM Event Frenos Noviembre 2016.csv') as csvfile:
#    readCSV1 = csv.reader(csvfile, delimiter=',')

from time import strptime
strptime('Dec', '%b').tm_mon

pld = []
ff = []
cc = []
beacon = []
for row in csv_f:
    # date
    ff.append(row[6])
    # Tons value
    p = row[7]
    pld.append(p)
    # truck
    c = row[1]
    cc.append(c)
    # beacon (ID)
    bcn = row[9]
    beacon.append(bcn)

#for x in range(len(pld)):
#    if 200 < pld[x] < 600:


print type(pld[1])

ca = pd.DataFrame({'EquipmentName': cc}, index=None)
equipment = ca.EquipmentName.unique()

df = pd.DataFrame({'truck': cc[1:], 'payload': pld[1:]})
df.reset_index(inplace=True)

df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()

for i in equipment:
    if i != 'EquipmentName':
        df4 = df[df['truck'] == i]
        df4.reset_index(inplace=True)
        df2['truck'] = [i]

        payloadtruck = []
        for j in range(len(df4['payload'])):
            payloadtruck.append(float(df4['payload'][j]))
        df2['payloadmean'] = [(np.mean(payloadtruck))]

        df3 = pd.concat([df3, df2], ignore_index=True)

x = df3['payloadmean'].values.tolist()
y = df3['truck'].values.tolist()


plt.show()
objects = (y)
y_pos = np.arange(len(objects))
performance = x

plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Average payload')
plt.title('Truck')
plt.ylim([0,600])
plt.xlim([0,10])
plt.show()

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import plotly.plotly as py  # tools to communicate with Plotly's server

fig = plt.figure()

# example data
mu = 290 # mean of distribution
sigma = np.var(x) # standard deviation of distribution
#x = mu + sigma * np.random.randn(10000)

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()