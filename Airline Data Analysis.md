
After skimming through the dataset, below questions came in into the prominense.

1) Which flights departured delayed, ontime or before time delays
2) Which are the best airports in terms of on time departures ?
3) What is the average speed of all the aircrafts which flown from 3 origins to several destinations?
4) What is the frequnecy of those speed ? 
5) Which speed one flight should keep if it wants to arrive on time ? 
6) What are the maximum number flights flown to a particular destination ? 

The 1st step is to import all the required libraries in python.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
```
A new dataframe named **"airline"** has been created to hold all the values of the csv files.

```python
airline = pd.read_csv('flight_data.csv')
airline.shape
    (336776, 19)
```

**airline_null** is the new dataframe been created with a view to contain all the null values of the **airline** dataframe.

```python
airline_null = pd.DataFrame((airline.isnull().sum()),columns=['Null_Values'])
airline_null['%ofNullValeues'] = ((airline_null['Null_Values'])/336776*100).sort_values(ascending=True)
airline_null
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Null_Values</th>
      <th>%ofNullValeues</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>year</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>day</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>dep_time</th>
      <td>8255</td>
      <td>2.451184</td>
    </tr>
    <tr>
      <th>sched_dep_time</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>dep_delay</th>
      <td>8255</td>
      <td>2.451184</td>
    </tr>
    <tr>
      <th>arr_time</th>
      <td>8713</td>
      <td>2.587180</td>
    </tr>
    <tr>
      <th>sched_arr_time</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>arr_delay</th>
      <td>9430</td>
      <td>2.800081</td>
    </tr>
    <tr>
      <th>carrier</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>flight</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tailnum</th>
      <td>2512</td>
      <td>0.745896</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>dest</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>air_time</th>
      <td>9430</td>
      <td>2.800081</td>
    </tr>
    <tr>
      <th>distance</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>minute</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>time_hour</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
airline['dep_delay'].fillna(value=airline['dep_delay'].mean(),axis=0,inplace=True)
airline['air_time'].fillna(value=airline['air_time'].mean(),axis=0,inplace=True)
airline['air_time'].isnull().sum(),airline['distance'].isnull().sum()
airline['Velocity_Miles_Minutes'] = airline['distance']/airline['air_time']
airline.isnull().sum()
    year                         0
    month                        0
    day                          0
    dep_time                  8255
    sched_dep_time               0
    dep_delay                    0
    arr_time                  8713
    sched_arr_time               0
    arr_delay                 9430
    carrier                      0
    flight                       0
    tailnum                   2512
    origin                       0
    dest                         0
    air_time                     0
    distance                     0
    hour                         0
    minute                       0
    time_hour                    0
    Velocity_Miles_Minutes       0
    dtype: int64
```

A new dataframe called **airline_no_delays** has been made in order to hold all the rows related to with airlines with no delays.

```python
airline_no_delays = airline[airline['dep_delay'] == 0]
airline_no_delays.isnull().sum()

    year                       0
    month                      0
    day                        0
    dep_time                   0
    sched_dep_time             0
    dep_delay                  0
    arr_time                  16
    sched_arr_time             0
    arr_delay                 48
    carrier                    0
    flight                     0
    tailnum                    0
    origin                     0
    dest                       0
    air_time                   0
    distance                   0
    hour                       0
    minute                     0
    time_hour                  0
    Velocity_Miles_Minutes     0
    dtype: int64
```



```python
airline_no_delays['arr_time'].fillna(value=airline_no_delays['arr_time'].mean(),axis=0,inplace=True)
airline['origin_to_dest'] = airline['origin'] +'_to_'+ airline['dest']
```


```python
from matplotlib.pyplot import show
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(6, 6))
total=airline_no_delays.shape[0]
ax = sns.countplot(x='origin', data=airline_no_delays)
for p in ax.patches:
    height = p.get_height()
    ax.text((p.get_x() + p.get_width()/2),
           height + 70,
           '{:1.2f}'.format(height/total),
            ha="center")
show()
```
![png](output_10_0.png)


```python
airline_no_delays['origin'].value_counts()
```
    JFK    6239
    EWR    5585
    LGA    4690
    Name: origin, dtype: int64


```python
airline_no_delays['origin_to_dest'] = airline_no_delays['origin'] +'_to_'+ airline_no_delays['dest']
```



```python
airline_no_delays_JFK = airline_no_delays[airline_no_delays['origin'] == 'JFK']
airline_no_delays_EWR = airline_no_delays[airline_no_delays['origin'] == 'EWR']
airline_no_delays_LGA = airline_no_delays[airline_no_delays['origin'] == 'LGA']
airline_no_delays_JFK.shape, airline_no_delays_EWR.shape, airline_no_delays_LGA.shape
    ((6239, 21), (5585, 21), (4690, 21))
```


```python
from matplotlib.pyplot import show
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(50,8))
total=airline_no_delays_JFK.shape[0]
ax = sns.countplot(x='dest', data=airline_no_delays_JFK)
ax.set_xlabel('Destination')
ax.set_ylabel('Number of Flights in Percentage ')
ax.set_title('Number of Flights towards differnet destination from JFK')
#ticks = ax.set_xticks(airline_no_delays_JFK['dest'])
#labels = ax.set_xticklabels([airline_no_delays_JFK['dest'].values], rotation=30,fontsize='small')
for p in ax.patches:
    height = p.get_height()
    ax.text((p.get_x() + p.get_width()/2),
           height + 10,
           '{:1.3f}'.format(height/total),
            ha="center")
show()

```
![png](output_16_0.png)



```python
from matplotlib.pyplot import show
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(50,8))
total=airline_no_delays_EWR.shape[0]
ax = sns.countplot(x='dest', data=airline_no_delays_EWR)
ax.set_xlabel('Destination')
ax.set_ylabel('Number of Flights in Percentage ')
ax.set_title('Number of Flights towards differnet destination from EWR')
#ticks = ax.set_xticks(airline_no_delays_JFK['dest'])
#labels = ax.set_xticklabels([airline_no_delays_JFK['dest'].values], rotation=30,fontsize='small')
for p in ax.patches:
    height = p.get_height()
    ax.text((p.get_x() + p.get_width()/2),
           height + 10,
           '{:1.3f}'.format(height/total),
            ha="center")
show()
```

![png](output_17_0.png)



```python
from matplotlib.pyplot import show
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(50,8))
total=airline_no_delays_LGA.shape[0]
ax = sns.countplot(x='dest', data=airline_no_delays_LGA)
ax.set_xlabel('Destination')
ax.set_ylabel('Number of Flights in Percentage ')
ax.set_title('Number of Flights towards differnet destination from LGA')
#ticks = ax.set_xticks(airline_no_delays_JFK['dest'])
#labels = ax.set_xticklabels([airline_no_delays_JFK['dest'].values], rotation=30,fontsize='small')
for p in ax.patches:
    height = p.get_height()
    ax.text((p.get_x() + p.get_width()/2),
           height + 10,
           '{:1.3f}'.format(height/total),
            ha="center")
show()
```
![png](output_18_0.png)



```python
import seaborn as sns
plt.figure(figsize=(15,4))
%timeit
sns.distplot(airline_no_delays_LGA['Velocity_Miles_Minutes'],hist=True,rug=True)

plt.figure(figsize=(15, 4))
sns.boxplot(airline_no_delays_LGA['Velocity_Miles_Minutes'],color='orchid')
```

![png](output_19_1.png)

![png](output_19_2.png)



```python
import seaborn as sns
plt.figure(figsize=(15, 4))
%timeit
sns.distplot(airline_no_delays_JFK['Velocity_Miles_Minutes'],hist=True,rug=True)

plt.figure(figsize=(15, 4))
sns.boxplot(airline_no_delays_JFK['Velocity_Miles_Minutes'],color='orchid')
```

![png](output_20_1.png)

![png](output_20_2.png)



```python
import seaborn as sns
sns.set_style('whitegrid')
plt.figure(figsize=(15, 4))
%timeit
sns.distplot(airline_no_delays_EWR['Velocity_Miles_Minutes'],hist=True,rug=True)

plt.figure(figsize=(15, 4))
sns.boxplot(airline_no_delays_EWR['Velocity_Miles_Minutes'],color='orchid')
```
![png](output_21_1.png)

![png](output_21_2.png)



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = airline_no_delays_EWR['Velocity_Miles_Minutes'].plot(kind='kde')
ax2 = airline_no_delays_JFK['Velocity_Miles_Minutes'].plot(kind='kde')
ax3 = airline_no_delays_LGA['Velocity_Miles_Minutes'].plot(kind='kde')

ax1.set_xlim([3,10])
ax2.set_xlim([3,10])
ax3.set_xlim([3,10])

 # plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution from different Origin points for no delays")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```
![png](output_22_0.png)



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = sns.kdeplot(airline_no_delays_EWR['Velocity_Miles_Minutes'], shade = True)
ax2 = sns.kdeplot(airline_no_delays_JFK['Velocity_Miles_Minutes'], shade = True)
ax3 = sns.kdeplot(airline_no_delays_LGA['Velocity_Miles_Minutes'], shade = True)

ax1.set_xlim([3,10])
ax2.set_xlim([3,10])
ax3.set_xlim([3,10])

# plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution from different Origin points for no delays ")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```
![png](output_23_0.png)

A new dataframe named **airline_early_departures** has been created and later on tho


```python
airline_early_departures = airline[airline['dep_delay'] < 0]
airline_early_departures['Velocity_Miles_Minutes'] = airline_early_departures['distance']/airline_early_departures['air_time']
airline_early_departures['origin_to_dest'] = airline_early_departures['origin'] +'_to_'+ airline_early_departures['dest']
airline_early_departures_JFK = airline_early_departures[airline_early_departures['origin'] == 'JFK']
airline_early_departures_EWR = airline_early_departures[airline_early_departures['origin'] == 'EWR']
airline_early_departures_LGA = airline_early_departures[airline_early_departures['origin'] == 'LGA']
airline_early_departures_JFK.shape, airline_early_departures_EWR.shape, airline_early_departures_LGA.shape

    ((61146, 21), (59300, 21), (63129, 21))
```



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = airline_early_departures_EWR['Velocity_Miles_Minutes'].plot(kind='kde')
ax2 = airline_early_departures_JFK['Velocity_Miles_Minutes'].plot(kind='kde')
ax3 = airline_early_departures_LGA['Velocity_Miles_Minutes'].plot(kind='kde')

ax1.set_xlim([1,10])
ax2.set_xlim([1,10])
ax3.set_xlim([1,10])

 # plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution for early departure flights from different Origin points")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```


![png](output_25_0.png)



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = sns.kdeplot(airline_early_departures_EWR['Velocity_Miles_Minutes'], shade = True)
ax2 = sns.kdeplot(airline_early_departures_JFK['Velocity_Miles_Minutes'], shade = True)
ax3 = sns.kdeplot(airline_early_departures_LGA['Velocity_Miles_Minutes'], shade = True)

ax1.set_xlim([1,10])
ax2.set_xlim([1,10])
ax3.set_xlim([1,10])

# plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution for early departure flights from different Origin points")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```


![png](output_26_0.png)



```python
airline_delayed_departures = airline[airline['dep_delay'] > 0]
airline_delayed_departures['Velocity_Miles_Minutes'] = airline_delayed_departures['distance']/airline_delayed_departures['air_time']
airline_delayed_departures['origin_to_dest'] = airline_delayed_departures['origin'] +'_to_'+ airline_delayed_departures['dest']
airline_delayed_departures_JFK = airline_delayed_departures[airline_delayed_departures['origin'] == 'JFK']
airline_delayed_departures_EWR = airline_delayed_departures[airline_delayed_departures['origin'] == 'EWR']
airline_delayed_departures_LGA = airline_delayed_departures[airline_delayed_departures['origin'] == 'LGA']

airline_delayed_departures_JFK.shape, airline_delayed_departures_EWR.shape, airline_delayed_departures_LGA.shape

    ((43894, 21), (55950, 21), (36843, 21))
```



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = airline_delayed_departures_EWR['Velocity_Miles_Minutes'].plot(kind='kde')
ax2 = airline_delayed_departures_JFK['Velocity_Miles_Minutes'].plot(kind='kde')
ax3 = airline_delayed_departures_LGA['Velocity_Miles_Minutes'].plot(kind='kde')

ax1.set_xlim([1,10])
ax2.set_xlim([1,10])
ax3.set_xlim([1,10])

 # plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution for delayed flights from different Origin points")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```
![png](output_29_0.png)



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = sns.kdeplot(airline_delayed_departures_EWR['Velocity_Miles_Minutes'], shade = True)
ax2 = sns.kdeplot(airline_delayed_departures_JFK['Velocity_Miles_Minutes'], shade = True)
ax3 = sns.kdeplot(airline_delayed_departures_LGA['Velocity_Miles_Minutes'], shade = True)

ax1.set_xlim([1,10])
ax2.set_xlim([1,10])
ax3.set_xlim([1,10])

# plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution for delayed flights from different Origin points")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```
![png](output_30_0.png)

```python
airline_delayed_departures['origin'].value_counts()

    EWR    55950
    JFK    43894
    LGA    36843
    Name: origin, dtype: int64
```



```python
from matplotlib.pyplot import show
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(6, 6))
total=airline_delayed_departures.shape[0]
ax = sns.countplot(x='origin', data=airline_delayed_departures)
for p in ax.patches:
    height = p.get_height()
    ax.text((p.get_x() + p.get_width()/2),
           height + 100,
           '{:1.2f}'.format(height/total),
            ha="center")
show()
```
![png](output_32_0.png)



```python
airline_early_departures['origin'].value_counts()

    LGA    63129
    JFK    61146
    EWR    59300
    Name: origin, dtype: int64
```

```python
from matplotlib.pyplot import show
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(6, 6))
total=airline_early_departures.shape[0]
ax = sns.countplot(x='origin', data=airline_early_departures)
for p in ax.patches:
    height = p.get_height()
    ax.text((p.get_x() + p.get_width()/2),
           height + 100,
           '{:1.2f}'.format(height/total),
            ha="center")
show()
```

![png](output_34_0.png)



```python
# Best Airport as with "No Delays"

# JFK is the best airport from departure delays.

from matplotlib.pyplot import show
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(6, 6))
total=airline_no_delays.shape[0]
ax = sns.countplot(x='origin', data=airline_no_delays)
for p in ax.patches:
    height = p.get_height()
    ax.text((p.get_x() + p.get_width()/2),
           height + 100,
           '{:1.2f}'.format(height/total),
            ha="center")
show()
```
![png](output_35_0.png)


```python
airline_delayed_departures.shape, airline_no_delays.shape, airline_early_departures.shape

    ((136687, 21), (16514, 21), (183575, 21))
```

```python
airline_ontime_arrival['origin_to_dest'].value_counts().head()

    LGA_to_ATL    206
    JFK_to_LAX    196
    JFK_to_SFO    141
    LGA_to_CLT    120
    LGA_to_ORD    115
    Name: origin_to_dest, dtype: int64

```


```python
import seaborn as sns
plt.figure(figsize=(15,4))
%timeit
sns.distplot(airline_ontime_arrival['Velocity_Miles_Minutes'],hist=True,rug=True)
# plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")
#plt.ytitle("Density")
plt.title("Velocity Distribution for flights with no delays in arrivals")
```

![png](output_54_1.png)


```python
airline_ontime_arrival['origin'].value_counts()

    EWR    1916
    JFK    1804
    LGA    1689
    Name: origin, dtype: int64
```



```python
airline_ontime_arrival.groupby(['origin','month'])['origin'].count()

    origin  month
    EWR     1        175
            2        156
            3        154
            4        173
            5        141
            6        136
            7        136
            8        184
            9        125
            10       183
            11       177
            12       176
    JFK     1        166
            2        123
            3        139
            4        150
            5        127
            6        138
            7        157
            8        161
            9        121
            10       177
            11       180
            12       165
    LGA     1        164
            2        133
            3        137
            4        143
            5        112
            6        132
            7        108
            8        133
            9        138
            10       169
            11       161
            12       159
    Name: origin, dtype: int64
```



```python
airline_ontime_arrival_JFK = airline_ontime_arrival[airline_ontime_arrival['origin'] == 'JFK']
airline_ontime_arrival_EWR = airline_ontime_arrival[airline_ontime_arrival['origin'] == 'EWR']
airline_ontime_arrival_LGA = airline_ontime_arrival[airline_ontime_arrival['origin'] == 'LGA']
airline_ontime_arrival_JFK.shape, airline_ontime_arrival_EWR.shape, airline_ontime_arrival_LGA.shape

    ((1804, 21), (1916, 21), (1689, 21))
```



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = airline_ontime_arrival_JFK['Velocity_Miles_Minutes'].plot(kind='kde')
ax2 = airline_ontime_arrival_EWR['Velocity_Miles_Minutes'].plot(kind='kde')
ax3 = airline_ontime_arrival_LGA['Velocity_Miles_Minutes'].plot(kind='kde')

ax1.set_xlim([1,10])
ax2.set_xlim([1,10])
ax3.set_xlim([1,10])

 # plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution for the flights with no arrival delays ")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```

![png](output_60_0.png)



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(15, 6))

ax1 = sns.kdeplot(airline_ontime_arrival_JFK['Velocity_Miles_Minutes'], shade = True)
ax2 = sns.kdeplot(airline_ontime_arrival_EWR['Velocity_Miles_Minutes'], shade = True)
ax3 = sns.kdeplot(airline_ontime_arrival_LGA['Velocity_Miles_Minutes'], shade = True)

ax1.set_xlim([1,10])
ax2.set_xlim([1,10])
ax3.set_xlim([1,10])

# plots an axis lable
plt.xlabel("Velocity in Miles per Minutes")    
plt.title("Velocity Distribution for the flights with no delays in arrivals")
# sets our legend for our graph.
plt.legend(('Origin EWR', 'Origin JFK','Origin LGA'),loc='best') ;
```

![png](output_61_0.png)



```python
airline_ontime_arrival['origin'].value_counts()

    EWR    1916
    JFK    1804
    LGA    1689
    Name: origin, dtype: int64
```
