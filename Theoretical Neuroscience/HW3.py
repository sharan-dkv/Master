# Import section
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# The delta time for the izhikevich model
dt = 0.01



def file_parser(file_n):
    # Function to load the file
    info = scipy.io.loadmat('orientation_tuning_data.mat')
    return info

def izhikevich_model(mode,dt,total_time):
    # Model to run the izhikevich model
    # The izhikevich model can be run in 3 modes
    # RS - Regular Spiking
    # FS - Fast Spiking
    # Chatter - Chattering
    # The parameters can be changed by changing the values in the izhikevich_model function
    if mode == "RS":
        a = 0.02
        b = 0.2
        c = -65
        d = 8
        v = []
        v.append(c)
        u = b*v[-1]

    elif mode == "FS":
        a = 0.1
        b = 0.2
        c = -65
        d = 2
        v = []
        v.append(c)
        u = b * v[-1]

    elif mode == "Chatter":
        a = 0.02
        b = 0.2
        c = -50
        d = 2
        v = []
        v.append(c)
        u = b * v[-1]

    else :
        print("Not an option")
        return(0)

    i = np.full([int(total_time / dt)], 10)
    t_li = np.linspace(0, total_time, (int(total_time / dt)))
    for j in range(len(i)):
        temp = du(a,b,v[-1],u,dt)
        v.append( dv(v[-1],u,i[j]))
        u = temp
        if v[-1]>=30:
            print(1)
            v[-1] = c
            u = u + d
    #print(v)
    plt.plot(t_li,v[:(len(t_li))])
    plt.xlabel('Time in ms')
    plt.ylabel('Vm')
    plt.show()

def dv(v,u,i):
    # Returns the membrane potential
    return (v+(0.04*pow(v,2)+5*v+140-u+i)*dt)

def du(a,b,v,u,dt):
    # Returns the recovery variable u
    return (u+dt*a*(b*v - u))

def Vm_plotter(orient_mat):
    # Plotting and fitting functions for question 2 are done her

    # Parsing the data into 2 different dataframes
    orientations = orient_mat['Stimuli'][:, 0]
    timestamps = orient_mat['Stimuli'][:, 1]
    Vm = orient_mat['Vm'][0]

    # Plotting the first 5 seconds
    temp_time = np.linspace(0, len(Vm)/10000, len(Vm))
    plt.plot(temp_time[:50000], Vm[:50000])
    plt.xlabel('Time in s')
    plt.ylabel('Vm')

    # counting the number of spikes using the function spike counter
    print("The number of spikes in the data is - ")
    print(spike_counter(Vm[5739:]))

    # Finding the number of spikes for each time window
    previous_time = 5739
    spikes = pd.DataFrame(orientations,columns=['orientation'])
    r_t = []
    for i in range(1,len(timestamps)+1):
        if i==len(timestamps):
            temp_vm = Vm[previous_time:]
            r_t.append(spike_counter(temp_vm))
        else:
            temp_vm = Vm[previous_time:timestamps[i]]
            r_t.append(spike_counter(temp_vm))
            previous_time = timestamps[i]

    # Adding all the data to a dataframe called spikes
    spikes['spikes'] = r_t
    spikes['timestamps'] = timestamps
    spikes['time_delta'] = timestamps

    # Calculating the time in each orientation
    previous_time = 0
    for i in range(len(spikes)):
        spikes['time_delta'][i] = spikes['timestamps'][i] - previous_time
        previous_time = spikes['timestamps'][i]

    # Calculating the firing rate in each orientation
    summation = np.zeros(17)
    previous_time = 0
    for i in range(len(summation)):
        summation[i] = spikes[spikes['orientation']==i]['spikes'].mean()
        summation[i] = (summation[i]*10000)/(spikes[spikes['orientation']==1]['time_delta'].mean())


    # Plotting the tuning curve
    orientation_array = np.arange(0,16)
    plt.figure('Tuning curve')
    plt.plot(22.5*orientation_array,summation[:16])
    plt.xlabel('Orientation')
    plt.ylabel('Firing rate')
    plt.title('Tuning curve')

    # Fitting the Gaussian tuning curve for the points from 90 to 270 degrees
    parameters, covariance = curve_fit(gauss_tuning_curve,orientation_array[4:12],summation[4:12])

    r_max = parameters[0]
    theta_0 = parameters[1]
    sig = parameters[2]

    print("The Parameters for the fit 1 are  -")
    print(r_max,theta_0,sig)
    # Plotting the fit data vs the initial data
    y = []
    temp_angles = np.linspace(4, 12, 100)
    for orient in temp_angles:
        y.append(gauss_tuning_curve(orient,r_max,theta_0,sig))

    plt.figure('Gauss Tuning curve - 1')
    plt.scatter(22.5 * orientation_array[4:12], summation[4:12],label = 'Data')
    plt.xlabel('Orientation')
    plt.ylabel('Firing rate')
    plt.title('Gauss Tuning curve - 1')
    plt.plot(22.5*temp_angles,y,'k-',label = 'Guass fit')
    plt.legend()

    # Fitting the second part of the data to the gaussian tuning curve
    temp_angles = np.linspace(0,7,8)
    temp_summation = summation[12:16].tolist() + summation[0:4].tolist()

    parameters, covariance = curve_fit(gauss_tuning_curve,temp_angles,temp_summation)
    r_max = parameters[0]
    theta_0 = parameters[1]
    sig = parameters[2]

    print("The Parameters for the fit 2 are  -")
    print(r_max,theta_0,sig)

    plt.figure('Gauss Tuning curve - 2')
    plt.scatter(22.5 * temp_angles, temp_summation, label='Data')
    plt.xlabel('Orientation with a phase difference of pi/2')
    plt.ylabel('Firing rate')
    plt.title('Gauss Tuning curve - 2')

    y = []
    temp_angles = np.linspace(0, 8, 100)
    for orient in temp_angles:
        y.append(gauss_tuning_curve(orient, r_max, theta_0, sig))

    plt.plot(22.5 * temp_angles, y, 'k-', label='Guass fit')
    plt.legend()
    plt.show()


def gauss_tuning_curve(x,r_max, theta_0,sig):
    # Returns the gauss tuning curve function
    return r_max*np.exp(-0.5*(pow(((x-theta_0)/sig),2)))


def spike_counter(Vm):
    # Function to count the number of spikes in a array Vm
    # Counts spikes based on the number of times the membrane potential crosses 0
    counter = 0
    times = []
    start = 0
    flag = 0
    for i in range(len(Vm)):
        if Vm[i] >= 0 and flag ==0:
            flag = 1
            counter += 1
            start = i
        elif Vm[i] <0 and flag ==1:
            flag = 0
            times.append(i-start)
    #print(mean(times))
    return counter

if __name__ == '__main__':
    # Part - 1
    izhikevich_model("RS",dt,100)

    #Part - 2
    file_name = 'orientation_tuning_data.mat'
    orient_mat = file_parser(file_name)
    Vm_plotter(orient_mat)