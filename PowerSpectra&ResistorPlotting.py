import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import glob
import re
from scipy.optimize import curve_fit


"""
Scripts to calculate the average noise of the spectral density of a resistor in a copper shell connected to an amplifier 
and a soundcard to experimentally verify the formula for thermal noise (V = (4* * k_b * T)**.5) and plot the spectral 
densities of the resistor measurements, and the linear regression of the resistance vs the voltage squared.
"""

def reduce_range(x, y, start, delta):
    """
    Extracts a time window from the wav file data

    args:
        x: List
        y: List
        start: Start time (seconds)
        delta: Duration of the time window

    returns:
        x: Filtered x array
        y: Filtered y array
    """
    return x[(x >= start) & (x <= start + delta)], y[(x >= start) & (x <= start + delta)]



def read_piecewise_functions(df):
    """
    extracts the data for the piecewise functions from a dataframe containing the information of the functions
    args:
        df: a dataframe with the columns minimum, maximum, Error_Percentage, and an n+1 number of columns representing
            the factors for a polynomial function of order n named from 0 to n.

    returns:
        domain: a list of lists that contain 2 floats that represent the start and end for each polynomial function
        functions: a list of poly1d objects with the factors for each polynomial function
        error_within_percentage: the error of the fit for each polynomial function in relative %
    """

    minimum = df['minimum']
    maximum = df['maximum']
    error_within_percentage = df["Error_Percentage"].to_numpy()

    # Create a list that contains the range for each function, each item in the list contains a list with 2 elements in
    # it representing the start and end of the function range
    domain = []
    for i in range(len(maximum)):
        domain.append([minimum[i], maximum[i]])

    # Removes all other values that are not the factors for the polynomial function
    functions = []
    df = df.drop(["minimum", "maximum", "Error_Percentage"], axis=1)

    # creates a list containing poly1d objects with the factors for each polynomial function
    for i in df.to_numpy():
        functions.append(np.poly1d(i))

    return domain, functions, error_within_percentage


def importWAV(filename):
    """
    Imports a WAV file and extracts the left and right channels data

    args:
        filename:  filepath to the wav file

    returns:
        x: Array of time values (seconds)
        y: Dictionary containing an array with the readings of the left and right channels data
    """
    samplerate, rawData = wavfile.read(filename)
    x = np.linspace(0, rawData.shape[0] / samplerate, rawData.shape[0])

    y = {'left': rawData[:, 0], 'right': rawData[:, 1]}
    return x, y

""" 
Deprecated used for non stereo data during testing


def plot_welch_dataORG(x, y):
    fs = np.size(x) / (max(x) - min(x))

    f, Pxx_den = signal.welch(y, fs, nperseg=8 * 1024)

    f, Pxx_den, yerr = rescale_welch(f, Pxx_den)

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_xlabel('Frequency [Hz]', fontsize=10)
    ax1.set_ylabel('PSD [V**2/Hz]', fontsize=10)
    ax1.set_title('Power Spectral Density, linear scale', fontsize=14)
    ax2.set_xlabel('Frequency [Hz]', fontsize=10)
    ax2.set_ylabel('PSD [V**2/Hz]', fontsize=10)
    ax2.set_title('Power Spectral Density, semilog scale', fontsize=14)

    ax1.plot(f, Pxx_den)
    ax2.semilogy(f, Pxx_den)

    fig.tight_layout()
    # plt.show()
    
    
    
def plot_periodogram(x, y):
    fs = np.size(x) / (max(x) - min(x))

    f, Pxx_den = signal.periodogram(y, fs, 'flattop', scaling='density')

    f, Pxx_den = rescale_welch(f, Pxx_den)
    plt.semilogy(f, Pxx_den)
    return np.mean(Pxx_den)
    
    
def plot_Spectral_Density(x, y):
    fs = np.size(x) / (max(x) - min(x))
    f, Pxx_spec = signal.periodogram(y['left'], fs, 'flattop', scaling='spectrum')
    f, Pxx_spec, yerr = rescale_welch(f, Pxx_spec)
    f, Pxx_spec = reduce_range(f, Pxx_spec, 3000, 5000)
    plt.semilogy(f, Pxx_spec)

    plt.xlabel('Frequency [Hz]', fontsize=15)
    plt.ylabel('[V**2]', fontsize=15)
    plt.title('Power Spectrum', fontsize=20)

    # plt.show()

    return np.mean(Pxx_spec), np.std(Pxx_spec)
"""

def plot_welch_data(x, y):
    """
    Plots the cross correlation spectral density of a list x and a dictionary with keys left and right that each
    contain lists.


    args:
        x: Array of time values (seconds)
        y: Dictionary containing an array with the readings of the left and right channels data

    return:
        np.mean(Pxx_den): mean value of the spectral density
        np.std(Pxx_den) + error_propagated: the standard deviation of the spectral density plus the propagated error
            of the rescaling used as the absolute error of the mean
    """

    fs = np.size(x) / (max(x) - min(x))

    f, Pxx_den = signal.csd(y['left'], y['right'], fs, nperseg=8 * 1024)

    # rescale the data into proper units (Voltage squared) and remove analytically the gain from the amplifier
    f, Pxx_den, y_error = rescale_welch(f, Pxx_den)

    # Reduce the range of the data to the range of the measured data for a range in which the piecewise functions had
    # good fits (50Hz - 10kHz)
    f, Pxx_den = reduce_range(f, Pxx_den, 50, 10000)


    # mean of the error
    error_propagated = ((sum(y_error ** 2)) ** (1 / 2)) / (len(y_error))

    plt.semilogy(f, Pxx_den)
    return np.mean(Pxx_den), np.std(Pxx_den) + error_propagated


def rescale_welch(x, y):
    """
    Function to rescale the data into proper units (Voltage squared) and remove analytically the gain from the amplifier
    using the piecewise functions calculated previously in file PieceFunctionFit.py

    gain_frequencies, gain_functions, gain_error_within_percentage, calibration_frequencies, calibration_functions, and
    calibration_error_within_percentage are constants grabbed from the outer scope to prevent rereading the csv each time
    this function is called, and to keep the argument list for this function and plot_welch_data() to variables only.

    args:
        x: list of frequencies
        y: list of amplitudes

    return:
        x: list of frequencies unchanged
        y: list of voltages squared
        yerr: absolute error of the rescaling
    """

    freq_min = calibration_frequencies[0][0]
    freq_max = calibration_frequencies[len(calibration_frequencies)-1][1]


    j = 0
    yerr = np.zeros(np.size(y))

    for i in range(len(x)):
        if (x[i] >= calibration_frequencies[j][0]) and (x[i] <= calibration_frequencies[j][1]):
            y[i] = y[i] / ((calibration_functions[j](x[i])) ** 2)
            yerr[i] = calibration_error_within_percentage[j]
        elif freq_min < x[i] < freq_max:
            while not ((x[i] >= calibration_frequencies[j][0]) and (x[i] <= calibration_frequencies[j][1])):
                j += 1
                if j >= len(calibration_frequencies):
                    break
            y[i] = y[i] / ((calibration_functions[j](x[i])) ** 2)
            yerr[i] = calibration_error_within_percentage[j]



    j = 0

    for i in range(len(x)):
        if (x[i] >= gain_frequencies[j][0]) and (x[i] <= gain_frequencies[j][1]):
            y[i] = y[i] / ((gain_functions[j](x[i])) ** 2)
            yerr[i] = (yerr[i] + gain_error_within_percentage[j]) * y[i]
        elif freq_min < x[i] < freq_max:
            while not ((x[i] >= gain_frequencies[j][0]) and (x[i] <= gain_frequencies[j][1])):
                j += 1
                if j >= len(gain_frequencies):
                    break
            y[i] = y[i] / ((gain_functions[j](x[i])) ** 2)
            yerr[i] = (yerr[i] + gain_error_within_percentage[j]) * y[i]


    return x, y, yerr


def linearFunc(x,intercept,slope):
    """
    A linear function of the form y = a*x + b

    args:
        x: list of values
        intercept: value of y at x=0 (float)
        slope: rate of change of y with respect to x (float)

    returns:
        intercept + slope * x: a list of values calculated using the return formula

    """
    return intercept + slope * x



# MAIN SCRIPT

# Read the calibration and gain piecewise function fit data and store them as variables
calibrationFits_df = pd.read_csv("CalibrationFits.cvs")
amplifierFits_df = pd.read_csv("AmplifierFits.cvs")
gain_frequencies, gain_functions, gain_error_within_percentage = read_piecewise_functions(amplifierFits_df)
calibration_frequencies, calibration_functions, calibration_error_within_percentage = read_piecewise_functions(calibrationFits_df)




directory_root = 'ResistorData/'
folders = glob.glob(directory_root + "/*")
resistor = []
avg_noise = []
stds = []

# Read each of the files in the folder for each resistance used
for h in folders:
    time, data = importWAV(h)

    # Create superimposed plots of the spectral density of the data
    noise, std = plot_welch_data(time, data)

    # Parse filename to extract resistance
    # Expected format: ...<Resistance>Ohm.wav
    u = re.split('ResistorData|Ohm.wav', h)
    resistance = float(u[1][1:])


    resistor.append(resistance)
    avg_noise.append(noise)
    stds.append(std)
plt.show()




# Plot the resistance vs the average noise of the spectral density and its linear regression
cm = 1 / 2.54  # centimeters in inches
fig, ax1 = plt.subplots(figsize=(10 * cm, 10 * cm))

ax1.errorbar(resistor, avg_noise, stds, fmt="o")
df = pd.DataFrame()
df["Resistor"] = resistor
df["Avg_noise"] = avg_noise
df["Std"] = stds
df["Avg_noise/(4 * Resistor * 298)"] = df["Avg_noise"] / (4 * df["Resistor"] * 298)



# Create a linear regression using the resistance used as the x-axis and the average noise of the spectral density as
# the y-axis
# https://faculty1.coloradocollege.edu/~sburns/LinearFitting/SimpleDataFittingWithError.html
a_fit, cov = curve_fit(linearFunc, resistor, avg_noise, sigma=stds, absolute_sigma=True)



inter = a_fit[0]
slope = a_fit[1]
d_inter = np.sqrt(cov[0][0])
d_slope = np.sqrt(cov[1][1])

yfit = slope * df["Resistor"]


# Print the results of the final linear regression
print(f'The slope = {slope}, with uncertainty {d_slope}')
print(f'The slope/(4*298) = {slope/(4*298)}, with uncertainty {(d_slope/slope) * (slope/(4*298))}')
print(f'The intercept = {inter}, with uncertainty {d_inter}')


plt.plot(resistor,yfit,label='Fit')

ax1.set_ylabel("V**2")
ax1.set_xlabel("Resistance (Ohms)")
ax1.set_title("Resistance vs Spectral Density")
plt.tight_layout()
plt.show()
