import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import optimize
import glob
import re

"""
Scripts for organizing data collected from a sound card connected to a function generator into cvs files and graphs
to show irregularities at high or low frequencies to determine the limits of the sound card.
"""


## Imports the WAV file and reads only the left channel of the stereo file
def importWAV(filename):
    """
    Imports a WAV file and extracts the left channel data

    args:
        filename:  filepath to the wav file

    returns:
        time: Array of time values (seconds)
        data: Dictionary containing an array with the readings of the left channel data
    """
    samplerate, rawData = wavfile.read(filename)
    time = np.linspace(0, rawData.shape[0] / samplerate, rawData.shape[0])

    data = {'left': rawData}
    return time, data

def reduce_range(time, data, start, delta):
    """
    Extracts a time window from the wav file data

    args:
        time: Array of time values (seconds)
        data: Dictionary containing an array with the readings of the left channel data
        start: Start time (seconds)
        delta: Duration of the time window

    returns:
        time: Filtered time array
        data: Filtered Data array
    """

    return time[(time > start) & (time < start + delta)], data[(time > start) & (time < start + delta)]


def test_func(x, a, b, c):
    """
    Sinusoidal fitting function

    args:
        x: Time Array (seconds)
        a: Amplitude of the Sinusoidal function (Unitless)
        b: Frequency of the Sinusoidal function (Hz)
        c: Phase shift of the Sinusoidal function (Unitless)
    returns:
        y = a * sin(2 * pi * b * x + c)
    """

    return a * np.sin(2 * np.pi * b * x + c)

# main loop to create the dataframe
def main_loop(directory):
    """"
    Processes all the files in a directory to create a dataframe containing the data required to calibrate the sound
    card to a function generator.

    The directory was structured to contain a folder for each frequency used, each frequency folder contains multiple
    wav files each with a different input voltage from the function generator. The data is used to fit a sine wave to
    extract its amplitude

    args:
        directory: location of the directory containing the Frequency folders

    returns:
        df: dataframe with columns: frequency, input_voltage, amplitude_fit, amplitude_error
    """

    folders = glob.glob(directory_root + "/*")
    input_voltage = []
    amplitude_fit = []
    amplitude_error = []
    frequency = []

    folders.sort()

    # Read the sub folders for each frequency
    for i in folders:
        files = glob.glob(i + "/*.wav")
        files.sort()
        # Read each of the files in the folder for each voltage used
        for j in files:
            time, data = importWAV(j)

            # Parse filename to extract input voltage and frequency
            # Expected format: ...In<voltage>mVAm<amplitude>V<freq>Hz.wav

            # amplitude is not used as an initial guess due to the lack of units on the soundcard data it was stored
            # just for recordkeeping
            u = re.split('In|mVAm|V|Hz.wav', j)

            # Extract a 2-second window in order to avoid startup issues
            time, data = reduce_range(time, data['left'], 1, 2)

            # Stereo to mono conversion for the wav file data
            data = data * 2

            # Fit the data to a sinusoidal function to extract its amplitude
            if np.size(data) > 0:
                params, params_covariance = optimize.curve_fit(test_func, time, data, p0=[.1, int(u[3]), 0])
                p_error = np.sqrt(np.diag(params_covariance))
                if p_error[1] < 25:
                    amplitude_fit.append(float(params[0]))
                    amplitude_error.append(p_error[0])
                    input_voltage.append(float(u[1]) / 2000)
                    frequency.append(float(u[3]))

                    '''
                    Deprecated, used to plot the fits to ensure they did not have any errors

                    # plt.plot(time, data)
                    # plt.plot(time, test_func(time, params[0], params[1], params[2]))
                    # print(u)
                    # print(float(u[1])/2000)
                    # print(params[0])
                    # plt.show()

                    # fig, ax1 = plt.subplots()
                    # ax1.plot(time, data)
                    # ax1.set_ylabel("Sound card scale (Unitless)")
                    # ax1.set_xlabel("Time (s)")
                    # ax1.set_title("Sin wave with a 100mV amplitude at 1kHz")
                    # ax2 = ax1.twinx()
                    # ax2.plot(time, (data / 9.19913004009878), color='red')
                    # ax2.set_ylabel("Calibrated Scale (mV)")
                    # fig.tight_layout()

                    # plt.show()
                    '''

    # Create and return dataframe
    df = pd.DataFrame()
    df["frequency"] = frequency
    df["input_voltage"] = input_voltage
    df["amplitude_fit"] = amplitude_fit
    df["amplitude_error"] = amplitude_error
    return df


# MAIN SCRIPT

directory_root = 'CalibrationData/'

# Process all Wav files
df = main_loop(directory_root)




# Create a dataframe that only contains the calibration factor for each frequency and its error
final_data = pd.DataFrame()
frq = []
cal = []
error = []
unique_frequencies = sorted(df.frequency.unique())

for i in unique_frequencies:
    tempdf = df[df["frequency"] == i]

    # Linear fit of the amplitude for each input voltage at a specific frequency
    # The slope represents the calibration factor
    model, res, _, _, _ = np.polyfit(tempdf["input_voltage"], abs(tempdf["amplitude_fit"]), 1, full=True)

    # Propagate the errors
    amplitude_propagated_error = ((sum(tempdf["amplitude_error"] ** 2)) ** (1 / 2)) / (len(tempdf["amplitude_error"]))

    frq.append(i)
    cal.append(model[0])
    error.append(float(amplitude_propagated_error + res[0]))

final_data["frequency"] = frq
final_data["Calibration"] = cal
final_data["Error"] = error


# Save both dataframes in .csv format for recordkeeping and posterior use respectively
df.to_csv("RawCalibrationData.csv", encoding='utf-8', index=False)
final_data.to_csv("Final_Calibration_Data.csv", encoding='utf-8', index=False)


# Plot fitted amplitude for each wav file against input voltage
for i in unique_frequencies:
    tempdf = df[df["frequency"] == i]
    plt.errorbar(tempdf["input_voltage"], abs(tempdf["amplitude_fit"]), yerr=tempdf["amplitude_error"], fmt="o",
                 label=(str(i) + "Hz"))

plt.ylabel("Fit Amplitude")
plt.xlabel("Input voltage (mV)")
plt.title("Amplitude vs input voltage")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(13, 13)
fig.tight_layout()
plt.show()

# Plot calibration factor vs frequency (Log vs linear)
plt.ylabel("Slope from sound card fit function (1/mV)")
plt.xlabel("Frequency")
plt.title("Amplitude vs input voltage")
plt.errorbar(final_data["frequency"], final_data["Calibration"], yerr=final_data["Error"], fmt='o')
plt.xscale("log")
fig = plt.gcf()
fig.set_size_inches(10, 6)
fig.tight_layout()
plt.show()
