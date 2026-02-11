import pandas as pd
import glob
import re
from scipy import optimize
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


"""
Scripts for organizing data collected from a sound card connected to a circuit containing a voltage divider, 
an amplifier, and a function generator into cvs files and graphs to show irregularities at high or low 
frequencies to determine the limits of the amplifier.
"""


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

    folders = glob.glob(directory + "/*")

    input_voltage = []
    amplitude_fit = []
    frequency = []
    amplitude_error = []

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

            # Extract a 1.5-second window in order to avoid startup issues
            time, data = reduce_range(time, data['left'], 1, 1.5)

            # Stereo to mono conversion for the wav file data
            data = data * 2

            # Fit the data to a sinusoidal function to extract its amplitude
            if np.size(data) > 0:
                params, params_covariance = optimize.curve_fit(test_func, time, data, p0=[.1, float(u[3]), 30])
                p_error = np.sqrt(np.diag(params_covariance))
                if p_error[1] < 25:
                    amplitude_fit.append(float(params[0]))
                    amplitude_error.append(p_error[0])
                    input_voltage.append(float(u[1]) / 2000)
                    frequency.append(float(u[3]))

                    '''        
                    Deprecated, used to plot the fits to ensure they did not have any errors

                    fig, ax1 = plt.subplots()
                    ax1.plot(time, data)
                    ax1.set_ylabel("Sound card scale (Unitless)")
                    ax1.set_xlabel("Time (s)")
                    ax1.set_title("Sin wave with a 100mV amplitude at 1kHz")
                    ax2 = ax1.twinx()
                    ax2.plot(time, (data / 9.19913004009878), color='red')
                    ax2.set_ylabel("Calibrated Scale (mV)")
                    fig.tight_layout()

                    plt.show()
                    '''

    # Create and return dataframe
    df = pd.DataFrame()
    df["frequency"] = frequency
    df["input_voltage"] = input_voltage
    df["amplitude_fit"] = amplitude_fit
    df["amplitude_error"] = amplitude_error

    return df

# MAIN SCRIPT

directory_root = 'AmplifierData/'


# Process all Wav files
df = main_loop(directory_root)
# Save dataframe in .csv format for recordkeeping

df.to_csv("Amplifier_Data.csv", encoding='utf-8', index=False)




# Resistor Values for the voltage divider
r2 = 22
r1 = 500e3
r_total = r1 + r2
r1_error = r1 * .1
r_total_error = r1 * .1 + r2 * .1
gain = []
errors = []

calibration_df = pd.read_csv("Final_Calibration_Data.csv", index_col=False)

# Gain Calculation for each WAV file using the calibration factor of the soundcard to calculate the output voltage
for index, row in df.iterrows():

    # Input voltage to the soundcard calculated after the voltage divider
    Vin = abs(row["input_voltage"] * (1 - r1 / r_total))
    #error_Vin = (r1_error / r1 + r_total_error / r_total) * Vin
    error_Vin = 0

    # get calibration factor for the amplifier
    gain_calibration = calibration_df[calibration_df.frequency
                                      == row["frequency"]]["Calibration"].reset_index(drop=True)[0]


    Vout = (row["amplitude_fit"]) / gain_calibration
    error_Vout = abs(Vout) * (row["amplitude_error"] / row["amplitude_fit"])

    # Get the % increase between the output voltage and input voltage
    gain.append(abs(Vout / Vin))
    error = (error_Vout/Vout + error_Vin/Vin) * abs(Vout / Vin)
    errors.append(error)

df["gain"] = gain
df["Error"] = errors

gain_final = []
unique_frequencies = sorted(df.frequency.unique())

std = []

# Save the mean gain for each unique frequency for posterior use
for i in unique_frequencies:
    tempdf = df[df["frequency"] == i]
    amplitude_propagated_error = ((sum(tempdf["Error"] ** 2)) ** (1 / 2)) / (len(tempdf["Error"]))
    gain_final.append(float(tempdf["gain"].mean()))
    std.append(tempdf["gain"].std() + amplitude_propagated_error)

gain_df = pd.DataFrame()
gain_df["frequency"] = unique_frequencies
gain_df["gain"] = gain_final
gain_df["std"] = std


# Save dataframe in .csv format for posterior use
gain_df.to_csv("Gain_Data.csv", encoding='utf-8', index=False)










# Plot fitted amplitude for each wav file against input voltage
for i in unique_frequencies:
    tempdf = df[df["frequency"] == i]
    plt.errorbar(tempdf["input_voltage"], abs(tempdf["amplitude_fit"]), yerr=tempdf["amplitude_error"], fmt="o",
                 label=(str(i) + "Hz"))

plt.ylabel("Fit Amplitude")
plt.xlabel("Input voltage (mV)")
plt.title("Amplitude vs input voltage before voltage divider")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(13, 13)
fig.tight_layout()
plt.show()

# Plot mean gain for each frequency file against input voltage

plt.ylabel("Gain")
plt.xlabel("Frequency")
plt.title("Gain vs Frequency")
plt.errorbar(gain_df["frequency"], gain_df["gain"], yerr=gain_df["std"], fmt='o')
plt.xscale("log")
fig = plt.gcf()
fig.set_size_inches(10, 6)
fig.tight_layout()
plt.show()