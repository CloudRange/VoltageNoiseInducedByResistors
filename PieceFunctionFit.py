
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Scripts to create and plot piecewise polynomial functions for a set of data and save the results into a .CSV file 
it includes code to plot its results for the amplifier gain and the calibration factor of the sound card
"""

## Creates a set of piecewise polynomial functions for a set of data
def get_fits(x, y, yerr, n=3):
    """
    Gets a polynomial function fitted between n data points

    args:
        x: x-axis values
        y: y-axis values
        yerr: Error values on the y-axis
        n: Number of data points used to create a piecewise polynomial function (Default 3)

    returns:
        range_of_functions: A list of arrays that contains the range of which each piecewise function is made for
        fitting: A list of arrays that contains the parameters of each piecewise function (the degree of which is
            decided by n)
        error: A list containing the relative error of the y-axis values
    """
    range_of_functions = []
    fitting = []
    error = []
    i = 0
    while i < len(x):
        if i + n < len(x) - 1:
            fit = np.polyfit(x[i:i + n], y[i:i + n], deg=n-1)
            chi_squared = np.sum(
                (np.polyval(fit, x[i:i + n]) - y[i:i + n]) ** 2)
            # print(chi_squared)
            a = np.asarray([min(x[i:i + n]), max(x[i:i + n])])
            range_of_functions.append(a)
            error.append(sum(yerr[i:i + n]/y[i:i + n]))
        else:
            fit = np.polyfit(x[i:], y[i:], deg=2)
            a = np.asarray([float(min(x[i:])), float(max(x[i:]))])
            range_of_functions.append(a)
            error.append(sum(yerr[i:]/y[i:]))
        fitting.append(fit)
        i += n - 1
    return range_of_functions, fitting, error


def plotData(x, y, yerr, range_of_functions, fitting):
    """
    Plots the piecewise function calculated by get_fits() over the dataset with no labels or titles

    args:
        x: x-axis values
        y: y-axis values
        yerr: Error values on the y-axis
        range_of_functions: A list of arrays that contains the range of which each piecewise function is made for
        fitting: A list of arrays that contains the parameters of each piecewise function (the degree of which is
            decided by n)
    """
    x = x.to_numpy()
    y = y.to_numpy()
    plt.errorbar(x, y, yerr=yerr)
    for i in range(len(fitting)):
        temp_x = np.linspace(range_of_functions[i][0], range_of_functions[i][1], 100)
        function = np.poly1d(fitting[i])
        plt.plot(temp_x, function(temp_x))


def to_csv(file_name, range_of_functions, fitting, error):
    """
    Saves to a csv the range_of_functions, fitting, and error lists for use in a different .py file

    args:
        file_name: x-axis values
        range_of_functions: A list of arrays that contains the range of which each piecewise function is made for
        fitting: A list of arrays that contains the parameters of each piecewise function (the degree of which is
            decided by n)
        error: A list containing the relative error of the y-axis values
    """
    df = pd.DataFrame(range_of_functions)
    df.columns = ['minimum', 'maximum']
    df["Error_Percentage"] = error
    df2 = pd.DataFrame(fitting)
    df = df.join(df2)
    df.to_csv(file_name + ".cvs", encoding='utf-8', index=False)




# MAIN SCRIPT

# Read csv into dataframes
gain_df = pd.read_csv("Gain_Data.csv").sort_values(by=['frequency'])
calibration_df = pd.read_csv("Final_Calibration_Data.csv").sort_values(by=['frequency'])


# fit the piecewise polynomial functions and returns the data for the amplifier gain
ranges, amplifier_fits, error_within_percentage = get_fits(gain_df["frequency"], gain_df["gain"], gain_df["std"], 3)


# Plots the Amplifier gain vs frequency and the fitted piecewise functions
plotData(gain_df["frequency"], gain_df["gain"], gain_df["std"], ranges, amplifier_fits)
plt.xscale("log")
plt.title("Gain vs Frequency")
plt.xlabel("Frequency")
plt.ylabel("Gain")
plt.tight_layout()
plt.show()

# Saves the parameters of the amplifier gain piecewise functions
to_csv('AmplifierFits', ranges, amplifier_fits, error_within_percentage)




# fit the piecewise polynomial functions and returns the data for the sound card calibration factor
ranges, calibration_fits, error_within_percentage = get_fits(calibration_df["frequency"], calibration_df["Calibration"]
                                                             , calibration_df["Error"], 3)

# Plots the calibration factor vs frequency and the fitted piecewise functions
plotData(calibration_df["frequency"], calibration_df["Calibration"], 0*calibration_df["Calibration"],
         ranges, calibration_fits)
plt.xscale("log")
plt.title("Calibration Factor vs Frequency")
plt.xlabel("Frequency")
plt.ylabel("Calibration Factor")
plt.tight_layout()
plt.show()

# Saves the parameters of the calibration factor piecewise functions
to_csv('CalibrationFits', ranges, calibration_fits, error_within_percentage)
