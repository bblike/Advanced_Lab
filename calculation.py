import matplotlib.pyplot as plt
import numpy
import scipy.optimize
import scipy.stats

x_values = numpy.array([230
                           , 227.5
                           , 225
                           , 222.5
                           , 220
                           , 217.5
                           , 215
                           , 212.5
                           , 210
                           , 207.5
                           , 205
                           , 202.5
                           , 200])
y_values = numpy.array([114.4664352
                           , 113.474537
                           , 111.4583333
                           , 109.7719907
                           , 109.8454861
                           , 108.5711806
                           , 107.0474537
                           , 107.9756944
                           , 107.3744213
                           , 107.6487269
                           , 109.0925926
                           , 109.9479167
                           , 112.2893519])
y_errors = numpy.array([0.058230491,
                        0.054734598,
                        0.051089971,
                        0.065772928
                           , 0.044746316
                           , 0.049540277
                           , 0.054992498
                           , 0.03819596
                           , 0.049435855
                           , 0.075949406
                           , 0.056600058
                           , 0.063435438
                           , 0.060568226])

def linear_model(x, param_vals):
    return param_vals[0] + param_vals[1] * x


def quadratic_model(x, param_vals):
    return param_vals[0] + param_vals[1] * x + param_vals[2] * x ** 2


def chi_squared(model_params, model, x_data, y_data, y_error):
    return numpy.sum(((y_data - model(x_data, model_params)) / y_error) ** 2)




def swap(a, b):
    temp = b
    b = a
    a = temp
    return a, b


def rearrange(xvals, yvals, yerrs):
    length = len(xvals)
    for i in range(length):
        for j in range(i, length):
            if xvals[i] < xvals[j]:
                xvals[i], xvals[j] = swap(xvals[i], xvals[j])
                yvals[i], yvals[j] = swap(yvals[i], yvals[j])
                yerrs[i], yerrs[j] = swap(yerrs[i], yerrs[j])
    print("xvals=", xvals)
    print("yvals=", yvals)
    print("yerrors=", yerrs)
    return xvals, yvals, yerrs


def calculation(x_values, y_values, y_errors):
    assert len(y_values) == len(x_values)
    assert len(y_errors) == len(y_values)

    x_values, y_values, y_errors = rearrange(x_values, y_values, y_errors)

    plt.figure(figsize=(8, 6))
    plt.errorbar(x_values,
                 y_values,
                 yerr=y_errors,  # use y_errors array for y error bars
                 marker='o',  # circular markers at each datapoint
                 linestyle='None')  # no connecting lines

    # plt.plot(x_values, quadratic_model(x_values, params))

    plt.xlabel('angle (units)')  # axis labels and units
    plt.ylabel('intensity (units)')
    plt.show()

    model_function = quadratic_model

    initial_values = numpy.array([60, -4000, 600000])

    deg_freedom = x_values.size - initial_values.size  # Make sure you understand why!
    print('DoF = {}'.format(deg_freedom))

    fit_linear = scipy.optimize.minimize(chi_squared, initial_values,
                                         args=(model_function, x_values, y_values, y_errors))
    # print(fit_linear.success) # Did the minimisation complete successfully?
    # print(fit_linear.message)

    a_solution = fit_linear.x[0]
    b_solution = fit_linear.x[1]
    c_solution = fit_linear.x[2]

    print('best linear fit a = {} a_units?'.format(a_solution))
    print('best linear fit b = {} b_units?'.format(b_solution))
    print('best linear fit c = {} b_units?'.format(c_solution))

    # minimized value for chisq function is fit_linear.fun
    print('minimised chi-squared = {}'.format(fit_linear.fun))

    chisq_min = chi_squared([a_solution, b_solution, c_solution], model_function, x_values, y_values, y_errors)
    # print('chi^2_min = {}'.format(chisq_min))

    chisq_reduced = chisq_min / deg_freedom
    print('reduced chi^2 = {}'.format(chisq_reduced))

    P_value = scipy.stats.chi2.sf(chisq_min, deg_freedom)
    print('P(chi^2_min, DoF) = {}'.format(P_value))

    plt.figure(figsize=(8,6))
    plt.errorbar(x_values,
                 y_values,
                 yerr=y_errors,
                 marker='o',
                 linestyle='None')

    plt.xlabel('x data (units)') # Axis labels
    plt.ylabel('y data (units)')

    # Generate best fit line using model function and best fit parameters, and add to plot
    fit_line = model_function(x_values, [a_solution, b_solution, c_solution])
    plt.plot(x_values, fit_line, 'r')
    plt.show()

    smooth_xvals = numpy.linspace(min(x_values), max(x_values), 1000)
    # make a smoother line - use 1000 equally spaced points over the range of the measured points.
    """
    plt.figure(figsize=(8,6))
    plt.errorbar(x_values,
                 y_values,
                 yerr=y_errors,
                 marker='o',
                 linestyle='None')

    plt.xlabel('x data (units)')
    plt.ylabel('y data (units)')"""

    simulated_line = model_function(smooth_xvals, [a_solution, b_solution, c_solution])
    # plt.plot(smooth_xvals, simulated_line, 'r')
    # plt.show()

    a_range = 1
    b_range = 0.05

    n_points = 100

    # Generate grid and data
    a_axis = numpy.linspace(a_solution - a_range, a_solution + a_range, num=n_points)
    b_axis = numpy.linspace(b_solution - b_range, b_solution + b_range, num=n_points)
    plot_data = numpy.zeros((n_points, n_points))

    """for i, b_val in enumerate(b_axis): # Nested loops to demonstrate what is happening...
        for j, a_val in enumerate(a_axis): # (numpy can actually do this far more efficiently as a vectorised calculation)
            plot_data[i][j] = chi_squared([a_val, b_val], model_function, x_values, y_values, y_errors)"""

    errs_Hessian = numpy.sqrt(numpy.diag(2 * fit_linear.hess_inv))

    b_err = errs_Hessian[1]
    c_err = errs_Hessian[2]

    print('Parameter a = {} +/- {}'.format(b_solution, b_err))
    print('Parameter b = {} +/- {}'.format(c_solution, c_err))

    xmin = (-b_solution) / (2 * c_solution)
    print("xmin=", xmin)

    xmin_error = xmin * ((b_err / b_solution) ** 2 + (c_err / c_solution) ** 2) ** (1 / 2)
    print("xmin_error=", xmin_error)
    return xmin, xmin_error


calculation(x_values, y_values, y_errors)