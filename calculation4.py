
# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt
import numpy
import scipy.optimize
import scipy.stats
# In[3]:


x_val = numpy.array([980.665
,1961.33
,2981.2216
,3971.69325
,5011.19815
,5883.99])
y_val = numpy.array([327.23887543891874
,24.537111387273516
,74.00785228529382
,116.53769081326011
,167.14221556858803
,212.26746729247134])
y_err = numpy.array([0.017146783
,0.00350311
,0.004357089
,0.003466589
,0.007204742
,0.002827255])


# In[3]:


def linear_model(x, param_vals):
    return param_vals[0] + param_vals[1]*x

def quadratic_model(x, param_vals):
    return param_vals[0] + param_vals[1]*x**2

def chi_squared(model_params, model, x_data, y_data, y_error):
    return numpy.sum(((y_data - model(x_data, model_params))/y_error)**2)


# In[4]:

def calculation(x_values, y_values, y_errors, diameter, thickness, lamb0):
    model_function = linear_model

    initial_values = numpy.array([0, 10**(-6)])


    deg_freedom = x_values.size - initial_values.size # Make sure you understand why!
    print('DoF = {}'.format(deg_freedom))


    fit_linear = scipy.optimize.minimize(chi_squared, initial_values, args=(model_function, x_values, y_values, y_errors))

    print(fit_linear.success) # Did the minimisation complete successfully?
    print(fit_linear.message)


    # In[5]:


    a_solution = fit_linear.x[0]
    b_solution = fit_linear.x[1]

    print('best linear fit a = {} a_units?'.format(a_solution))
    print('best linear fit b = {} b_units?'.format(b_solution))

    # minimized value for chisq function is fit_linear.fun
    print('minimised chi-squared = {}'.format(fit_linear.fun))


    # In[6]:


    chisq_min = chi_squared([a_solution, b_solution], model_function, x_values, y_values, y_errors)
    print('chi^2_min = {}'.format(chisq_min))

    chisq_reduced = chisq_min/deg_freedom
    print('reduced chi^2 = {}'.format(chisq_reduced))


    # In[7]:




    P_value = scipy.stats.chi2.sf(chisq_min, deg_freedom)
    print('P(chi^2_min, DoF) = {}'.format(P_value))


    # In[8]:


    plt.figure(figsize=(8,6))
    plt.errorbar(x_values,
                 y_values,
                 yerr=y_errors,
                 marker='o',
                 linestyle='None')

    plt.xlabel('applied force (N)') # axis labels and units
    plt.ylabel('fringe order')

    # Generate best fit line using model function and best fit parameters, and add to plot
    fit_line = model_function(x_values, [a_solution, b_solution])
    plt.plot(x_values, fit_line, 'r')
    plt.show()


    # In[9]:


    smooth_xvals = numpy.linspace(min(x_values), max(x_values), 1000)
    # make a smoother line - use 1000 equally spaced points over the range of the measured points.

    plt.figure(figsize=(8,6))
    plt.errorbar(x_values,
                 y_values,
                 yerr=y_errors,
                 marker='o',
                 linestyle='None')

    plt.xlabel('applied force (N)') # axis labels and units
    plt.ylabel('fringe order')

    simulated_line = model_function(smooth_xvals, [a_solution, b_solution])
    plt.plot(smooth_xvals, simulated_line, 'r')
    plt.title("2")
    plt.show()


    # In[10]:


    a_range = 1
    b_range = 0.05

    n_points = 100

    # Generate grid and data
    a_axis = numpy.linspace(a_solution-a_range, a_solution+a_range, num=n_points)
    b_axis = numpy.linspace(b_solution-b_range, b_solution+b_range, num=n_points)
    plot_data = numpy.zeros((n_points, n_points))

    for i, b_val in enumerate(b_axis): # Nested loops to demonstrate what is happening...
        for j, a_val in enumerate(a_axis): # (numpy can actually do this far more efficiently as a vectorised calculation)
            plot_data[i][j] = chi_squared([a_val, b_val], model_function, x_values, y_values, y_errors)


    # In[11]:


    plt.figure(figsize=(6,6))
    im = plt.imshow(plot_data, extent=(a_solution-a_range, a_solution+a_range, b_solution-b_range, b_solution+b_range),
                    origin='lower', aspect='auto')

    plt.xlim(a_solution-a_range, a_solution+a_range) # axis ranges
    plt.ylim(b_solution-b_range, b_solution+b_range)

    plt.ylabel('b (units?)') # Axis labels
    plt.xlabel('a (units?)')

    cbar=plt.colorbar(im, orientation='vertical') # # Colorbar and label
    cbar.set_label(r'$\chi^2$', fontsize=12)
    plt.plot(a_solution, b_solution, 'wo') # Add in best fit point and dashed lines
    plt.plot((a_solution, a_solution), (b_solution-b_range, b_solution), linestyle='--', color='w')
    plt.plot((a_solution-a_range, a_solution), (b_solution, b_solution), linestyle='--', color='w')
    plt.title("3.1")
    plt.show()


    # In[12]:


    X, Y = numpy.meshgrid(a_axis, b_axis, indexing='xy')
    contour_data = plot_data - chisq_min

    levels = [1, 4, 9] # Contour levels to plot - delta chi-squared of 1, 4 & 9 correspond to 1, 2 & 3 standard deviations
    plt.figure(figsize=(7,8))
    contour_plot = plt.contour(X, Y, contour_data, levels=levels, colors='b', origin = 'lower')
    plt.clabel(contour_plot, levels, fontsize=12, inline=1, fmt=r'$\chi^2 = \chi^2_{min}+%1.0f$')

    plt.xlabel('a (units?)') # Axis labels
    plt.ylabel('b (units?)')

    import matplotlib.ticker as ticker # This allows you to modify the tick markers to assess the errors from the chi-squared contour plots.

    xtick_spacing = 0.25
    ytick_spacing = 0.02

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))


    plt.plot(a_solution, b_solution, 'ro') # Add in best fit point and dashed lines to axes
    plt.plot((a_solution, a_solution), (b_solution-b_range, b_solution), linestyle='--', color='r')
    plt.plot((a_solution-a_range, a_solution), (b_solution, b_solution), linestyle='--', color='r')
    plt.title("contour plot")
    plt.show()


    # In[13]:


    """contours = contour_plot.collections[0].get_paths()    # Get hold of the contours from the plot
    onesigma_contour = contours[0].vertices       # Grab the set of points constituting the one confidence interval contour

    maxs = numpy.amax(onesigma_contour, axis=0)   # Get the extrema along the two axes - max and min values
    mins = numpy.amin(onesigma_contour, axis=0)   # These should be symmetric about the solution...
    errs_graphical = (maxs-mins)/2                          # Calculate one standard error in the parameters

    a_error = errs_graphical[0]
    b_error = errs_graphical[1]

    print('Parameter a = {} +/- {}'.format(a_solution, a_error))
    print('Parameter b = {} +/- {}'.format(b_solution, b_error))
    """

    # In[14]:


    errs_Hessian = numpy.sqrt(numpy.diag(2*fit_linear.hess_inv))

    a_err = errs_Hessian[0]
    b_err = errs_Hessian[1]

    print('Parameter a = {} +/- {}'.format(a_solution, a_err))
    print('Parameter b = {} +/- {}'.format(b_solution, b_err))


    # In[15]:


    m = b_solution
    m_err = b_err

    D = diameter[0]
    D_err = diameter[1]

    f_sigma = 8/(numpy.pi*D*m)

    f_sigma_err = f_sigma*((m_err/m)**2+(D_err/D)**2)**(1/2)

    print(f_sigma, f_sigma_err)


    # In[16]:


    lamb = lamb0[0] * 10**(-6)
    lamb_err = lamb0[1] * 10**(-6)
    #lamda_err = 0 # mm

    C = lamb/f_sigma

    C_err = C*((lamb_err/lamb)**2+(f_sigma_err/f_sigma)**2)**(1/2)
    print(C,C_err)
    return [C, C_err]
#calculation(x_val, y_val, y_err)
