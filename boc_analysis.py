#Función que grafica BOC + PDFS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from sklearn.metrics import roc_auc_score
from scipy.signal import argrelextrema

def bocplot(D, TITLE, i, units = "units", intervals = 20):
    '''
    
    This function describe the data with the PDF plots involved in the the BOC
    D: Dictionary with the databse
    Title: plot titles.
    i: enumerated variable
    units: units
    intervals: evaluation of 20 equidistak values with Kde

    '''

    #we initialize an array of four plots
    figs, axs = plt.subplots(1, 4, figsize=(12, 3))

    #this is the number format 
    formatter = plt.FuncFormatter(lambda x, _: f"{x:.0e}")

    axs[2].yaxis.set_major_formatter(formatter)
    axs[3].yaxis.set_major_formatter(formatter)

    for ti, d in enumerate(D):

        #rank
        X = d["X"]

        #rank labels
        Y = d["Y"]

        #getting color
        color = d["color"]

        #Prior
        M = np.average(Y)

        #AUC score
        auc_score = roc_auc_score(Y, X)

        #Kernel density estimation from the complete data
        kde_prior = gaussian_kde(X)

        #Kernle density estimation of the likelihood using all the data
        kde_lklihood = gaussian_kde(X[Y==1])

        # Assuming the data is continuous we create a grid of 20 equidistant values
        x = np.linspace(X.min(), X.max(), intervals)  

        #thiis is the delta
        w = x[1:]-x[:-1]

        #evaluating the equidistant values

        #prior p(x)
        pdf_prior = kde_prior.evaluate(x) 

        #likelihood #p(X | Y=1)
        pdf_lklihood = kde_lklihood.evaluate(x) 

        #
        marginal = np.average(Y) #p(Y=1)

        if i == 0:

            axs[3].set_title("Marginal")
            axs[2].set_title("Likelihood")
            axs[1].set_title("Likelihood/Marginal ratio")
            axs[0].set_title(r'BOC')

        axs[3].set_ylabel(r"$p(\theta)$")
        axs[3].set_xlabel(r"$\theta$")
        axs[3].plot(x, pdf_prior, color = color)
    
  
        ###
        # Configure the y-axis to use scientific notation with one integer and one decimal
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # Adjusts when scientific notation is used
        axs[3].yaxis.set_major_formatter(formatter)


        P = np.copy(pdf_prior)
        L = np.copy(pdf_lklihood*marginal)
        P = np.vstack((P, L))

        pdf_lklihood = np.min(P, axis = 0)/marginal

        axs[1].plot(x, pdf_lklihood/pdf_prior, color = "tab:red", alpha = 0.3)
        axs[2].plot(x, pdf_lklihood, color = color)
        axs[2].set_ylabel(r"$p_{X|Y=1}(\theta))$")
        axs[2].set_xlabel(r"$\theta$")
        axs[2].yaxis.set_major_formatter(formatter)
        axs[3].plot(x, pdf_lklihood*marginal, color = color, linestyle = ":")
 
        mx = argrelextrema(pdf_lklihood/pdf_prior, np.greater)[0]
        mn = argrelextrema(pdf_lklihood/pdf_prior, np.less)[0]

        mx = np.append(mx, 0)
      

        axs[1].scatter(x[mx], (pdf_lklihood/pdf_prior)[mx], color = color, alpha = 1, s = 10, label = "maxima")
        axs[1].scatter(x[mn], (pdf_lklihood/pdf_prior)[mn], facecolor="k", edgecolor='k', alpha = 1, s = 8, label = "minima")

        axs[1].set_ylim(-0.2, 7)
        axs[1].set_ylabel(r"$p_{X | Y=1}(\theta)/p_{X | Y=1}(\theta) $")
        axs[1].set_xlabel(r"$\theta$")
        axs[1].hlines(y=1, xmin = np.min(X), xmax = np.max(X), linestyle="--", color = "gray", label = r"$P(Y=1)$={s}".format(s=np.round(marginal,2)), linewidth=1)
        axs[1].hlines(y=0, xmin = np.min(X), xmax = np.max(X), linestyle="--", color = "tab:green", label = r"Ratio Boundaries", linewidth=1)

        axs[1].legend(loc="upper right")

        nvar = np.cumsum(pdf_lklihood[:-1]*w)[-1]
        nvary = np.cumsum(pdf_prior[:-1]*w)[-1]

        axs[1].scatter(x[mx], (pdf_lklihood/pdf_prior)[mx], color = color, alpha = 1, s = 10, label = "maxima")
        axs[1].scatter(x[mn], (pdf_lklihood/pdf_prior)[mn], facecolor="k", edgecolor='k', alpha = 1, s = 8, label = "minima")

        if i == 0:

            axs[1].annotate( '(0, 101.7)',  # No text for the arrow itself
                        xy=(0, 7),  # Head of the arrow
                        xytext=(10, 2),  # Tail of the arrow
                        arrowprops=dict(facecolor='blue', arrowstyle='->')  # Arrow style
                        )

    
        

        if ti==0:

            X1 = np.append(0, np.cumsum(pdf_prior[:-1]*w))/nvary
            Y1 = np.append(0, np.cumsum(pdf_lklihood[:-1]*w))/nvar
            

        else:

            X2 = np.append(0, np.cumsum(pdf_prior[:-1]*w))/nvary
            Y2 = np.append(0, np.cumsum(pdf_lklihood[:-1]*w))/nvar  
        
        auc_score = round(roc_auc_score(Y, -X),2)


        axs[0].text(0.6, 0.012, r"AUC = ${c}$".format(c = auc_score))
        axs[0].text(0.02, 0.93, r"$P(Y=1)={s}$".format(s=round(M, 3)))
        axs[0].scatter(np.append(0, np.cumsum(pdf_prior[:-1]*w))/nvary, np.append(0, np.cumsum(pdf_lklihood[:-1]*w))/nvar, color = "tab:red", linestyle = '-', label = d["label"], alpha = 0.3)
        axs[0].plot(np.append(0, np.cumsum(pdf_prior[:-1]*w))/nvary, np.append(0, np.cumsum(pdf_lklihood[:-1]*w))/nvar, color = "tab:red", linestyle = '-', alpha = 0.3)
        axs[0].plot([0, marginal, 1, 1-marginal,  0], [0, 1, 1, 0, 0], color = "tab:green",  linewidth=2, linestyle = "--", alpha=0.5)
        axs[0].set_ylabel(r"$P(X \leq \theta | Y=1)$")
        axs[0].set_xlabel(r"$P(X \leq \theta)$")
        #axs[0].legend(loc="best")

        axs[0].plot([0, 1], [0,1], color = "tab:gray", linestyle = '-.', alpha = 0.5)

        axs[0].set_title = TITLE[0]
        axs[1].set_title = TITLE[1]
        axs[2].set_title = TITLE[2]
        axs[3].set_title = TITLE[3]

    

    return figs, axs


def getLPM(x, y, jumps):

    
    #it calculates the index sequence to sort its rank in increasing sort
    min_to_max_ix = np.argsort(x)

 
    #it take a sample. The smaller the jumps variable is, more precise the approximations are
    sample_ix = min_to_max_ix[::jumps]

    #it generates a sample
    X = x[sample_ix]
    Y = y[sample_ix]


    #The marginal
    kde_prior = gaussian_kde(X)

    #the likelihood
    kde_lklihood = gaussian_kde(X[Y==1])

    #Prior
    M = np.average(y)

    #evaluando la probabilidad de la poblacion de valores 
    L = kde_lklihood.evaluate(x)
    P = kde_prior.evaluate(x)

    return L, P, M