import matplotlib.pyplot as plt
import seaborn as sns

import glob
import argparse
import warnings
import pickle

import numpy as np
import pandas as pd

from gatspy import periodic


def _make_path(datadir, ra_dir, dec_dir, fname):
    f = datadir + "testLCdir/" + ra_dir + "/" + dec_dir + "/" + fname
    return f

def read_lc(fpath):
    d = np.loadtxt(fpath, skiprows=2)
    if len(d) == 0:
        return None
    if len(d.shape) < 2:
        return None
    ra = d[:, 1]
    dec = d[:, 2]
    mjd = d[:, 3]
    objctype = d[:, 4]
    objc_rowc = d[:, 5]
    objc_colc = d[:, 6]
    nchild = d[:, 7]
    rowc = d[:, 8:13]
    rowcerr = d[:, 13:18]
    colc = d[:, 18:23]
    colcerr = d[:, 23:28]
    psfmag = d[:, 28:33]
    psfmagerr = d[:, 33:38]
    modelmag = d[:, 38:43]
    modelmagerr = d[:, 43:48]
    tai = d[:, 48:53]
    taifrac = d[:, 53:58]
    airmass = d[:, 58:63]
    psf_fwhm = d[:, 63:68]
    skyflux = d[:, 68:73]

    # get out only data points that are real:
    mask_u = (psfmag[:,0] >= 13) & (psfmag[:,0] <= 23)
    mask_g = (psfmag[:,1] >= 13) & (psfmag[:,1] <= 23)
    mask_r = (psfmag[:,2] >= 13) & (psfmag[:,2] <= 23)
    mask_i = (psfmag[:,3] >= 13) & (psfmag[:,3] <= 23)
    mask_z = (psfmag[:,4] >= 13) & (psfmag[:,4] <= 23)


    data_dict = {"ra": ra, "dec":dec,  "tai_u": tai[mask_u,0]+taifrac[mask_u, 0],
                 "tai_g": tai[mask_g,0]+taifrac[mask_g, 0],  
                 "tai_r": tai[mask_r,0]+taifrac[mask_r, 0],   
                 "tai_i": tai[mask_i,0]+taifrac[mask_i, 0],   
                 "tai_z": tai[mask_z,0]+taifrac[mask_z, 0],    
                 "u":psfmag[mask_u,0], "g":psfmag[mask_g,1], "r":psfmag[mask_r,2], "i":psfmag[mask_i,3], "z":psfmag[mask_z,4],
                 "u_err":psfmagerr[mask_u,0], "g_err":psfmagerr[mask_g,1], "r_err":psfmagerr[mask_r,2], "i_err":psfmagerr[mask_i,3], 
                 "z_err":psfmagerr[mask_z,4]}
    
    data_dict = calculate_variability_measures(data_dict)

    return data_dict


def calculate_variability_measures(d):
    """
    Calculate median, mean and rms of each band in the data set.

    Parameters
    ----------
    d : dict
        Dictionary with the magnitudes

    """

    if len(d["u"]) > 1:
        d["mean_u"] = np.mean(d["u"])
        d["var_u"] = np.var(d["u"])
        d["rms_u"] = 0.7413*(np.percentile(d["u"], 75)-np.percentile(d["u"], 25))
        d["med_u"] = np.median(d["u"])

    if len(d["g"]) > 1:
        d["mean_g"] = np.mean(d["g"])
        d["var_g"] = np.var(d["g"])
        d["rms_g"] = 0.7413*(np.percentile(d["g"], 75)-np.percentile(d["g"], 25))
        d["med_g"] = np.median(d["g"])


    if len(d["r"]) > 1:
        d["mean_r"] = np.mean(d["r"])
        d["var_r"] = np.var(d["r"])
        d["rms_r"] = 0.7413*(np.percentile(d["r"], 75)-np.percentile(d["r"], 25))
        d["med_r"] = np.median(d["r"])

    if len(d["i"]) > 1:
        d["mean_i"] = np.mean(d["i"])
        d["var_i"] = np.var(d["i"])
        d["rms_i"] = 0.7413*(np.percentile(d["i"], 75)-np.percentile(d["i"], 25))
        d["med_i"] = np.median(d["i"])

    if len(d["z"]) > 1:
        d["mean_z"] = np.mean(d["z"])
        d["var_z"] = np.var(d["z"])
        d["rms_z"] = 0.7413*(np.percentile(d["z"], 75)-np.percentile(d["z"], 25))
        d["med_z"] = np.median(d["z"])

    return d


def plot_single_filter_psd(tai, mag, magerr, periods, scores, best_period, tfit, magfit, filename):
 
    """
    Plot the time series, Lomb-Scargle periodogram and folded phase curve at the 
    best-fit period for a given Lomb-Scargle periodogram.
    
    Parameters
    ----------
    
    tai : numpy.ndarray
        Array with time stamps where observations were taken
    
    mag : numpy.ndarray
        Array with magnitudes of observations corresponding to time stamps in `tai`
    
    magerr : numpy.ndarray
        Array with magnitude uncertainties for `mag`
    
    periods : numpy.ndarray
        An array with possible periods
    
    scores : numpy.ndarray
        The Lomb-Scargle periodogram corresponding to `periods`
    
    best_period : float
        The best-fit period 

    tfit : numpy.ndarray
        An array of time stamps for model values

    magfit : numpy.ndarray
        Values of the best-fit model at the time stamps given in `tfit`
    
    filename : str
        Full path plus root string that will be used to store 
        the figure to disk
    
    
    """

    # set up figure and axes objects
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

    # first figure: observed time series
    ax1.errorbar(tai, mag, yerr=magerr, fmt="o", markersize=5, c="black")

    # invert y-axis because magnitudes are stupid, and this way 
    # brighter things go *up* in the figure
    ax1.invert_yaxis()
    ax1.set_xlabel("TAI")
    ax1.set_ylabel("Magnitude")
    
    # second subplot shows the Lomb-Scargle periodogram
    ax2.plot(periods, scores)

    # set labels
    ax2.set(xlabel='period (days)', ylabel='Lomb Scargle Power',
           xlim=(periods[0], periods[-1]), ylim=(0, np.max(scores)*1.1))

    # set x-scale to logarithmic because I'm used to looking at periodograms this way
    ax2.set_xscale("log")

    # plot a black line with the best-fit period
    ax2.vlines(best_period, 0, np.max(scores)*1.1, lw=3, linestyle="dashed", color="black")
    ax2.set_title("P = " + str(best_period))
    

    # get the phase information for the real time stamps 
    # and the time stamps used to generate the model
    phase = (tai / best_period) % 1
    phasefit = (tfit / best_period)

 
    # third plot: data and best-fit model folded on the best-fit period
    ax3.errorbar(phase, mag, magerr, fmt='o')
    ax3.plot(phasefit, magfit, '-', color='gray')
    ax3.set(xlabel='phase', ylabel='r magnitude')
    ax3.invert_yaxis()

    fig.tight_layout()
   
    # save figure to file and close it
    fig.savefig(filename + ".pdf", format="pdf")
    plt.close()

    return


def plot_multi_filter_psd(time, mag, magerr, filts, periods, scores, best_period, tfit, magfit, filename):

    """
    Plot the time series, Lomb-Scargle periodogram and folded phase curve at the 
    best-fit period for a given Lomb-Scargle periodogram.
    
    Parameters
    ----------
    
    tai : numpy.ndarray
        Array with time stamps where observations were taken
    
    mag : numpy.ndarray
        Array with magnitudes of observations corresponding to time stamps in `tai`
    
    magerr : numpy.ndarray
        Array with magnitude uncertainties for `mag`
    
    filts : numpy.ndarray
        For each data point in `tai`, `mag` and `magerr`, which filter that data point 
        was taken with

    periods : numpy.ndarray
        An array with possible periods
    
    scores : numpy.ndarray
        The Lomb-Scargle periodogram corresponding to `periods`
    
    best_period : float
        The best-fit period 

    tfit : numpy.ndarray
        An array of time stamps for model values

    magfit : numpy.ndarray
        Values of the best-fit model at the time stamps given in `tfit`
    
    filename : str
        Full path plus root string that will be used to store 
        the figure to disk
    
    
    """

    # set up figure and axes objects
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))


    filttype = np.array(list("ugriz"))
    for f in filttype:
        ax1.errorbar(time[filts == f], mag[filts == f], yerr=magerr[filts == f], fmt="o", markersize=5, label=f)


    # invert y-axis because magnitudes are stupid, and this way 
    # brighter things go *up* in the figure
    ax1.invert_yaxis()
    ax1.set_xlabel("TAI")
    ax1.set_ylabel("Magnitude")
    ax1.legend()
   
    # second subplot shows the Lomb-Scargle periodogram
    ax2.plot(periods, scores)

    # set labels
    ax2.set(xlabel='period (days)', ylabel='Lomb Scargle Power',
           xlim=(periods[0], periods[-1]), ylim=(0, np.max(scores)*1.1))

    # set x-scale to logarithmic because I'm used to looking at periodograms this way
    ax2.set_xscale("log")

    # plot a black line with the best-fit period
    ax2.vlines(best_period, 0, np.max(scores)*1.1, lw=3, linestyle="dashed", color="black")
    ax2.set_title("P = " + str(best_period))
   

    # get the phase information for the real time stamps 
    # and the time stamps used to generate the model
    phase = (time / best_period) % 1
    phasefit = (tfit / best_period)



    for i, f in enumerate(filttype):
            m = (filts == f)
            errorbar = ax3.errorbar(phase[m], mag[m], magerr[m], fmt='.')
            ax3.plot(phasefit, magfit[i], color=errorbar.lines[0].get_color())


    # third plot: data and best-fit model folded on the best-fit period
    #ax3.errorbar(phase, mag, magerr, fmt='o')
    #ax3.plot(phasefit, magfit, '-', color='gray')
    ax3.set(xlabel='phase', ylabel='r magnitude')
    ax3.invert_yaxis()

    fig.tight_layout()
  
    # save figure to file and close it
    fig.savefig(filename + ".pdf", format="pdf")
    plt.close()

    return


def single_filter_periodogram(d, pmin=1.0, filt="r", datadir="./"):
    """
    Run a lomb-scargle periodogram on a single band.

    Parameters
    ----------
    d : dict
       A dictionary with a single data set

    p_min : float
       Minimum period for the Lomb-Scargle periodogram.

    filt : str, {"u", "g", "r", "i", "z"}
       The filter to run the periodogram in
    
    datadir : str
       The directory path where to store output

    """
    # list of possible filters
    ft = np.array(["u", "g", "r", "i", "z"])

    # index for the TAI array to get out the right 
    # observation times for the correct filter
    idx = int(np.argwhere(ft == filt))

    # get out time array, magnitude and magnitude error
    # for the correct filter
    tai = d["tai_" + filt]
    mag = d[filt]
    magerr = d[filt + "_err"]

    # set the maximum time scale to be searched to the 
    # baseline of the time series
    pmax = (np.max(tai)-np.min(tai))

    # instantiate the LombScargle periodogram class
    model = periodic.LombScargle(fit_period=True)
    # set the range of periods to search over

    model.optimizer.period_range = (pmin, pmax)

    # fit the model to the data
    model.fit(tai, mag, magerr)
  
    # what's the best-fit period?
    period = model.best_period

    # set up a grid of periods
    periods = np.linspace(pmin, pmax, 10000)
   
    # calculate LSP over a grid of periods: 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = model.score(periods)


     # make an array of time stamps within one period
    tfit = np.linspace(0, period, 1000)

    # predict the model within that period
    magfit = model.predict(tfit)

    # plot the result
    froot =  d["filename"].split("/")[-1][:-4]
    froot = "./lombscargle/" + froot
    plot_single_filter_psd(tai, mag, magerr, periods, scores, model.best_period, tfit, magfit, froot) 

    # store some output
    with open(froot + "_f=" + filt + "_log.txt", "w") as f:
       f.write("Filter: " + filt + "\n")
       f.write("best-fit period: " + str(period) + "\n")

    np.savetxt(froot + "_f=" + filt + "_lsp.txt", np.array([periods, scores]).T) 
    
    return

def multi_filter_periodogram(d, pmin=1.0, datadir="./"):
    # get data out of dictionary in a useful format
    utai = d["tai_u"]
    gtai = d["tai_g"]
    rtai = d["tai_r"]
    itai = d["tai_i"]
    ztai = d["tai_z"]

    umag = d["u"]
    gmag = d["g"]
    rmag = d["r"]
    imag = d["i"]
    zmag = d["z"]
    
    ukey = np.array(["u" for u in umag])
    gkey = np.array(["g" for u in gmag])
    rkey = np.array(["r" for u in rmag])
    ikey = np.array(["i" for u in imag])
    zkey = np.array(["z" for u in zmag])
    
    uerr = d["u_err"]
    gerr = d["g_err"]
    rerr = d["r_err"]
    ierr = d["i_err"]
    zerr = d["z_err"]

    # stack all the necessary measurements and filters
    mag = np.hstack([umag, gmag, rmag, imag, zmag])
    magerr = np.hstack([uerr, gerr, rerr, ierr, zerr])
    filters = np.hstack([ukey, gkey, rkey, ikey, zkey])
    time = np.hstack([utai, gtai, rtai, itai, ztai])


    pmax = (np.max(time)-np.min(time))
    
    # set up and fit multi-band Lomb-Scargle periodogram
    model = periodic.LombScargleMultiband(fit_period=True)
    model.optimizer.period_range = (pmin, pmax)
    model.fit(time, mag, magerr, filts=filters)
    print("Best period estimate: " + str(model.best_period))
   
    # what's the best-fit period?
    period = model.best_period

    # set up a grid of periods
    periods = np.linspace(pmin, pmax, 10000)
   
    # calculate LSP over a grid of periods: 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = model.score(periods)


     # make an array of time stamps within one period
    tfit = np.linspace(0, period, 1000)

    filtsfit = np.array(list('ugriz'))[:, np.newaxis]
    magfit = model.predict(tfit, filts=filtsfit)


    # plot the result
    froot =  d["filename"].split("/")[-1][:-4]
    froot = "./lombscargle/" + froot + "_multifilter"
    plot_multi_filter_psd(time, mag, magerr, filters, periods, scores, model.best_period, tfit, magfit, froot)

    # store some output
    with open(froot + "_log.txt", "w") as f:
       f.write("Combined model for all filters \n")
       f.write("best-fit period: " + str(period) + "\n")

    np.savetxt(froot + "_lsp.txt", np.array([periods, scores]).T)

    return



def process_lightcurves(datadir):
    # read the file with all the background information
    # including the path information for the individual 
    # CSV files with light curves
    lst = pd.read_csv(datadir+"LC_dirs_fils_new.lst", sep='\s+', skiprows=0, index_col=0)
    

    print("minimum number of data points: " + str(ndata))
    print("maximum RMS amplitude: " + str(lim_rms))
    print("maximum magnitude to consider (faint end): " + str(max_mag))


    # get out the light curves we're interested in 
    lst_new = lst[(lst["ndata"] >= ndata) & (lst["rms_r"] <= lim_rms) & (lst["med_r"] <= max_mag)]

    print("There are %i entries that match the criteria: " + str(len(lst_new)))
   
    data_all = []
    # loop over all light curves:
    for i in lst_new.index:
        # sub-dir named after RA
        ra_dir = lst_new.loc[i, "RAdir"]
        # sub-dir named after Dec
        dec_dir = lst_new.loc[i, "DecDir"]
        # actual name of the file
        fname = lst_new.loc[i, "LCfilename"]

        # put together the path
        f = _make_path(datadir, ra_dir, dec_dir, fname)

        # read the data and return as dictionary
        data = read_lc(f)
    
        # if there's actual data in the file, 
        # check whether it contains at least 50 
        # data points, and if it does, include in 
        # data set
        data["filename"] = f
        data_all.append(data)
    
    print("Finished reading all light curves.")

    print("There are %i light curves with at least %i data points and r-band rms < 0.1."%(len(data_all), ndata))
  
    return data_all


def main():

    if procdata:
        data_all = process_lightcurves(datadir)
        with open(datadir+"processed_lcs.dat", "wb") as f:
            pickle.dump(data_all, f)


    else:
        with open(datadir+"processed_lcs.dat", "rb") as f:
            data_all = pickle.load(f)


    for d in data_all:
        if single:
            single_filter_periodogram(d, pmin=pmin, filt=filt, datadir=datadir)
        else:
            multi_filter_periodogram(d, pmin=pmin,  datadir=datadir)
    return


# run this code if script is executed from command line
if __name__ == "__main__":

    # set up command line argument parser and arguments
    parser = argparse.ArgumentParser("Lomb-Scargle Periodogram for SDSS Data")
    parser.add_argument("-d", "--dir", action="store", dest="datadir", required=False, 
                        default="/Users/danielahuppenkothen/work/data/Stripe82/",
                        help="The path to the root directory with all the data.")

    parser.add_argument("-n", "--ndata", action="store", dest="ndata", required=False,
                        default=50, type=int, help="The minimum number of data points to consider. Default is 50")


    parser.add_argument("-s", "--single", action="store", dest="single", required=False,
                        default=True, type=bool, help="Run on single filter (True, default) or on multiple filters simultaneously?")

    parser.add_argument("-f", "--filter", action="store", dest="filter", required=False,  
                        default="r", type=str, help="If single=True, this is used to decide which filter to run the periodogram on. Default is r-band.")

    parser.add_argument("--pmin", action="store", dest="pmin", required=False, default=1.0, type=float, help="Smallest period to search. Default is 1 day.")

    parser.add_argument("--max-mag", action="store", dest="max_mag", required=False, default=20, type=float, help="Largest magnitude (faintest objects) to consider. Default is 20.")

    parser.add_argument("--lim-rms", action="store", dest="lim_rms", required=False, default=0.01, type=float, help="Upper limit to the rms in the light curve to consider. Default is 0.01.")

    parser.add_argument("-p", "--process-data", action="store", dest="procdata", required=False, 
                        default=False, type=bool, help="If True, process the data. If False, read from file.")

    # read out command line arguments
    clargs = parser.parse_args()

    # minimum number of data points in light curve to keep
    ndata = clargs.ndata

    # directory with the data
    datadir = clargs.datadir

    # whcih filter to use, if single is True
    filt = clargs.filter

    # if True, run on single filter; if False, do multi-band periodogram
    single = clargs.single

    # minimum period to search with LSP
    pmin = clargs.pmin    

    # maximum median magnitude to consider
    max_mag = clargs.max_mag

    # upper limit to the rms variability to consider
    lim_rms = clargs.lim_rms

    # process the data from scratch or load from file?
    procdata = clargs.procdata


    main()
