import numpy as np
import scipy.signal as sg
import pandas as pd

def get_stft(signal):
    """
    A one-sided spectrogram and stft (for real signals)
    """
    assert np.isreal(np.all(signal)), "The signal should be real."

    N = len(signal)
    Nfft = 2*N
    window = sg.gaussian(Nfft, np.sqrt(Nfft/2/np.pi))
    window = window/ np.sum(window**2)
    sigaux = np.zeros((Nfft,))
    sigaux[0:N] = signal
        
    # Compute the STFT
    _, _, stft = sg.stft(sigaux,
                        window=window, 
                        nperseg=Nfft, 
                        noverlap=Nfft-1, 
                        return_onesided=True
                        )
    stft = stft[:,0:N]
    return stft

def invert_stft(stft,mask=None):
    """ Invert STFT computed with the function get_stft(...) 
    A filtering mask can be provided.
    """

    if mask is None:
        np.ones_like(stft)

    N = stft.shape[1]
    Nfft = 2*N
    window = sg.gaussian(Nfft, np.sqrt(Nfft/2/np.pi))
    window = window/window.sum()
    stftaux = np.zeros((stft.shape[0],Nfft), dtype=complex)
    stftaux[:,0:N] = stft*mask.astype(float)
        
    # Compute the STFT
    t, xr = sg.istft(stftaux, window=window, nperseg=Nfft, noverlap=Nfft-1)
    xr = xr[0:N]
    return xr


def spectrogram_thresholding(signal, coeff, fun='hard'):
    
    stft= get_stft(signal)
    gamma = np.median(np.abs(np.real(stft)))/0.6745
    thr = coeff*np.sqrt(2)*gamma

    mask = np.abs(stft)
    
    if fun == 'hard':
        mask[mask<thr] = 0
        mask[mask>=thr] = 1

    if fun == 'soft':
        mask[mask<=thr] = 0
        mask[mask>thr] = mask[mask>thr]-(thr**2/mask[mask>thr])
        mask = mask / np.abs(stft)

    xr = invert_stft(stft,mask=mask)

    return xr

def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values - np.mean(total.values)