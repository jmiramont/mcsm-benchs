import numpy as np
import scipy.signal as sg

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