import h5py
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pathlib import Path

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from scipy.io.wavfile import write
from gwpy.plot import Plot

def make_wavable(array):
    """Make a numpy array ready to be saved as a WAV file.
    This entails turning it into integers, normalizing them 
    to their highest attainable value.
    """
    return (array / array.max() * np.iinfo(np.int16).max).astype(np.int16)

DATA_PATH = Path(__file__).parent
SOUNDS_PATH = Path(__file__).parent.parent / 'sounds'
FIGURE_PATH = Path(__file__).parent.parent / 'presentation' / 'figures'

# downloaded from https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW170817/v3
# it is the L1, 4kHz, 4096 second dataset
f = h5py.File(DATA_PATH / 'L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5')
strain = f['strain']['Strain'][:]
duration = f['meta']['Duration'][()]
gps_start = f['meta']['GPSstart'][()]
srate = 4096

gps = event_gps('GW170817')

around_evt = slice(
    int(( gps - gps_start - 30 ) * srate),
    int(( gps - gps_start + 2 ) * srate),
)

around_evt_short = slice(
    int(( gps - gps_start - 8 ) * srate),
    int(( gps - gps_start + 2 ) * srate),
)

ldata = TimeSeries(
    data=strain, 
    sample_rate=srate * u.Hz, 
    t0=gps_start
)

# plots and sounds for the unfiltered LIGO data
write(SOUNDS_PATH / "1-bare.wav", srate, make_wavable(ldata.bandpass(20, 1200)[around_evt_short]))
ldata[around_evt].plot()
plt.ylabel('Strain [dimensionless]')
plt.title('LIGO Livingston data')
plt.savefig(FIGURE_PATH / 'bare.pdf')
plt.close()

# plot for the amplitude spectral density
asd = ldata.asd(fftlength=4, method='median')
asd.plot()
plt.ylabel('Amplitude Spectral Density [Hz$^{1/2}$]')
plt.xlim(10, 1700)
plt.ylim(1e-24, 1e-20)
plt.savefig(FIGURE_PATH / 'asd.pdf')
plt.close()

# plots and sounds for the whitened, bandpassed data
ldata = ldata.whiten(asd=asd)
ldata = ldata.bandpass(30, 500)
write(SOUNDS_PATH / "2-whitened.wav", srate, make_wavable(ldata[around_evt_short]))
ldata[around_evt].plot()
plt.title('LIGO Livingston data')
plt.ylabel('Whitened strain [dimensionless]')
plt.savefig(FIGURE_PATH / 'whitened.pdf')
plt.close()

# plot the Q-transform of the data
spectrum = ldata.q_transform(frange=(30, 500), outseg=(gps-30, gps+2), qrange=(100, 110), logf=True)
plot = spectrum.plot()
plot.colorbar(label="Normalised energy")
plt.yscale('log')
plt.savefig(FIGURE_PATH / 'q_transform.pdf')
plt.close()

# theoretical waveform for this event
# as it is approaching the Earth
hp, hc = get_td_waveform(
    mass1=1.46, 
    mass2=1.27, 
    lambda1=400, 
    lambda2=400, 
    distance=40,
    approximant='TEOBResumS',
    delta_t=1/srate,
    f_lower=35,
)

hp.start_time += gps
hc.start_time += gps

livingston = Detector('L1')

# NGC 4993 coordinates
declination = -0.40812585
right_ascension = 3.44613079
polarization = 0.

# projection of the bare wave onto our detector
signal = livingston.project_wave(
    hp, 
    hc, 
    right_ascension, 
    declination, 
    polarization
)

# whitening the projection
gwpy_signal = TimeSeries.from_pycbc(signal).whiten(asd=asd)

Plot(ldata[around_evt], gwpy_signal)
plt.xlim(gps-30, gps+2)
plt.ylabel('Whitened strain [dimensionless]')

plt.savefig(FIGURE_PATH / 'true_signal.pdf')

write(SOUNDS_PATH / "3-BNS-signal.wav", 
    srate, 
    make_wavable(signal.numpy()[-len(signal)//8:]))

plt.xlim(gps-1.5, gps-1)
plt.savefig(FIGURE_PATH / 'true_signal_zoomed.pdf')
plt.close()
