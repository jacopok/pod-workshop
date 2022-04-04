from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import h5py
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

from GW170817 import DATA_PATH, SOUNDS_PATH, FIGURE_PATH, make_wavable, srate

# downloaded from https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3
# it is the H1, 4kHz, 32 second dataset 
f = h5py.File(DATA_PATH / 'H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5')
gps_start2 = f['meta']['GPSstart'][()]
strain_bbh = f['strain']['Strain'][:]

gps2 = event_gps('GW150914')

hdata = TimeSeries(
    data=strain_bbh, 
    sample_rate=srate * u.Hz, 
    t0=gps_start2
)
hdata = hdata.whiten(
    asd=hdata.asd(fftlength=4, method='median')
    ).crop(gps2-2, gps2+1)
hdata.plot()
plt.xlim(gps2-.3, gps2+.2)
plt.ylabel('Whitened strain [dimensionless]')
plt.title('GW150914')
plt.savefig(FIGURE_PATH / 'GW150914.pdf')
plt.close()

write(SOUNDS_PATH / "4-BBH-data.wav", 
    srate, 
    make_wavable(hdata))
