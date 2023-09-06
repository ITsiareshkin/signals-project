from scipy.signal import spectrogram, lfilter, freqz, tf2zpk, buttord, butter, filtfilt
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def DFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    f = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * f * n / N)
    return np.dot(M, x)

def impl(b, a, imp, N_imp, j):
    h = lfilter(b, a, imp)
    plt.figure(figsize=(5,3))
    plt.stem(np.arange(N_imp), h, basefmt=' ')
    plt.gca().set_xlabel('n')
    plt.gca().set_title('Impulsní odezva h[n]')

    plt.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.savefig('pdf/4.7.' + '%i' %(j+1) + '.pdf')

def nulyapoly(b, a, j):
    z, p, k = tf2zpk(b, a)
    plt.figure(figsize=(4,3.5))
    ang = np.linspace(0, 2*np.pi,100)
    plt.plot(np.cos(ang), np.sin(ang))

    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

    plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper left')

    plt.title('Filter %i' %(j+1))
    plt.tight_layout()
    plt.savefig('pdf/4.8.' + '%i' %(j+1) + '.pdf')

def modarg(b, a, j):
    w, H = freqz(b, a)
    _, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].plot(w / 2 / np.pi * data, np.abs(H))
    ax[0].set_xlabel('Frekvence [Hz]')
    ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
    ax[0].set_rasterized(True)

    ax[1].plot(w / 2 / np.pi * data, np.angle(H))
    ax[1].set_xlabel('Frekvence [Hz]')
    ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
    ax[1].set_rasterized(True)

    for ax1 in ax:
        ax1.grid(alpha=0.5, linestyle='--')

    _.suptitle('Filter %i' %(j+1))
    plt.tight_layout()
    plt.savefig('pdf/4.9.' + '%i' %(j+1) + '.pdf')

##################1##################
fs, data = sf.read('xtsiar00.wav')
# sek and vzor
print("ve vzorcích:", len(fs),"\nv sekundách:", len(fs)/data)
# min and max
print("min:", fs.min(),"\nmax:", fs.max());

t = np.arange(len(fs)) / data
plt.figure(figsize=(10,3)) # size
plt.plot(t, fs)

plt.gca().set_xlabel('t[s]')
plt.gca().set_title('Zvukový signál')

plt.tight_layout()
plt.savefig('pdf/4.1.pdf')

##################2##################
fs = fs - np.mean(fs) # stred
fs = fs / np.abs(fs).max() # normalizace

frame = []
for j in range(0, len(fs)):
    frame.append(fs[j * 512: 1024 + j * 512])

t = np.arange(len(frame[10])) / data
frm = frame[10] # best frame

plt.figure(figsize=(10,3)) # size
plt.plot(t, frame[10])

plt.gca().set_xlabel('t[s]')
plt.gca().set_title('Pěkný rámec')

plt.tight_layout()
plt.savefig('pdf/4.2.pdf')

##################3##################
###DFT####
X = DFT(frm)
fq = np.arange(0, data, data/len(X))
plt.figure(figsize=(10,3)) # size
plt.plot(fq[:len(X)//2], np.abs(X[:len(X)//2])) # F/2
plt.gca().set_xlabel('f[Hz]')
plt.gca().set_title('DFT')

plt.tight_layout()
plt.savefig('pdf/4.3.1.pdf')
###FFT####
# F = np.fft.fft(frm)
# plt.figure(figsize=(10,3)) # size
# plt.plot(fq[:len(F)//2], np.abs(F[:len(F)//2])) # F/2
# plt.gca().set_xlabel('Fq[Hz]')
# plt.gca().set_title('FFT')
#
# plt.tight_layout()
# plt.savefig('4.3.2.pdf')

###################4##################
fq, t, sgr = spectrogram(fs, data, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr + 1e-20)
plt.figure(figsize=(10,3))
plt.pcolormesh(t, fq, sgr_log, shading='auto')
plt.gca().set_xlabel('t[s]')
plt.gca().set_ylabel('f[Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('pdf/4.4.pdf')

###################5##################
fq0 = 720
fq1 = fq0 * 2 # 1440
fq2 = fq0 * 3 # 2160
fq3 = fq0 * 4 # 2880

print('f0 =', fq0)
print('f1 =', fq1)
print('f2 =', fq2)
print('f3 =', fq3)

###################6##################



arr = []
t = np.arange(len(fs)) / data

for i in range(len(fs)):
        arr.append(i / data)

out1 = np.cos(2 * np.pi * fq0 * np.array(arr))
out2 = np.cos(2 * np.pi * fq1 * np.array(arr))
out3 = np.cos(2 * np.pi * fq2 * np.array(arr))
out4 = np.cos(2 * np.pi * fq3 * np.array(arr))

out = out1+out2+out3+out4

fq, t, sgr = spectrogram(out, data, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr + 1e-20)
plt.figure(figsize=(10,3))
plt.pcolormesh(t, fq, sgr_log, shading='auto')
plt.gca().set_xlabel('t[s]')
plt.gca().set_ylabel('f[Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('pdf/4.6.pdf')

sf.write('audio/4cos.wav', out, data)

###################7##################
N_imp = 32
imp = [1, *np.zeros(N_imp-1)]
denominator = 1
numenator = 1
filt_num = 0

########### 1. p.z ###########
wp = [fq0-50, fq0+50]
ws = [fq0-10, fq0+10]

N, wn = buttord(wp, ws, 3, 40, False, data)
b, a = butter(N, wn, 'bandstop', False, 'ba', data)

####4.7,8,9####
impl(b, a, imp, N_imp, filt_num)
nulyapoly(b, a, filt_num)
modarg(b, a, filt_num)

filt_num+=1
denominator = np.convolve(denominator, a)
numenator = np.convolve(numenator, b)

########### 2. p.z ###########
wp = [fq1-50, fq1+50]
ws = [fq1-10, fq1+10]

N, wn = buttord(wp, ws, 3, 40, False, data)
b, a = butter(N, wn, 'bandstop', False, 'ba', data)

####4.7,8,9####
impl(b, a, imp, N_imp, filt_num)
nulyapoly(b, a, filt_num)
modarg(b, a, filt_num)

filt_num+=1
denominator = np.convolve(denominator, a)
numenator = np.convolve(numenator, b)

########### 3. p.z ###########
wp = [fq2-50, fq2+50]
ws = [fq2-10, fq2+10]

N, wn = buttord(wp, ws, 3, 40, False, data)
b, a = butter(N, wn, 'bandstop', False, 'ba', data)

####4.7,8,9####
impl(b, a, imp, N_imp, filt_num)
nulyapoly(b, a, filt_num)
modarg(b, a, filt_num)

filt_num+=1

denominator = np.convolve(denominator, a)
numenator = np.convolve(numenator, b)

########### 4. p.z ###########
wp = [fq3-50, fq3+50]
ws = [fq3-10, fq3+10]

N, wn = buttord(wp, ws, 3, 40, False, data)
b, a = butter(N, wn, 'bandstop', False, 'ba', data)

####4.7,8,9####
impl(b, a, imp, N_imp, filt_num)
nulyapoly(b, a, filt_num)
modarg(b, a, filt_num)

denominator = np.convolve(denominator, a)
numenator = np.convolve(numenator, b)

###################10##################
filtered = filtfilt(numenator, denominator, fs)

wavfile.write("audio/clean_bandstop.wav", data, (filtered * np.iinfo(np.int16).max).astype(np.int16))

f, t, sfgr = spectrogram(filtered, data, nperseg=1024, noverlap=512)
sfgr_log = 10 * np.log10(sfgr+1e-20)
plt.figure(figsize=(10,3))
plt.pcolormesh(t,f,sfgr_log, shading='auto')
plt.gca().set_title('Spektrogram vyfiltrovaného signálu')
plt.gca().set_xlabel('t[s]')
plt.gca().set_ylabel('f[Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('pdf/4.10.pdf')
