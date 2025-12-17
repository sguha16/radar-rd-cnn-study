# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 15:19:54 2025

@author: uig67136
"""
#R=Range=30m
#v=velocity 10 m/s
#amp=amplitude 1 
#fc=carrier frequency 77 GHz
#B=Bandwidth 150 MHz
#T_chirp= time period of a chirp = 50 microsec
#N_samples=no of samplesfor 1 chirp in time domain (fast time) =64
#N_chirps= no of chirps

import numpy as np
from skimage.transform import resize

def generate_single_rd_map(targets,
                           fc=77e9, B=150e6, T_chirp=50e-6,
                           N_samples=64, N_chirps=32, noise_std=0.5):
    """
    Generate a single synthetic Range-Doppler map (16x16) for one target.
    targets: list of dicts [{'R':..., 'v':..., 'amp':...}, ...]

    Returns:
        RD_map_cnn: np.array of shape (1,16,16) ready for CNN
    """
    c = 3e8  # speed of light

    # Time axes
    t_fast = np.linspace(0, T_chirp, N_samples)#0:64:T_chirp-array
    t_slow = np.linspace(0, N_chirps*T_chirp, N_chirps)#0:32:32*50e^-6--array

    #initialize beat signal (matrix)
    beat_signal = np.zeros((N_samples, N_chirps), dtype=complex)

    # Beat signal
    for tgt in targets:
        R, v, amp = tgt['R'], tgt['v'], tgt['amp']
        f_range = 2*B*R/(c*T_chirp)# fbeat= k*tau = (B/T_chirp)*(2R/c)
        f_doppler = 2*v*fc/c
        beat_signal += amp * np.exp(1j*2*np.pi*(f_range*t_fast[:,None] + f_doppler*t_slow[None,:]))
        # s= A exp(j2*pi*f*t), ft-> range component, doppler component
        #dim(t_fast)+dim(t_slow)
        
    #adding noise to beat signal before 2d fft
    # --- add complex receiver noise ---
    noise = (np.random.normal(0, noise_std, beat_signal.shape) +
             1j*np.random.normal(0, noise_std, beat_signal.shape))
    beat_signal += noise

    # Range-Doppler map
    RD_range = np.fft.fft(beat_signal, axis=0)#fft along fast time axis --> range * slow time
    RD_map = np.fft.fft(RD_range, axis=1)#fft along slow time axis--> range* doppler
    RD_map_mag = np.abs(RD_map)#taking only mag--ignoring phase
    RD_map_mag /= np.max(RD_map_mag)#normalize = mag/. max (mag) entire map

    # Resize to 16x16 for CNN
    #RD_map_cnn = resize(RD_map_mag, (16,16), mode='reflect', anti_aliasing=True)#interpolating into 16*16
    #RD_map_cnn = RD_map_cnn[np.newaxis, :, :]  # shape (1,16,16)
    RD_map_cnn = RD_map_mag[np.newaxis, :, :]  # shape (1,16,16)
    print("RD_map_cnn shape",np.shape(RD_map_cnn))

    return RD_map_cnn


def generate_dataset(N_samples=100, R_range=(5,50), v_range=(-20,20)):
    """
    Generate multiple synthetic RD maps and labels.
    Returns:
        X: np.array of shape (N_samples,1,16,16)
        y: np.array of integer labels (dummy for now)
    """
    X_list = []
    y_list = []
    true_velocities=[]
    for i in range(N_samples):
        
        n_targets = np.random.randint(1, 5)#every Range Dopp map can have between 1 to 3 targets
        #n_targets=1#signle target test
        targets = []
        label=[]#array to be filled for 1 RD map
        RD_vel_arr=[]
        for _ in range(n_targets):
            tgt = {
                'R': np.random.uniform(10, 50),     # range 10-50 m
                'v': np.random.uniform(0, 10),      # velocity 0-20 m/s
                'amp': np.random.uniform(0.5, 1.0)  # simulate RCS variation
                }
            if(tgt['v']<0.5):
                label.append(0)#stationary object
            elif(tgt['v']<5):
                label.append(1)#pedestrian
            else:
                label.append(2)#vehicle
            targets.append(tgt)#tgt is a dictionary-tgt['R'], tgt['V']
            RD_vel_arr.append(tgt['v'])#real velocities of each RD map

    
        #--calling RD map generation func
        rd_map = generate_single_rd_map(targets)
        X_list.append(rd_map)
        y_list.append(np.max(label))#label has all labels of this RD map appended-max of this
        true_velocities.append(max(RD_vel_arr))#real velocities
    X = np.stack(X_list, axis=0)  # (N_samples,1,16,16)
    y = np.array(y_list, dtype=np.int64)
    return X, y,true_velocities


# Optional test
if __name__ == "__main__":
    X, y,vel = generate_dataset(100)#generate 30 sets of 16*16 RD maps with ranges and velocities as in function
    print("X shape:", X.shape)
    print("y:", y)
    np.save("C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Raw/X_synth.npy", X)#(30,16,16) data
    np.save("C:/Users/uig67136/.spyder-py3/python scripts/RadarClassification/Data/Raw/Y_synth.npy", y)#(30,) corresponding labels
