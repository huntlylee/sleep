# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 00:37:18 2021

file name: read and plot update insp detection.py

@author: liy45
"""



import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, butter, lfilter #find_peaks_cwt, general_gaussian, fftconvolve
import pandas as pd
import os
import tensorflow as tf
# from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
# from BaselineRemoval import BaselineRemoval
from pathlib import Path

from scipy import integrate, stats, signal, interpolate
from itertools import groupby
# from pybaselines.polynomial import imodpoly
from matplotlib.lines import Line2D      
import matplotlib.patches as mpatches
from moviepy.editor import * 



''' read signal'''

def get_signal_data(dur):
       
    f = adi.read_file(file_path)
    
    sig_dict = {}
    sr_dict = {}

    for ch in dict_ch.keys():
        
        sr_ch = f.channels[dict_ch[ch]-1].fs[record_id-1] 
        start_point = int(dur[0]*sr_ch)
        stop_point = int(dur[1]*sr_ch)
        
        sig_dict[ch] = f.channels[dict_ch[ch]-1].get_data(record_id, start_sample=start_point, stop_sample=stop_point)                
        sr_dict[ch] = sr_ch

    return sig_dict, sr_dict


''' high/low pass filters'''
        
def low_pass_filter(wavedata, sample_rate, cutoff=1): # stim signal
    
    # Filter requirements.   
    nyq = 0.5 * sample_rate  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic   
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, wavedata)
    return y

def highpass_filter(wavedata, sample_rate, cutoff=1, order=2):
    
    nyq = 0.5 * sample_rate  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)   
    y = signal.filtfilt(b, a, wavedata)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):   
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')    
    y = lfilter(b, a, data)
    return y

''' resample'''

def re_sample(sig, sr_old, sr_new, req_len = None):
    
    dur = len(sig)/sr_old

    f = interpolate.interp1d(x = np.arange(0, len(sig)), y = sig)
    if req_len is None:
        xnew = np.linspace(0, len(sig), round(dur*sr_new), endpoint=False)
    else:
        xnew = np.linspace(0, len(sig), req_len, endpoint=False)
    return f(xnew)

''' Cycle detection'''


def get_seq_len(c, thresh = 10000): # get the length of an inspiration effort (consecutive 1s)
    seqs = [(key, len(list(val))) for key, val in groupby(c)]
    seqs = [(key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)]
    # seqs = [(key, start, length), ...]
    return [[s[1], s[1] + s[2] - 1] for s in seqs if s[0] == 1 and s[2] > thresh]


def detect_cycle(based = 'belt c', smoothing = 0.3, t_shift =  0.1, min_distance = 1.5):
    
    ''' algorithm for insp phase:
        1. if based on belt: high pass fitler 0.1, then nadir to peak plus a small delay
        2. if based on pressure catheter: bandpass fitler 0.1-1, then negative values within a cycle
        3. if based on flow: high pass filter 0.1, then positive values within a cycle
    '''
    
    if based == 'belt c':
        sig = highpass_filter(sig_rip_chst, sample_rate, 0.1, 2)
    elif based == 'belt a':
        sig = highpass_filter(sig_rip_abd, sample_rate, 0.1, 2)
    elif based == 'belt 2':
        sig = highpass_filter( sig_rip_abd + sig_rip_chst , sample_rate, 0.1, 2)
    elif based == 'p v':
        sig = butter_bandpass_filter(sig_pv, lowcut=0.1, highcut=1, fs=sr_dict['ch_epi'], order = 2)
    elif based == 'p e':
        sig = butter_bandpass_filter(sig_epi, lowcut=0.1, highcut=1, fs=sr_dict['ch_epi'], order = 2)
    elif based == 'p 2':
        sig = butter_bandpass_filter(sig_pv + sig_epi, lowcut=0.1, highcut=1, fs=sr_dict['ch_epi'], order = 2)
    
    sig_f = pd.Series(sig).rolling(int(smoothing*sample_rate), min_periods=1).mean().shift(-int(smoothing*sample_rate/2)).fillna(0).to_numpy()
    max_ind = signal.argrelextrema(sig_f, np.greater, order = int(min_distance*sample_rate))[0]
    min_ind = signal.argrelextrema(sig_f, np.less, order = int(min_distance*sample_rate))[0]
    
    ins_start, ins_end, ex_end = [], [], []
    if 'belt' in based:
        for idx in zip(min_ind, min_ind[1:]):
            ins_start.append(idx[0])
            ex_end.append(idx[1]-1)
            a = np.where((max_ind >= ins_start) & (max_ind <= ex_end))
            if len(a[0])>0:
                ins_end.append(int(max_ind[a[0][0]] + t_shift*sample_rate))
            else:
                ins_end.append(np.nan)
        ins_start.append(min_ind[-1])
        ex_end.append(len(sig_f)-1)
        a = np.where((max_ind >= min_ind[-1]) & (max_ind < len(sig_f)))
        if len(a[0])>0:
            ins_end.append(int(max_ind[a[0][0]] + t_shift*sample_rate))
        else:
            ins_end.append(np.nan)
    elif 'p' in based:
        a = np.where(sig_f<=0, 1, 0)
        d = get_seq_len(a, thresh = int(sample_rate*min_distance/2))   
        for i in range(len(d)-1):
            ins_start.append(d[i][0])
            ex_end.append(d[i+1][0] - 1)
            ins_end.append(d[i][1])
        ins_start.append(d[-1][0])
        ins_end.append(d[-1][1])
        ex_end.append(len(sig_f)-1)    
    
    df_cycle = pd.DataFrame(list(zip(ins_start, ins_end, ex_end)), columns=['Ins start', 'Ins end', 'Exp end'])
    # df_cycle['Ex start'] = df_cycle['Ins end'] +1
    # return df_cycle.reindex(columns = ['Ins start', 'Ins end', 'Ex start', 'Ex end'])
    return df_cycle


def detect_ins_flow(sig_flow, smoothing = 0.1, min_dur = 0.8):   
    ma = int(smoothing*sample_rate)
    sig_f = pd.Series(sig_flow).rolling(ma, min_periods=1).mean().shift(-int(ma/2)).fillna(0).to_numpy()
    a = np.where(sig_f>0, 1, 0)
    d = get_seq_len(a, thresh = int(sample_rate*min_dur))    
    return d

def remeasure_cycle(df_cycle, force, smoothing = 0.1, min_insp_dur = 0.8, lock = True):
    df_cycle2 = df_cycle.copy()
    for row in range(len(df_cycle2)):
        # row = 1
        cyc_start = df_cycle2['Ins start'][row]
        cyc_end = df_cycle2['Exp end'][row]
        sig_temp = sig_flow[int(cyc_start): int(cyc_end)+1]
        d = detect_ins_flow(sig_temp, smoothing = smoothing, min_dur = min_insp_dur)           
        if len(d) >0 and (row+1 not in force):
            if not lock:
                df_cycle2.loc[row, 'Ins start'] = d[0][0] + cyc_start
            df_cycle2.loc[row, 'Ins end'] = d[0][1] + cyc_start         
        else :        
            sig_pressure_seg = sig_epi[int(cyc_start): int(cyc_end)+1]
            sig_p1 = pd.Series(sig_pressure_seg).rolling(int(smoothing*sample_rate), min_periods=1).mean().shift(-int(smoothing*sample_rate/2)).fillna(0).to_numpy()
            max_effort_t = np.argmin(sig_p1)
            df_cycle2.loc[row, 'Ins end'] = max_effort_t + cyc_start 
    
    df_cycle2['Exp end'] = df_cycle2['Ins start'].shift(-1)-1
    df_cycle2.loc[len(df_cycle2)-1, 'Exp end'] = len(sig_flow)-1
    df_cycle2['Exp start'] = df_cycle2['Ins end'] + 1
    df_cycle2['Dur'] = (df_cycle2['Exp end'] - df_cycle2['Ins start'])/sample_rate
    return df_cycle2


def manual_combine(df_breaths, c_list):           
        return df_breaths.drop(c_list, axis = 0).reset_index(drop = True)
    


'''measurement'''

def get_baseline_flow(row):
    return np.mean(sig_flow[int(row['Exp start']): int(row['Exp end'] + 1)])

def get_baseline_chst(row):
    return np.mean(sig_rip_chst[int(row['Exp start']): int(row['Exp end'] + 1)])

def get_baseline_abd(row):
    return np.mean(sig_rip_abd[int(row['Exp start']): int(row['Exp end'] + 1)])

def get_nadir_chst(row):
    return np.amin(sig_rip_chst[int(row['Exp start']): int(row['Exp end'] + 1)])

def get_baseline_velo(row):
    return np.mean(sig_pv[int(row['Exp start']): int(row['Exp end'] + 1)])

def get_baseline_epi(row):
    return np.mean(sig_epi[int(row['Exp start']): int(row['Exp end'] + 1)])

def get_measurement(df_cycle):
    
    df_breaths = df_cycle.copy()
    df_breaths['Exp flow mean'] = df_breaths.apply(get_baseline_flow, axis = 1)   
    df_breaths['Exp chst mean'] = df_breaths.apply(get_baseline_chst, axis = 1)    
    df_breaths['Exp abd mean'] = df_breaths.apply(get_baseline_abd, axis = 1)    
    df_breaths['Exp pv mean'] = df_breaths.apply(get_baseline_velo, axis = 1)    
    df_breaths['Exp pepi mean'] = df_breaths.apply(get_baseline_epi, axis = 1)

    for i in range(1, len(df_breaths)):

        df_breaths.at[i, 'Exp flow mean prev'] = df_breaths.iloc[i-1]['Exp flow mean']
        df_breaths.at[i, 'Exp chst mean prev'] = df_breaths.iloc[i-1]['Exp chst mean']
        df_breaths.at[i, 'Exp abd mean prev'] = df_breaths.iloc[i-1]['Exp abd mean']
          
    df_breaths.at[0, 'Exp flow mean prev'] = np.mean(sig_flow[0: df_breaths['Ins start'][0] - 1])
    df_breaths.at[0, 'Exp chst mean prev'] = np.mean(sig_rip_chst[0: df_breaths['Ins start'][0] - 1])    
    df_breaths.at[0, 'Exp abd mean prev'] = np.mean(sig_rip_abd[0: df_breaths['Ins start'][0] - 1])
    
    # cycle_div = df_in_ex['Ins start'].tolist()
    ''' def measures'''
    stim_mean = []
    stim_max = []
    
    pv_min_time = []
    pv_min = []
    pv_std = []
   
    pepi_min_time = []
    pepi_min = []
    pepi_std = []
   
    flow_max_time = []
    flow_max = []
    flow_mean = []
    flow_std = []
    cpap_mean = []
    stim_flow_max_time = []
    pepi_flow_max_time = []
    stim_pepi_min_time = []
    flow_pepi_min_time = []
    flow_median = []
    
    vimax = []
    vimean = []
    vit50 = []
    vi13mean = []
    tv = []
    
    for i in range(len(df_breaths)):
                    
        start_t = int(df_breaths['Ins start'][i])
        end_t = int(df_breaths['Exp end'][i] + 1)
        end_ins_t = int(df_breaths['Ins end'][i] + 1)
        
        stim_mean.append(np.mean(sig_stim_lfilt[start_t : end_t]))
        stim_max.append(np.amax(sig_stim_lfilt[start_t : end_t]))
        
        pv_min_time_temp = start_t + np.argmin(sig_pv[start_t : end_t]) 
        pv_min_time.append(pv_min_time_temp/sample_rate + t_start)
        pv_min.append(np.amin(sig_pv[start_t : end_t]))
        pv_std.append(np.std(sig_pv[start_t : end_t]))
        
        pepi_min_time_temp = start_t + np.argmin(sig_epi[start_t : end_t]) 
        pepi_min_time.append(pepi_min_time_temp/sample_rate + t_start)
        pepi_min.append(np.amin(sig_epi[start_t : end_t]))
        pepi_std.append(np.std(sig_epi[start_t : end_t]))

        flow_max_time_temp = start_t + np.argmax(sig_flow[start_t : end_t])
        flow_max_time.append(flow_max_time_temp/sample_rate + t_start)
        flow_max.append(np.amax(sig_flow[start_t : end_t]))
        flow_mean.append(np.mean(sig_flow[start_t : end_t]))
        flow_std.append(np.std(sig_flow[start_t : end_t]))
        
        cpap_mean.append(np.mean(sig_cpap[start_t : end_t]))
        
        stim_flow_max_time.append(sig_stim_lfilt[flow_max_time_temp])
        pepi_flow_max_time.append(sig_epi[flow_max_time_temp])
        stim_pepi_min_time.append(sig_stim_lfilt[pepi_min_time_temp])
        flow_pepi_min_time.append(sig_flow[pepi_min_time_temp])
        
        flow_median.append(np.median(sig_flow[start_t : end_t]))
        
        vimax.append(np.amax(sig_flow[start_t : end_ins_t]))
        vimean.append(np.mean(sig_flow[start_t: end_ins_t]))
        
        v_dur = end_ins_t - start_t 
        vi13mean.append(np.mean(sig_flow[ int(v_dur/3 + start_t) : int (end_ins_t - v_dur/3) + 1 ]))
        
        vit50.append(sig_flow[int(v_dur/2 + start_t)])
        tv.append(np.trapz(sig_flow[start_t : end_ins_t]/60, dx = 1/sample_rate))
        
    df_measure = pd.DataFrame(list(zip(stim_mean, stim_max, pv_min_time, pv_min, pv_std, pepi_min_time, pepi_min, pepi_std, flow_max_time, flow_max, flow_mean, flow_std, cpap_mean, 
                               stim_flow_max_time, pepi_flow_max_time, stim_pepi_min_time, flow_pepi_min_time, flow_median, vimax, vimean, vit50, vi13mean, tv)),
              columns=['Stim mean','Stim max', 'Pv min time', 'Pv min', 'Pv std', 'Pepi min time', 'Pepi min', 'Pepi std', 'Flow max time', 'Flow max unadj', 'Flow mean', 'Flow std', 'CPAP mean',
                       'Stim at max flow', 'Pepi at max flow', 'Stim at min pepi', 'Flow at min pepi', 'Flow median', 'Vimax', 'Vimean', 'Vit50', 'Vi13mean', 'Tidal volume'])
        
    return pd.concat([df_breaths, df_measure], axis=1)


def stim_mode(row):
    if row['Stim at max flow'] >= 0.001:
        val = 'ACS'
    # elif row['Stim at max flow'] >= 0.001: 
    #     val = 'ACS'
    else:
        val = 'No stim' 
    return val


def airway_status(row):   
    if row['Flow drop'] <= -0.2:
        val = 'FL'
    elif row['Flow max unadj'] <= 1.5 and row['Flow std'] < 1:
        val = 'Apnea'        
    elif (row['Flow max unadj'] > 18 or row['Flow std'] > 1) and row['Flow drop'] > -0.1:
        val = 'NFL'
    else:
        val = 'Other'
    return val


def airway_status_2(row):
    if row['Airway status'] == 'FL' and row['Airway status DL'] == 'FL':
        val = 'FL'
    elif row['Airway status'] == 'Apnea' or row['Airway status DL'] == 'Apnea':
        val = 'Apnea'       
    elif row['Airway status'] == 'NFL' and row['Airway status DL'] == 'NFL':
        val = 'NFL'
    else:
        val = 'Other'
    return val


def flow_limit_site(row):
    if row['Airway status'] != 'NFL':               
        if row['Pepi std'] < 1 and row['Pv std'] < 1: # no negative swing
            return 'E'  
        elif row['Pepi std'] >= 1 and row['Pv std'] >= 1:
            return 'V'          
        elif row['Pepi std'] >= 1 and row['Pv std'] < 1:        
            return 'O'        
        else:
           return 'U'
    else:
        return 'U'
    
def flow_limit_site_update(row):
    if row['Airway status final'] != 'NFL':
        if row['Pepi std'] < 1 and row['Pv std'] < 1: # no negative swing
            return 'E'  
        elif row['Pepi std'] >= 1 and row['Pv std'] >= 1:
            return 'V'          
        elif row['Pepi std'] >= 1 and row['Pv std'] < 1:        
            return 'O'        
        else:
           return 'U'
    else:
        return 'U'


def data_analysis(df, initial = True):
    
    z = np.polyfit(df['CPAP mean'], df['Flow median'], 1)
    p = np.poly1d(z) 
        
    df['Fit flow by CPAP'] = p(df['CPAP mean'])
    
    df['vi50 vimax percent'] = df['Vit50'] / df['Vimax']
    
    df['delta Pv'] = df['Pv min'] - df['Exp pv mean']
    df['delta Pepi'] = df['Pepi min'] - df['Exp pepi mean']
    
    df['Time diff'] = (df['Flow max time'] - df['Pepi min time']) * 1000
    
    df['Flow drop'] = df['Flow at min pepi']/df['Flow max unadj'] - 1
    df['Flow drop rate'] = df['Flow drop']/abs(df['Time diff'])*1000
    
    df['Stim mode'] = df.apply(stim_mode, axis=1)
    df['Stim mode review'] = ''
    df['Stim mode final'] = ''
    
    if initial:
        df['Airway status'] = df.apply(airway_status, axis=1)
    else:
        df['Airway status'] = df.apply(airway_status_2, axis=1)
        df['Flow limit site'] = df.apply(flow_limit_site, axis=1)
        df['Flow limit site review'] = ''
        df['NED review'] = ''
        df['Flow limit site final'] = ''
    
    df['Airway status review'] = ''
    df['Airway status final'] = ''      
    df['Flow max adj'] = df['Flow max unadj'] -  df['Fit flow by CPAP']    
    df['CPAP ceil'] = np.ceil(df['CPAP mean'])
    
    return df

'''
df_cycle = detect_cycle(based = 'p e', smoothing = 0.3, t_shift =  0.1, min_distance = 1.5)
df_cycle2 = remeasure_cycle(df_cycle, force=[], smoothing = 0.1, min_insp_dur = 0.7, lock = False)



#  combine two cycles
# df_cycle2 = manual_combine(df_cycle2, c_list = [6,8,10,14,17,19,21,23,25,27,29,31,33,35,37,43,53,58,62,66,69,71,73,75,77,79,81,83,85,87,89,92,94]) 

#  adjust time
b = [49]
df_cycle2.loc[[x - 1 for x in b], 'Ins start'] += 25

b = [57,61]
df_cycle2.loc[[x - 1 for x in b], 'Ins start'] -= 240
# df_cycle2['Dur'] = (df_cycle2['Ins end'] - df_cycle2['Ins start'] )/100

# b = [61]
df_cycle2.loc[[x - 1 for x in b], 'Ins start'] -= 45

# b = [41]
# df_cycle2.loc[[x - 1 for x in b], 'Ins end'] -= 100
# df_cycle2['Dur'] = (df_cycle2['Ins end'] - df_cycle2['Ins start'] )/100

# All
# df_cycle2.loc[:, 'Ins start'] -= 30
# df_cycle2['Dur'] = (df_cycle2['Ins end'] - df_cycle2['Ins start'] )/100

df_cycle2 = remeasure_cycle(df_cycle2, force=[57,61,62], smoothing = 0.1, min_insp_dur = 0.8, lock = True)

# def anchor_ins_end(b, anchor = 'p e', shift = 0.1):
    
#     if anchor == 'p e':
#         sig = sig_epi
#     elif anchor == 'p v':
#         sig = sig_pv
#     elif anchor == 'belt c':
#         sig = sig_rip_chst
#     elif anchor == 'belt a':
#         sig = sig_rip_abd    
#     for x in b:
#         cyc_start = int(df_cycle2['Ins start'][x-1])
#         cyc_end = int(df_cycle2['Exp end'][x-1])
#         sig_temp = sig[cyc_start, cyc_end+1]
#         if 'p' in anchor:
#             t = np.argmin(sig_temp) + cyc_start + shift*sample_rate
#         else:
#             t = np.argmax(sig_temp) + cyc_start + shift*sample_rate
    
#         df_cycle2.loc[x-1, 'Ins end'] = int(t)

# anchor_ins_end([57,61,62], anchor = 'p e', shift = 0.1)

# df_cycle2['Dur'] = (df_cycle2['Exp end'] - df_cycle2['Ins start'])/sample_rate

df_breaths = get_measurement(df_cycle2)
df_breaths = data_analysis(df_breaths, initial = True)
plot_multi_waves(df_breaths, start_t = t_start, review=False, save_fig = False)

with pd.ExcelWriter(save_path, mode='a', if_sheet_exists = 'replace') as writer: 
    df_breaths.to_excel(writer, index=False, sheet_name='R{} {}-{} insp remeasure'.format(record_id, t_start, t_end))   
'''


''' plot'''

def get_peak(wave_data, cycle_div, peak_type = 'h'):
    
    peak_loc = []
    peak_val = []
    if peak_type== 'h':
        for i in range(len(cycle_div)-1):
            peak_val.append(np.amax(wave_data[cycle_div[i]: cycle_div[i+1]]))
            peak_loc.append(cycle_div[i] + np.argmax(wave_data[cycle_div[i]: cycle_div[i+1]]))
        peak_val.append(np.amax(wave_data[cycle_div[-1]: ])) 
        peak_loc.append(cycle_div[-1] + np.argmax(wave_data[cycle_div[-1]:]))  
    elif peak_type== 'l':
        for i in range(len(cycle_div)-1):
            peak_val.append(np.amin(wave_data[cycle_div[i]: cycle_div[i+1]]))
            peak_loc.append(cycle_div[i] + np.argmin(wave_data[cycle_div[i]: cycle_div[i+1]]))
        peak_val.append(np.amin(wave_data[cycle_div[-1]: ])) 
        peak_loc.append(cycle_div[-1] + np.argmin(wave_data[cycle_div[-1]:])) 
    
    return np.array(peak_loc), np.array(peak_val)  

  

def adj_flow_to_volume(sig_flow_adj, df):

    sig_vol_int = np.zeros(len(sig_flow_adj))
    
    for i in range(len(df)):
        
        start_point = df['Ins start'][i]
        end_point = df['Exp end'][i]
        
        signal_temp = sig_flow_adj[start_point : end_point]
        
        x = np.linspace(0, len(signal_temp)/sample_rate, len(signal_temp))
        
        sig_vol_int[start_point : end_point] = integrate.cumtrapz(signal_temp, x, initial=0)
        
    if df['Ins start'][0] > 0:
        
        signal_temp = sig_flow_adj[: df['Ins start'][0]]
        
        x = np.linspace(0, len(signal_temp)/sample_rate, len(signal_temp))

        sig_vol_int[ : df['Ins start'][0]] = integrate.cumtrapz(signal_temp, x, initial=0)
        
    return sig_vol_int



def plot_wave(wave_data, sample_rate, cycle_div, ax, wave_color, title, start_t, lbl_list = None, markersize = 16, 
              show_div = True, show_peak = 'h', show_zero = False, zero_lvl=0, show_color = 'No', 
              show_num = False, num_size = None, num_lbl_list = None,
              show_rip_level = False, rip_arr=None):
   
    data = np.copy(wave_data)
    
    min_y = np.min(data)
    max_y = np.max(data)
 
    max_time = len(data)/sample_rate

    time_steps = np.linspace(0, max_time, len(data)) + start_t
    
    ax.plot(time_steps, data, color=wave_color)
    
    if show_peak != 'No': # 'No, 'h', 'l'
        peaks_loc, _ = get_peak(wave_data, cycle_div, peak_type = show_peak)
        ax.plot(peaks_loc/sample_rate + start_t, data[peaks_loc], "x", color = 'tomato', markersize = markersize)
    
    if show_zero:
        ax.hlines(zero_lvl, xmin = time_steps[0], xmax = time_steps[-1], colors="navy", linestyles = 'dashed', linewidth = 2)
    if show_div:
        for i in range(len(cycle_div)):
            start_t_temp = cycle_div[i]/sample_rate + start_t
            try:
                end_t_temp = cycle_div[i+1]/sample_rate + start_t
            except:
                end_t_temp = len(data)/sample_rate + start_t
            ax.axvline(x=start_t_temp, color='darkorange', linestyle='--', alpha = 0.5)
            if show_color == 'stim':
                if lbl_list[i] == 'GPN':
                    c = 'deepskyblue'
                elif lbl_list[i] == 'ACS':
                    c = 'aqua'  
                else:
                    c = 'white'
                ax.axvspan(start_t_temp, end_t_temp, color=c, alpha = 0.2)
            elif show_color == 'airway':
                if lbl_list[i] == 'Apnea':
                    c = 'lightcoral' 
                elif lbl_list[i] == 'FL' : 
                    c = 'khaki'
                elif lbl_list[i] == 'NFL' : 
                    c = 'lime'
                else:
                    c = 'white'
                ax.axvspan(start_t_temp, end_t_temp, color=c, alpha = 0.2)
            elif show_color == 'plot':
                c = 'dimgray' if lbl_list[i] == 'Remove' else 'white'
                ax.axvspan(start_t_temp, end_t_temp, color=c, alpha = 0.6)
            elif show_color == 'plot + airway':
                if lbl_list[i] == 'ApneaY':
                    c = 'lightcoral' 
                elif lbl_list[i] == 'FLY' : 
                    c = 'khaki'
                elif lbl_list[i] == 'NFLY' : 
                    c = 'lime'
                elif (lbl_list[i] == 'NFL'):
                    c = 'darkgreen'
                elif (lbl_list[i] == 'Dogleg') or (lbl_list[i] == 'Apnea'):
                    c = 'darkred'
                else:
                    c = 'dimgray' 
                ax.axvspan(start_t_temp, end_t_temp, color=c, alpha = 0.4)
            
            elif show_color == 'fls': # tab:blue, green, orange, red 
                if lbl_list[i] == 'FLV' or lbl_list[i] == 'ApneaV' or lbl_list[i] =='DoglegV' :
                    c = 'tab:purple'
                    alph = 0.8
                elif lbl_list[i] == 'FLVo' or lbl_list[i] == 'ApneaVo' or lbl_list[i] == 'DoglegVo':
                    c = 'tab:purple'
                    alph = 0.6
                elif lbl_list[i] == 'FLVt' or lbl_list[i] == 'ApneaVt' or lbl_list[i] == 'DoglegVt':
                    c = 'tab:purple'
                    alph = 0.4
                elif lbl_list[i] == 'FLVe' or lbl_list[i] == 'ApneaVe' or lbl_list[i] == 'DoglegVe':
                    c = 'tab:purple'
                    alph = 0.2
                elif 'FLO' in lbl_list[i]  or 'ApneaO' in lbl_list[i]  or 'DoglegO' in lbl_list[i]:
                    c = 'tab:cyan' 
                    alph = 0.8
                elif lbl_list[i] == 'FLT' or lbl_list[i] == 'ApneaT' or lbl_list[i] == 'DoglegT':
                    c = 'tab:olive'
                    alph = 0.8
                elif lbl_list[i] == 'FLTe' or lbl_list[i] == 'ApneaTe' or lbl_list[i] == 'DoglegTe':
                    c = 'tab:olive'
                    alph = 0.4
                elif lbl_list[i] == 'FLE' or lbl_list[i] == 'ApneaE' or lbl_list[i] == 'DoglegE':
                    c = 'tab:red' 
                    alph = 0.4
                else:
                    c = 'white'
                    alph = 0.6
                ax.axvspan(start_t_temp, end_t_temp, color=c, alpha = alph)
            if show_num:
                if num_lbl_list[i] == 'Combined':
                    c = 'deepskyblue'
                elif num_lbl_list[i] == 'ACS':
                    c = 'aqua'  
                else:
                    c = 'black'
                ax.annotate(str(i+1), xy=(((start_t_temp + end_t_temp)/2), min_y + (max_y - min_y)*0.1), xycoords='data', color = c,
                            ha='center', alpha=0.7, fontsize=num_size)                
            if show_rip_level:
                c = 'seagreen'
                ax.plot([start_t_temp, end_t_temp], [rip_arr[i], rip_arr[i]] , color=c, alpha = 0.6, linestyle='--')
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Amplitude")
    ax.annotate(title, xy=(0.96, 0.05), xycoords='axes fraction')
    # plt.show()        
    
def plot_multi_waves(df, start_t, review = False, save_fig = False):
    
    dur = len(sig_flow)/sample_rate    
    nrow = 7
    hw_rate = 0.8      
    fig_w = int(dur/4)
    fig_h = int(fig_w * hw_rate)   
    plt.rc('font', size=int(fig_w))    
    cycle_div = df['Ins start'].to_list()
    fig, ax = plt.subplots(nrows=nrow, ncols=1, sharex=True, figsize=(fig_w,fig_h))
    # Stim
    if not review:    
        plot_wave(sig_stim_lfilt*1000, sample_rate, cycle_div, ax=ax[0], wave_color = 'red', title = 'Stim', lbl_list = df['Stim mode'].tolist(),
                  start_t = start_t, show_div = True, show_peak = 'No', show_color = 'stim')            
    else:
        plot_wave(sig_stim_lfilt*1000, sample_rate, cycle_div, ax=ax[0], wave_color = 'red', title = 'Stim', lbl_list = df['Stim mode final'].tolist(),
                  start_t = start_t, show_div = True, show_peak = 'No', show_color = 'stim')       
    # Chest
    if review:
        plot_wave(sig_rip_chst, sample_rate, cycle_div, ax=ax[1], wave_color = 'red', title = 'Chest belt',  markersize=fig_w/2, 
                  start_t = start_t, show_div = True, show_peak = 'h', show_color = 'plot', show_zero = False, lbl_list = df['Airway status final'].tolist(),
                  show_num = False)
    else:
        plot_wave(sig_rip_chst, sample_rate, cycle_div, ax=ax[1], wave_color = 'red', title = 'Chest belt',  markersize=fig_w/2, 
              start_t = start_t, show_div = True, show_peak = 'h', show_color = 'No', show_zero = False, 
              show_num=False)
    # Abdomen
    if review:     
        plot_wave(sig_rip_abd, sample_rate, cycle_div, ax=ax[2], wave_color = 'blue', title = 'Abdomen belt',  markersize=fig_w/2, 
                  start_t = start_t, show_div = True, show_peak = 'h', show_color = 'plot', show_zero = False, lbl_list = df['Airway status final'].tolist(),
                  show_num=True, num_size = fig_w/3, num_lbl_list = df['Stim mode final'].tolist())
    else:
        plot_wave(sig_rip_abd, sample_rate, cycle_div, ax=ax[2], wave_color = 'blue', title = 'Abdomen belt',  markersize=fig_w/2, 
                  start_t = start_t, show_div = True, show_peak = 'h', show_color = 'No', show_zero = False, 
                  show_num=True, num_size = fig_w/3, num_lbl_list = df['Stim mode'].tolist())
    # Sum
    # if not review:        
    #     c_chst = 1#df_lv.loc[df_lv['Chest coef std'] < 1, 'Chest coef'].mean() #0.055944409145625165
    #     c_abd = 1#df_lv.loc[df_lv['Abd coef std'] < 1, 'Abd coef'].mean() #0.15022963957224789
    #     intercept_c = 0
    #     sig_sum = sig_rip_abd * c_abd + sig_rip_chst * c_chst  + intercept_c - sig_epi
    #     plot_wave(sig_sum, sample_rate, cycle_div, ax=ax[3], wave_color = 'm', title = 'Chest+Abd',  markersize=fig_w/2, 
    #               start_t = start_t, show_div = True, show_peak = 'h', show_color = 'No', show_zero = False)
    # CPAP
    plot_wave(sig_cpap, sample_rate, cycle_div, ax=ax[3], wave_color = 'c', title = 'CPAP', start_t = start_t, show_div = False, show_peak = 'No')        
    # Pv   
    if review:   
        lbl_list = [i + j for i, j in zip(df['Airway status final'].fillna('').tolist(), df['Flow limit site final'].fillna('').tolist())]                      
        plot_wave(sig_pv, sample_rate, cycle_div, ax=ax[4], wave_color = 'green', title = 'Pv', markersize=fig_w/2,  
                  start_t = start_t, show_div = True, show_peak = 'l', show_zero = True, zero_lvl = 0, show_color = 'fls', lbl_list = lbl_list)
    else:
        plot_wave(sig_pv, sample_rate, cycle_div, ax=ax[4], wave_color = 'green', title = 'Pv', markersize=fig_w/2,  
                 start_t = start_t, show_div = True, show_peak = 'l', show_zero =True, zero_lvl = 0, show_color = 'No')
    
    # Pepi
    if review:  
        lbl_list = [i + j for i, j in zip(df['Airway status final'].fillna('').tolist(), df['Flow limit site final'].fillna('').tolist())]                       
        plot_wave(sig_epi, sample_rate, cycle_div, ax=ax[5], wave_color = 'red', title = 'Pepi', markersize=fig_w/2,  
                  start_t = start_t, show_div = True, show_peak = 'l', show_zero = True, zero_lvl = 0, show_color = 'fls', lbl_list = lbl_list)
    else:
        plot_wave(sig_epi, sample_rate, cycle_div, ax=ax[5], wave_color = 'red', title = 'Pepi', markersize=fig_w/2,  
                 start_t = start_t, show_div = True, show_peak = 'l', show_zero =True, zero_lvl = 0, show_color = 'No')
    
    if review:
        patch_list = []
        for patch_lbl in df_breaths['Flow limit site review'].unique():
            if not pd.isna(patch_lbl):
                patch_list.append(mpatches.Patch(color = patch_cdict[patch_lbl][0], label = patch_lbl, alpha = patch_cdict[patch_lbl][1]))
        ax[4].legend(handles=patch_list, 
                     # loc='center left', 
                     # box_to_anchor=(1.01, 0.2)
                     )

    # Flow
    if review:       
        lbl_list = [i + j for i, j in zip(df['Airway status final'].fillna('').tolist(), df['Plotted'].fillna('').tolist())] 
        plot_wave(sig_flow, sample_rate, cycle_div, ax=ax[-1], wave_color = 'purple', title = 'Adj flow', markersize=fig_w/2,  
                  start_t = start_t, show_div = True, show_peak = 'h', show_zero = True, show_color = 'plot + airway', lbl_list = lbl_list,
                  show_num=True, num_size = fig_w/3, num_lbl_list = df['Stim mode final'].tolist())
    else:
        plot_wave(sig_flow, sample_rate, cycle_div, ax=ax[-1], wave_color = 'purple', title = 'Adj flow', markersize=fig_w/2,  
                  start_t = start_t, show_div = True, show_peak = 'h', show_zero = True, show_color = 'airway', lbl_list = df['Airway status'].tolist(),
                  show_num=True, num_size = fig_w/3, num_lbl_list = df['Stim mode final'].tolist())    
    
    # Ins
    for row in range(len(df)):

        bar_start = df['Ins start'][row]/sample_rate + start_t
        bar_end = df['Ins end'][row]/sample_rate + start_t      
        ax[-1].plot([bar_start, bar_end], [0.96, 0.96], color="teal", transform = ax[-1].get_xaxis_transform(), clip_on = False) 
   
    plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")  
    fig.suptitle('Recordings ({} {}, R{}, {} - {} sec)'.format(name, study_time, record_id, t_start, t_end))
    fig.tight_layout()   
    if save_fig:
        fig.savefig(os.path.join(save_dir, name + ' ' + study_time + ' ' + str(record_id) + ' ' + str(t_start) + ' ' + str(t_end) + '.png' ), dpi=100)        
    plt.show()




''' measurement'''



''' detect breath condition'''

def normalized_plot(breath_sig):
    my_dpi = 96
    f = plt.figure(figsize=(299/my_dpi, 299/my_dpi), dpi=my_dpi)
    ax = f.add_subplot(111)
    ax.plot(breath_sig, c = 'k')   
    ax.axis('off')    
    f.tight_layout(pad=0)
    # To remove the huge white borders
    # ax.margins(0)      
    f.canvas.draw( )
    # Get the RGBA buffer from the figure
    w,h = f.canvas.get_width_height()
    buf = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
 
    # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis = 2 )
    plt.close()
    return buf/255.


def build_model_inception(input_shape, num_classes):
    tf.keras.backend.clear_session()

    input_tensor = Input(shape = input_shape)
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
   
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
   
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer = optimizer, #'rmsprop', 
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])
    
    return model


def detect_airway_status_dl(sig_flow, df):
    
    cycle_div = df['Ins start'].tolist()
    arr_list = []
    for i in range(len(cycle_div)):    
        try:
            arr_list.append(normalized_plot(sig_flow[cycle_div[i]: cycle_div[i+1]]))
        except:
            arr_list.append(normalized_plot(sig_flow[cycle_div[i]: ]))

    img_array = np.stack(arr_list, axis=0, out=None)
    predictions = model.predict(img_array)
    df['Airway_DL'] = np.argmax(predictions, axis = 1)
    df['Airway status DL'] = df['Airway_DL'].apply(lambda x: 'NFL' if x == 2 else 'FL' if x == 1 else 'Apnea')  
    return df


def detect_airway_status(row):
    if row['Time diff'] < 0 and row['Airway_DL'] == 1:
        val = 'FL'
    
    elif row['Flow std'] < 1 or row['Airway_DL'] == 0:
        val = 'Apnea'
        
    elif row['Flow std'] > 1 and row['Airway_DL'] == 2:
        val = 'NFL'
    else:
        val = 'Other'
    return val




def plot_pressure_flow_curve(df_temp, lbl_size = 20, reviewed = False, save_fig = False, show_curve= True,
                             show_outside = False, vi = 'max'):
    
    stim_mode_list = list(df_temp['Stim mode final'].unique())   
    fig = plt.figure(figsize=(10,6))
    plt.style.use('default')    
    if reviewed:
        airway_lbl = 'Airway status final' 
        stim_lbl = 'Stim mode final'
    else:
        airway_lbl = 'Airway status DL'
        stim_lbl = 'Stim mode'   
    if vi == 'max':
        flow_lbl = 'Flow max unadj'
    elif vi == 'mean':
        flow_lbl = 'Vimean'
    else:
        flow_lbl = 'Vi13'
        
    legendtext = []    
    min_nfl_list = []    
    for stim_mode in stim_mode_list:        
        min_nfl_p = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'NFL')]['CPAP ceil'].min()        
        y_nfl = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'NFL') & (df_temp['CPAP ceil'] == min_nfl_p)][flow_lbl].tolist()
        if len(y_nfl) > 0:
            min_nfl_list.append(np.mean(y_nfl))    
    y_open = np.mean(min_nfl_list) * 1000/60
    for stim_mode in stim_mode_list:      
        if stim_mode == 'ACS':
            marker_color = 'green'              
            marker_type = 'o'                
        elif stim_mode == 'Combined' :
            marker_color = 'darkorange'              
            marker_type = '^'            
        elif stim_mode == 'GPN' :
            marker_color = 'peru'              
            marker_type = 'D' 
        else:
            marker_color = 'navy'              
            marker_type = 's' 
            
        x_fl = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'FL')]['CPAP ceil'].tolist()        
        y_fl = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'FL')][flow_lbl].tolist()        
        max_apnea_p = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'Apnea')]['CPAP ceil'].max()                    
        min_nfl_p = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'NFL')]['CPAP ceil'].min()
        
        x_apnea = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'Apnea') & (df_temp['CPAP ceil'] == max_apnea_p)]['CPAP ceil'].tolist()        
        y_apnea = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'Apnea') & (df_temp['CPAP ceil'] == max_apnea_p)][flow_lbl].tolist()        
        x_nfl = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'NFL') & (df_temp['CPAP ceil'] == min_nfl_p)]['CPAP ceil'].tolist()        
        y_nfl = df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'NFL') & (df_temp['CPAP ceil'] == min_nfl_p)][flow_lbl].tolist()
                
        df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'Apnea') & (df_temp['CPAP ceil'] == max_apnea_p), 'Plotted']= 'Y'        
        df_temp.loc[(df_temp[stim_lbl] == stim_mode) & (df_temp[airway_lbl] == 'NFL') & (df_temp['CPAP ceil'] == min_nfl_p), 'Plotted']= 'Y'
        df_temp.loc[(df_temp[stim_lbl] == stim_mode)  & (df_temp[airway_lbl] == 'FL'), 'Plotted'] = 'Y'
                
        x_fl_arr = np.asarray(x_fl, dtype=np.float32) 
        y_fl_arr = np.asarray(y_fl, dtype=np.float32) * 1000/60
        x_apnea_arr = np.asarray(x_apnea, dtype=np.float32) 
        y_apnea_arr = np.asarray(y_apnea, dtype=np.float32) * 1000/60
        x_nfl_arr = np.asarray(x_nfl, dtype=np.float32) 
        y_nfl_arr = np.asarray(y_nfl, dtype=np.float32) * 1000/60
        
        x_arr =  np.concatenate((x_fl_arr, x_apnea_arr, x_nfl_arr), axis=None)
        y_arr =  np.concatenate((y_fl_arr, y_apnea_arr, y_nfl_arr), axis=None)
        z = np.polyfit(x_arr, y_arr, 1)
        p = np.poly1d(z)
                
        plt.scatter(x_fl_arr, y_fl_arr, marker = marker_type, s = lbl_size, alpha = 0.3, color = marker_color)
        plt.scatter(x_apnea_arr, y_apnea_arr, marker = marker_type, alpha = 0.9, color = marker_color, s = lbl_size)
        plt.scatter(x_nfl_arr, y_nfl_arr, marker = marker_type, alpha = 0.7, facecolors = 'none', s = lbl_size, 
                    edgecolors = marker_color )
        
        if show_curve:
            x_zero = -z[1]/z[0]
            try:
                x_open = max((y_open - z[1])/z[0], x_nfl_arr.min())   
            except:
                x_open = (y_open - z[1])/z[0]
            x_plt = np.asarray([x_zero, x_open])            
            popen = (y_open - z[1])/z[0]            
            lt = '{}: y = {:.2f} x {:+.2f}\nPcrit: {:.2f}; Popen: {:.2f}'.format(stim_mode, z[0], z[1], x_zero, popen)
            legendtext.append(lt)            
            plt.plot(x_plt, p(x_plt), linestyle = '--', color = marker_color, lw = 1, label = lt )
        else:
            legendtext.append(Line2D([], [], color="white", marker=marker_type, mfc ="none", mec = marker_color, label = stim_mode))
                
        if show_outside:
       
           x_nfl_discard = df_temp.loc[(df_temp[stim_lbl] == stim_mode) 
                       & (df_temp[airway_lbl] == 'NFL') 
                       & (df_temp['CPAP ceil'] > min_nfl_p)]['CPAP ceil'].to_numpy()
           
           y_nfl_discard = df_temp.loc[(df_temp[stim_lbl] == stim_mode) 
                       & (df_temp[airway_lbl] == 'NFL') 
                       & (df_temp['CPAP ceil'] > min_nfl_p)][flow_lbl].to_numpy() * 1000/60
           
           x_apnea_discard = df_temp.loc[(df_temp[stim_lbl] == stim_mode) 
                       & ((df_temp[airway_lbl] == 'Apnea') & (df_temp['CPAP ceil'] < max_apnea_p)
                       | (df_temp[airway_lbl] == 'Dogleg'))
                       ]['CPAP ceil'].to_numpy()
           
           y_apnea_discard = df_temp.loc[(df_temp[stim_lbl] == stim_mode) 
                       & ((df_temp[airway_lbl] == 'Apnea') & (df_temp['CPAP ceil'] < max_apnea_p)
                       | (df_temp[airway_lbl] == 'Dogleg'))   
                       ]['Flow max unadj'].to_numpy() * 1000/60
           
           plt.scatter(x_nfl_discard, y_nfl_discard, marker = marker_type, alpha = 0.7, facecolors = 'none', s = lbl_size, 
                    edgecolors = marker_color)
           
           plt.scatter(x_apnea_discard, y_apnea_discard, marker = marker_type, alpha = 0.9, color = marker_color, s = lbl_size)
           
    plt.axhline(y=0, color='lightcoral', linestyle='-.', alpha = 0.7)
    plt.axhline(y=y_open, color='orchid', linestyle='--', alpha = 0.7)
    if show_curve:
        plt.legend(fontsize = 10, loc = 'best')
    else:
        plt.legend(handles = legendtext, fontsize = 10, loc = 'best')
    plt.title('Pressure flow curves', size = 16)
    plt.xlabel('CPAP pressure (cm H2O)', size = 12)
    if vi == 'max':
        plt.ylabel('Peak inspiratory flow (mL/sec)', size = 12)
    elif vi == 'mean':
        plt.ylabel('Mean inspiratory flow (mL/sec)', size = 12)
    else:
        plt.ylabel('Inspiratory flow at middle 1/3 (mL/sec)', size = 12)
    fig.tight_layout()        
    plt.show()    
    if save_fig:
        f_name  = name + ' ' + study_time + ' R{} {}-{} curve {}.png'.format(record_id, t_start, t_end, vi)

        fig.savefig(os.path.join(save_dir, f_name ), dpi=100)
    
    return df_temp

def save_df(df, save_path, sort = False):
       
    if sort:
        df.sort_values(['Stim mode final', 'Airway status final', 'CPAP ceil'], inplace=True)
    
    if os.path.isfile(save_path):
    
        with pd.ExcelWriter(save_path, mode='a', if_sheet_exists = 'replace') as writer: 
    
            df.to_excel(writer, index=False, sheet_name='R{} {}-{}'.format(record_id, t_start, t_end))       
    else:
        
        df.to_excel(save_path, index=False, sheet_name='R{} {}-{}'.format(record_id, t_start, t_end))

def final_lbl_stim(row):
    if pd.isna(row['Stim mode review']):
        val = row['Stim mode']

    else:
        val = row['Stim mode review']
    return val

def final_lbl_airway(row):
    if pd.isna(row['Airway status review']):
        val = row['Airway status DL']
    else:
        val = row['Airway status review']
    return val

def final_lbl_fls(row):
    if pd.isna(row['Flow limit site review']):
        val = row['Flow limit site']
    else:
        val = row['Flow limit site review']
    return val

def read_reviewed_df(save_path):
    
    df = pd.read_excel(save_path, index_col=None, sheet_name = 'R{} {}-{}'.format(record_id, t_start, t_end))  
    df['Stim mode final'] = df.apply(final_lbl_stim, axis=1)
    df['Airway status final'] = df.apply(final_lbl_airway, axis=1)
    df['Flow limit site'] = df.apply(flow_limit_site_update, axis = 1) 
    df['Flow limit site final'] = df.apply(final_lbl_fls, axis = 1)
    
    df['Plotted'] = ''
    
    # df.loc[df['Airway status final'] == 'Remove',  'Plotted'] = ''
    
    df.sort_values('Pepi min time', inplace = True, ignore_index = True)
    
    return df



''' overlay Pepi Pv and Pn whole run'''

def get_mean_sompensate(df_breaths, list_to_exclude = None):
    if list_to_exclude is not None:
        df_temp = df_breaths.loc[(df_breaths['Airway status final'] == 'NFL') & (~df_breaths.index.isin([i-1 for i in list_to_exclude])),
                                 # & (df_breaths['Stim mode final'] == 'No stim') 
                                 ].reset_index(drop=True)
    else:
        df_temp = df_breaths.loc[(df_breaths['Airway status final'] == 'NFL') 
                                 # & (df_breaths['Stim mode final'] == 'No stim') 
                                 ].reset_index(drop=True)
    lgtxt = []
    plt.style.use('default')
    for outcome, color, marker in zip(['Exp pv mean', 'Exp pepi mean', 'CPAP mean'], ['g', 'b', 'r'], ['o', '*', 'v']):
        x = df_temp['CPAP mean']
        y = df_temp[outcome]
        z1 = np.polyfit(x, y, 1)
        p1 = np.poly1d(z1)   
        plt.scatter(x, y, color = color, marker = marker)
        plt.plot([min(x), max(x)], p1([min(x), max(x)]), linestyle = '--', color = color, lw = 1, label = '_nolegend_')
        lgtxt.append('={:.2f}x{:+.2f}'.format(z1[0], z1[1]))
        if outcome == 'Exp pv mean':
            df_breaths['pv_c1'] = z1[0]
            df_breaths['pv_c0'] = z1[1]
            # if zero == 'max':
            #     df_breaths['compensate_pv'] = (1- z1[0]) * max(x) - z1[1] 
            # elif zero == 'min':
            #     df_breaths['compensate_pv'] = (1- z1[0]) * min(x) - z1[1] 
            # else:
            #     df_breaths['compensate_pv'] = (1- z1[0]) * pn - z1[1] 
        elif outcome == 'Exp pepi mean':
            df_breaths['pepi_c1'] = z1[0]
            df_breaths['pepi_c0'] = z1[1]
            # if zero == 'max':
            #     df_breaths['compensate_pepi'] =  (1-z1[0]) * max(x) - z1[1]
            # elif zero == 'min':
            #     df_breaths['compensate_pepi'] =  (1-z1[0]) * min(x) - z1[1]
            # else:
            #     df_breaths['compensate_pepi'] = (1- z1[0]) * pn - z1[1] 
    # plt.plot(df_temp['CPAP mean'], df_temp['Exp pepi mean'], 'b*')
    # plt.plot(df_temp['CPAP mean'], df_temp['CPAP mean'], 'rv')   
    df_breaths['max_nfl_pn'] = max(x)
    df_breaths['min_nfl_pn'] = min(x)
    plt.legend(["{}{}".format(a_,b_) for a_, b_ in zip(['Pv', 'Pepi', 'Pn'], lgtxt)])
    plt.show()
    
    # return df_breaths
    

def plot_overlay_p(start_b, end_b, zero = 'max'):

    sig_pv_temp = sig_pv[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_pepi_temp = sig_epi[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_flow_temp = sig_flow[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_stim_temp = sig_stim_lfilt[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_pn_temp = sig_cpap[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_belts_temp = (sig_rip_chst + sig_rip_abd)[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    
    vimax_time = (df_breaths['Flow max time'] - t_start) * sample_rate 
    vimax_time = vimax_time[start_b-1 : end_b]
    if zero == 'max':
        sig_pv_offset = (1- df_breaths['pv_c1'][0]) * df_breaths['max_nfl_pn'][0] - df_breaths['pv_c0'][0] 
        sig_pepi_offset = (1- df_breaths['pepi_c1'][0]) * df_breaths['max_nfl_pn'][0]  - df_breaths['pepi_c0'][0]  
    elif zero == 'min':
        sig_pv_offset = (1- df_breaths['pv_c1'][0]) * df_breaths['min_nfl_pn'][0] - df_breaths['pv_c0'][0] 
        sig_pepi_offset = (1- df_breaths['pepi_c1'][0]) * df_breaths['min_nfl_pn'][0]  - df_breaths['pepi_c0'][0] 
    else:
        sig_pv_offset = (1- df_breaths['pv_c1'][0]) * np.mean(sig_pn_temp) - df_breaths['pv_c0'][0] 
        sig_pepi_offset = (1- df_breaths['pepi_c1'][0]) * np.mean(sig_pn_temp)  - df_breaths['pepi_c0'][0] 
        
    plt.style.use("default")
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))
    ax[0].plot(sig_stim_temp*1000, 'b')
    ax[1].plot(sig_flow_temp, 'purple')
    ax[2].plot(sig_pn_temp, 'cyan', alpha = 0.5, label = 'N')
    ax[2].plot(sig_pv_temp+sig_pv_offset, 'g', label = 'V')
    ax[2].plot(sig_pepi_temp+sig_pepi_offset, 'r', alpha = 0.5, label = 'Epi')
    
    for xc in vimax_time:
        ax[2].axvline(x=xc-df_breaths['Ins start'][start_b-1], alpha = 0.5, c='purple', ls = '--')
    ax[3].plot(sig_belts_temp, 'r')
    
    ax[2].legend()
    ax[0].set_ylabel('Stim')
    ax[1].set_ylabel('Flow')
    ax[2].set_ylabel('Pressure')
    ax[3].set_ylabel('Sum Rip belts')
    fig.tight_layout()
    plt.show()
    

def plot_overlay_p_1b(b, zero = 'max', preset = 0, postset = 0, wlen = 50, annotate=True, save_fig = False, pv_add = 0, pe_add = 0):
    
    # b = 2
    t_range = range(df_breaths['Ins start'][b-1] - preset, df_breaths['Exp end'][b-1] - postset)
    
    sig_pv_temp = sig_pv[t_range]
    sig_pepi_temp = sig_epi[t_range]
    sig_flow_temp = sig_flow[t_range]
    sig_stim_temp = sig_stim_lfilt[t_range]
    sig_pn_temp = sig_cpap[t_range]
    # sig_belts_temp = (sig_rip_chst + sig_rip_abd)[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
            
    vimax_time = (df_breaths['Flow max time'] - t_start) * sample_rate 
    vimax_time = vimax_time[b-1]
    epi_min_time = (df_breaths['Pepi min time'] - t_start) * sample_rate  
    epi_min_time = epi_min_time[b-1]
    
    if wlen is not None:
        sig_flow_temp = smooth(sig_flow_temp, window_len=wlen, window='hanning')
        sig_pv_temp = smooth(sig_pv_temp, window_len=wlen, window='hanning')
        sig_pn_temp = smooth(sig_pn_temp, window_len=wlen, window='hanning')
        sig_pepi_temp = smooth(sig_pepi_temp, window_len=wlen, window='hanning')
        vimax_time = np.argmax(sig_flow_temp) + df_breaths['Ins start'][b-1] - preset
        # pvmin_time = np.argmin(sig_pv_temp) + df_breaths['Ins start'][b-1] - offset
        epi_min_time = np.argmin(sig_pepi_temp) + df_breaths['Ins start'][b-1] - preset
        
    if zero == 'max':
        sig_pv_offset = (1- df_breaths['pv_c1'][0]) * df_breaths['max_nfl_pn'][0] - df_breaths['pv_c0'][0]  + pv_add
        sig_pepi_offset = (1- df_breaths['pepi_c1'][0]) * df_breaths['max_nfl_pn'][0]  - df_breaths['pepi_c0'][0]  + pe_add
    elif zero == 'min':
        sig_pv_offset = (1- df_breaths['pv_c1'][0]) * df_breaths['min_nfl_pn'][0] - df_breaths['pv_c0'][0] + pv_add
        sig_pepi_offset = (1- df_breaths['pepi_c1'][0]) * df_breaths['min_nfl_pn'][0]  - df_breaths['pepi_c0'][0] + pe_add
    elif zero == 'local':
        sig_pv_offset = (1- df_breaths['pv_c1'][0]) * np.mean(sig_pn_temp) - df_breaths['pv_c0'][0] + pv_add
        sig_pepi_offset = (1- df_breaths['pepi_c1'][0]) * np.mean(sig_pn_temp)  - df_breaths['pepi_c0'][0] + pe_add
    else:
        start_temp = len(sig_flow_temp) - (df_breaths['Exp end'][b-1] - df_breaths['Exp start'][b-1])
        end_temp = len(sig_flow_temp)                 
        sig_pv_offset = np.mean(sig_pn_temp)- stats.mode(sig_pv_temp[int(start_temp): int(end_temp)])[0] + pv_add
        sig_pepi_offset = np.mean(sig_pn_temp)- stats.mode(sig_pepi_temp[int(start_temp): int(end_temp)])[0] + pe_add

        vimax_time = np.argmax(sig_flow_temp) + df_breaths['Ins start'][b-1] - preset
        epi_min_time = np.argmin(sig_pepi_temp) + df_breaths['Ins start'][b-1] - preset
        
    plt.style.use("default")
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3,4))
    # ax[0].plot(sig_stim_temp*1000, 'b')
    ax[0].plot(sig_flow_temp, 'purple')
    if np.mean(sig_stim_temp)*1000 >= 1:
        ax[0].set_facecolor('cyan')
    
    ax[0].set_ylim([sig_flow.min()*1.05, sig_flow.max()*1.05])
    
    ax[1].plot(sig_pn_temp, 'cyan', alpha = 0.5, label = 'N')
    ax[1].plot(sig_pv_temp+sig_pv_offset, 'g', label = 'V')
    ax[1].plot(sig_pepi_temp+sig_pepi_offset, 'r', alpha = 0.5, label = 'Epi')
    if annotate:
        ax[0].axvline(x = epi_min_time - df_breaths['Ins start'][b-1] + preset, alpha = 0.5, c='r', ls = '--')
        ax[1].axvline(x = vimax_time - df_breaths['Ins start'][b-1] + preset, alpha = 0.5, c='purple', ls = '--')
        ax[1].text(0.8, 0.1, '{} #{}'.format(name,b), transform=ax[1].transAxes, fontsize = 6)
    # for xc in vimax_time:
    
    # ax[3].plot(sig_belts_temp, 'r')
    
    ax[1].legend(fontsize = 6)
    # ax[0].set_ylabel('Stim')
    ax[0].set_ylabel('Flow')
    ax[1].set_ylabel('Pressure')
    # ax[3].set_ylabel('Sum Rip belts')
    fig.tight_layout()  
    if save_fig:
        save_f = os.path.join(save_dir, name + ' ' + study_time + ' ' + str(record_id) + ' ' + str(t_start) + ' ' + str(t_end))
        Path(save_f).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_f, str(b) + '.png' ), dpi=100)        
    plt.show()
    
    # return vimax_time, pvmin_time, pepimin_time

def smooth(x, window_len=11, window='hanning'):
    #['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1): -int(window_len/2)] 



def gen_v_gif(breath_num, preset = 0, postset = 0, p_speed=3):
    
    # breath_num = 7

    v_start = (df_breaths['Ins start'][breath_num-1] - preset )/sample_rate + t_start
    v_end = (df_breaths['Exp end'][breath_num-1] - postset) /sample_rate + t_start
    
    clip = VideoFileClip(video_path).subclip(t_start = v_start, t_end = v_end).resize(0.75)
    
    a = clip.size  
    b = clip.crop(x_center=a[0]/2 , y_center=a[1]/2, width=min(a), height=min(a))
    b.speedx(p_speed).write_gif(os.path.join(video_save_dir, '{}.gif'.format(breath_num)))


def gen_v_gif_multi(b_start, b_end, preset = 0, postset = 0, p_speed=3):

    v_start = (df_breaths['Ins start'][b_start-1] - preset )/sample_rate + t_start
    v_end = (df_breaths['Exp end'][b_end-1] - postset) /sample_rate + t_start
    
    clip = VideoFileClip(video_path).subclip(t_start = v_start, t_end = v_end).resize(0.75)
    
    a = clip.size  
    b = clip.crop(x_center=a[0]/2 , y_center=a[1]/2, width=min(a), height=min(a))
    b.speedx(p_speed).write_gif(os.path.join(video_save_dir, '{}-{}.gif'.format(b_start, b_end)))


'''
1.	No obstruction: NFL, Pv and Pepi align with each other, T vimax = T p nadirs
2.	Velopharynx: FL, Pv and Pepi align with each other, T vimax < T p nadirs
3.	Velopharynx(secondary oropharynx): FL, Pv and Pepi dessociate, T pv nadir < T pepi nadir
4.	Oropharynx (O or T): FL, Pv and Pepi dessociate, T pv nadir = T pepi nadir
5.	Epiglottis: FL, Pv and Pepi align with each other, p nadirs ~ Pn
'''

    



''' load model'''
model = build_model_inception(input_shape=(299,299,3), num_classes=3)
checkpoint_path =  r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\Other projects\ANSA stimulation\dl_model_flow.h5'   
model.load_weights(checkpoint_path)


''' Excecution'''


 
''' step 1. define channels and sections'''
  
name =  'DD'
month = '03'
year = '2024'  

record_id = 6									
t_start = 4353									 	
t_end = 4652


minimun_breath_dur = 2

study_time = year + '-' + month
adicht_name = name + ' ' + year[-2:] + '-' + month +  '.adicht'
file_path = os.path.join(r'D:\ANSA stim data', study_time, adicht_name)   
 

save_dir = os.path.join(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\Other projects\ANSA stimulation\Analysis\CN IX', study_time, name)
save_file = name + '-' + year[-2:] + '-' +  month + '.xlsx'
save_path = os.path.join(save_dir, save_file)
Path(save_dir).mkdir(parents=True, exist_ok=True)

video_save_dir = os.path.join(save_dir, name + ' ' + study_time + ' ' + str(record_id) + ' ' + str(t_start) + ' ' + str(t_end) + ' v')
Path(video_save_dir).mkdir(parents=True, exist_ok=True)  
video_path = os.path.join(r'D:\ANSA stim data', study_time, name + ' ' + year[-2:] + '-' + month + ' movies', 'Record{}.wmv'.format(record_id))

# All id numbering is 1 based, first channel, first block
# When indexing in Python we need to shift by 1 for 0 based indexing
# Functions however respect the 1 based notation ...

# These may vary for your file ...

dict_ch = {'ch_stim' : 1,
           'ch_rip_chst': 2,
           'ch_rip_abd': 3, 
           'ch_pv': 4, 
           'ch_epi' : 5,
           'ch_flow' : 6,
           'ch_cpap' : 8}

patch_cdict = {'V': ['tab:purple', 0.8],
               'Vo': ['tab:purple', 0.6],
               'Vt': ['tab:purple', 0.4],
               'Ve': ['tab:purple', 0.2],                  
               'O': ['tab:cyan', 0.8],
               'T': ['tab:olive', 0.8],  
               'Te': ['tab:olive', 0.4],  
               'E': ['tab:red', 0.4],
               }



''' step 2, read and analyze file'''

sig_dict, sr_dict = get_signal_data(dur=(t_start, t_end))

sig_pv, sig_epi, sig_cpap, sig_flow = sig_dict['ch_pv'], sig_dict['ch_epi'], sig_dict['ch_cpap'], sig_dict['ch_flow']
sig_stim, sig_rip_chst, sig_rip_abd = sig_dict['ch_stim'], sig_dict['ch_rip_chst'], sig_dict['ch_rip_abd']

sig_stim_lfilt = low_pass_filter(sig_stim, sr_dict['ch_stim'], cutoff=1)

sig_flow = highpass_filter(sig_flow, sr_dict['ch_flow'], 0.1, 2)
''' filter out flow noise'''
# sig_flow = low_pass_filter(sig_flow, sr_dict['ch_flow'], cutoff=1.5)


sample_rate = 100.

sig_pv =  re_sample(sig_pv, sr_dict['ch_pv'], sample_rate)
sig_epi =  re_sample(sig_epi, sr_dict['ch_epi'], sample_rate)
sig_rip_chst = re_sample(sig_rip_chst, sr_dict['ch_rip_chst'], sample_rate)
sig_rip_abd = re_sample(sig_rip_abd, sr_dict['ch_rip_abd'], sample_rate)
sig_flow =  re_sample(sig_flow, sr_dict['ch_flow'], sample_rate) 
sig_cpap =  re_sample(sig_cpap, sr_dict['ch_cpap'], sample_rate)
sig_stim_lfilt = re_sample(sig_stim_lfilt, sr_dict['ch_stim'], sample_rate, req_len = len(sig_cpap))

# Should not use highpass filter on these two channels as they are compared to Pn
# sig_pv_lfilt =  low_pass_filter(sig_pv, sample_rate, cutoff=20) 
# sig_epi_lfilt =  low_pass_filter(sig_epi, sample_rate,cutoff=20)




df_cycle = get_cycles(sig_rip_chst + sig_rip_abd - sig_epi , poly_order = 5, min_cycle = 7, rise_rate= 0.05)
    
df_cycle2 = recheck_cycles(df_cycle, minimun_rate = 0.5) #0.

# ''' combine two cycles'''
# df_cycle2 = manual_combine(df_cycle2, c_list = [6,8,10,14,17,19,21,23,25,27,29,31,33,35,37,43,53,58,62,66,69,71,73,75,77,79,81,83,85,87,89,92,94]) 

# # ''' adjust time'''
# b = [61]
# df_cycle2.loc[[x - 1 for x in b], 'Ins start'] += 120
# df_cycle2['Dur'] = (df_cycle2['Ins end'] - df_cycle2['Ins start'] )/100

# b = [54]
# df_cycle2.loc[[x - 1 for x in b], 'Ins start'] += 150
# df_cycle2['Dur'] = (df_cycle2['Ins end'] - df_cycle2['Ins start'] )/100

# b = [41]
# df_cycle2.loc[[x - 1 for x in b], 'Ins end'] -= 100
# df_cycle2['Dur'] = (df_cycle2['Ins end'] - df_cycle2['Ins start'] )/100

# '''All'''
# df_cycle2.loc[:, 'Ins start'] -= 30
# df_cycle2['Dur'] = (df_cycle2['Ins end'] - df_cycle2['Ins start'] )/100


df_breaths = get_measurement(df_cycle2)

df_breaths = data_analysis(df_breaths, initial = True)

plot_multi_waves(df_breaths, start_t = t_start, review=False, save_fig = False)



''' deep learning'''

df_breaths = detect_airway_status_dl(sig_flow, df_breaths)

df_breaths = data_analysis(df_breaths, initial = False)
 
# plot_multi_waves(df_breaths, start_t = t_start, review=False, save_fig = False)

save_df(df_breaths, save_path, sort = False)



''' step 3. perform review'''
df_breaths = read_reviewed_df(save_path)

# df_breaths.loc[:, 'Airway status final'] = 'NFL'

''' step 4. plot final curve'''

df_breaths = plot_pressure_flow_curve(df_breaths, lbl_size = 40, reviewed = True, show_outside = True, show_curve = True, save_fig = True)

plot_multi_waves(df_breaths, start_t = t_start, review = True, save_fig = True)



'''remeasurement'''


def detect_ins_flow(sig_flow, smoothing = 0.1, min_dur = 0.8):   
    ma = int(smoothing*sample_rate)
    sig_f = pd.Series(sig_flow).rolling(ma, min_periods=1).mean().shift(-int(ma/2)).fillna(0).to_numpy()
    a = np.where(sig_f>0, 1, 0)
    d = get_seq_len(a, thresh = int(sample_rate*min_dur))    
    return d

def detect_ins_belt_pressure(sig_b, sig_p, smoothing = 0.1, min_dur = 0.8):   
    ma = int(smoothing*sample_rate)
    sig_b1 = pd.Series(sig_b).rolling(ma, min_periods=1).mean().shift(-int(ma/2)).fillna(0).to_numpy()
    sig_p1 = pd.Series(sig_p).rolling(ma, min_periods=1).mean().shift(-int(ma/2)).fillna(0).to_numpy()
    max_effort_t = max(np.argmin(sig_p1), np.argmax(sig_b1))
    min_effort_t = np.argmin(sig_b1[:max_effort_t])
    
    if max_effort_t - min_effort_t < int(sample_rate*min_dur):   
        return [0, max_effort_t]
    else:
        return [min_effort_t, max_effort_t]
    
def detect_ins_pressure(sig_p, smoothing = 0.1):   
    ma = int(smoothing*sample_rate)
    sig_p1 = pd.Series(sig_p).rolling(ma, min_periods=1).mean().shift(-int(ma/2)).fillna(0).to_numpy()
    max_effort_t = np.argmin(sig_p1)
    return [0, max_effort_t]

    

def remeasure_insp(df_breaths, sig_belt, sig_pressure, min_dur):

    for row in range(len(df_breaths)):
        # row = 1
        cyc_start = df_breaths['Ins start'][row]
        cyc_end = df_breaths['Exp end'][row]
        sig_temp = sig_flow[int(cyc_start): int(cyc_end)+1]
        d = detect_ins_flow(sig_temp, smoothing = 0.1, min_dur = min_dur)   
        sig_belt_seg = sig_belt[int(cyc_start): int(cyc_end)+1]
        sig_pressure_seg = sig_pressure[int(cyc_start): int(cyc_end)+1]
        d1 = detect_ins_belt_pressure(sig_belt_seg, sig_pressure_seg, smoothing = 0.1, min_dur = min_dur)
        if len(d) >0:
            df_breaths.loc[row, 'Ins start remeasure'] = d[0][0] + cyc_start
            df_breaths.loc[row, 'Ins end remeasure'] = d[0][1] + cyc_start 
        else :
            df_breaths.loc[row, 'Ins start remeasure'] = d1[0] + cyc_start
            df_breaths.loc[row, 'Ins end remeasure'] = d1[1] + cyc_start 
            

def get_flow_measurement(df_breaths):
        
    for i in range(len(df_breaths)):        
        # i = 0                    
        start_t = df_breaths['Ins start remeasure'][i]     
        end_t = df_breaths['Ins end remeasure'][i]      
        sig_f = sig_flow[int(start_t) : int(end_t) + 1]         
        vimean = np.mean(sig_f)
        vi13 = sig_f[int(len(sig_f)/3)]
        
        if not (vimean > 0 and vi13 > 0):
            start_t_b = df_breaths['Ins start'][i]  
            end_t_b = df_breaths['Exp end'][i]  
            sig_f = sig_flow[int(start_t_b) : int(end_t_b) + 1] 
            a = sig_f[sig_f>0]
            vimean = np.mean(a)
            try:
                vi13 = a[int(len(a)/3)]
            except:
                vi13 = 0
            
        df_breaths.loc[i, 'Vimean'] = vimean
        df_breaths.loc[i, 'Vi13'] = vi13
             
    return df_breaths


df_breaths = read_reviewed_df(save_path)
remeasure_insp(df_breaths, sig_rip_abd, sig_epi, min_dur = 0.8)

df_breaths = pd.read_excel(save_path, index_col=None, sheet_name = 'R{} {}-{} insp remeasure'.format(record_id, t_start, t_end))  

df_breaths = plot_pressure_flow_curve(df_breaths, lbl_size = 20, reviewed = True, show_outside = True, show_curve = True, save_fig = True, vi='vi13')

plot_multi_waves(df_breaths, start_t = t_start, review = True, save_fig = False)

df_breaths = get_flow_measurement(df_breaths)

with pd.ExcelWriter(save_path, mode='a', if_sheet_exists = 'replace') as writer: 
    df_breaths.to_excel(writer, index=False, sheet_name='R{} {}-{} insp remeasure'.format(record_id, t_start, t_end))   
    




''' step 5. generate per pn plots and per breath plots, review FLS'''

df_breaths = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\Other projects\ANSA stimulation\Analysis\CN IX\2024-03\DD\DD-24-03.xlsx', 
                           sheet_name='R6 4353-4652 insp remeasure')

get_mean_sompensate(df_breaths)

start_b = 33
end_b = 47    

plot_overlay_p(start_b, end_b, zero = 'local') # if hypopnic or apneic, use min, otherwise local.
   

# vimax_list, pnmin_list, pepi_list = [], [], []
for i in range(1, len(df_breaths)):
    plot_overlay_p_1b(b = i, zero = 'breath', preset = 0, postset = 0, wlen = None, save_fig = True, pv_add = 0, pe_add = -0)
    # vimax_list.append(a1)
    # pnmin_list.append(a2)
    # pepi_list.append(a3)

plot_overlay_p_1b(b = 10, zero = 'breath', preset = 50, postset = 50, wlen = None, save_fig = False, annotate=False,
                  pv_add = 0, pe_add = 0)

# np.mean([a_i - b_i for a_i, b_i in zip(vimax_list, pnmin_list)])
# np.mean([a_i - b_i for a_i, b_i in zip(vimax_list, pepi_list)])

# np.std([a_i - b_i for a_i, b_i in zip(vimax_list, pnmin_list)])

# np.mean(df_breaths['Flow max time'] - df_breaths['Pv min time']) * sample_rate
# np.mean(df_breaths['Flow max time'] - df_breaths['Pepi min time']) * sample_rate

''' step 6. read and final plot'''

df_breaths = read_reviewed_df(save_path)

df_breaths = plot_pressure_flow_curve(df_breaths, lbl_size = 100, reviewed = True, show_outside = True, show_curve = True, save_fig = False)
plot_multi_waves(df_breaths, start_t = t_start, review = True, save_fig = False)

df_breaths = plot_pressure_flow_curve(df_breaths, lbl_size = 100, reviewed = True, show_outside = True, show_curve = True, save_fig = True)

df_breaths.loc[df_breaths['Airway status final'] == 'NFL', 'Flow limit site final'] = 'N'

last_bsl = 0
for i in range(len(df_breaths)):
    if df_breaths.iloc[i]['Airway status final'] == 'Arousal': # or df_breaths.iloc[i]['Airway status final'] == 'Remove' 
        continue
    elif df_breaths.iloc[i]['Stim mode final'] != 'ACS':
        df_breaths.at[i, 'Flow limit site bsl'] = df_breaths.iloc[i]['Flow limit site final']
        last_bsl = i
    else:
        df_breaths.at[i, 'Flow limit site bsl'] = df_breaths.iloc[last_bsl]['Flow limit site final']

plot_multi_waves(df_breaths, start_t = t_start, review = True, save_fig = True)

save_df(df_breaths, save_path, sort = False)



''' GIF'''

# df_breaths = pd.read_excel(save_path, index_col=None, sheet_name = 'R{} {}-{}'.format(record_id, t_start, t_end))  
gen_v_gif(breath_num = 34, preset = 50, postset = 50, p_speed=4)



# multi physiology and endoscopy
start_b = 40
end_b = 43    

gen_v_gif_multi(b_start = start_b, b_end = end_b, preset = 50, postset = 50, p_speed=3)

plot_overlay_p(start_b, end_b, zero = 'local')




''' for exploratory/backup'''

# df_breaths['Airway status final'] = 'NFL'


''' overlay Pepi Pv and Pn LOCAL'''
def get_mean_bsl_p(sig, start_b, end_b):
    mean_list = []
    for i in range(start_b-1, end_b):
        exp_start = df_breaths['Exp start'][i] 
        exp_end = df_breaths['Exp end'][i] 
        mean_list.append(np.mean(sig[exp_start:exp_end]))
    return np.mean(mean_list)
    
def plot_overlay_p(start_b, end_b):

    sig_pv_temp = sig_pv[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_pepi_temp = sig_epi[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_flow_temp = sig_flow[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]
    sig_stim_temp = sig_stim_lfilt[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]

    sig_pv_offset = get_mean_bsl_p(sig_pv, start_b, end_b)
    sig_pepi_offset = get_mean_bsl_p(sig_epi, start_b, end_b)
    plt.style.use("default")
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,6))
    ax[0].plot(sig_stim_temp*1000, 'b')
    ax[1].plot(sig_flow_temp, 'purple')
    ax[2].plot(sig_pv_temp-sig_pv_offset, 'g', label = 'V')
    ax[2].plot(sig_pepi_temp-sig_pepi_offset, 'r', alpha = 0.5, label = 'Epi')
    ax[2].legend()
    ax[0].set_ylabel('Stim')
    ax[1].set_ylabel('Flow')
    ax[2].set_ylabel('Pressure')
    fig.tight_layout()
    plt.show()


start_b = 33
end_b = 47
plot_overlay_p(start_b, end_b)






# RV : 6, 9, 22, 31, 36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54

b = 4
plot_overlay_p_1b(b = b, zero = 'local', offset = 50, wlen = None, save_fig = False)
plot_overlay_p_1b(b = b, zero = 'local', offset = 50, wlen = 40, save_fig = False)




''' '''
# start_b = 72
# end_b = 74  
# df_temp = df_breaths.iloc[start_b-1: end_b]
# sig_pepi_temp = sig_epi[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]] 
# sig_belts_temp = (sig_rip_chst + sig_rip_abd)[df_breaths['Ins start'][start_b-1] : df_breaths['Exp end'][end_b-1]]

# plt.style.use("default")
# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,6))
# ax[0].plot(sig_pepi_temp, 'b')
# ax[1].plot(sig_belts_temp, 'purple')


# x_epi_max_list, y_epi_max_list = [], []
# x_belt_min_list, y_belt_min_list = [], []
# x_vi_max_list = []
  
# for i in range(start_b-1, end_b):
#     # i = 61
#     epi_max_idx = np.argmax(sig_epi[df_breaths['Ins start'][i] : df_breaths['Exp end'][i]]) + df_breaths['Ins start'][i]
#     x_epi_max_list.append(epi_max_idx)
#     epi_max = sig_epi[epi_max_idx]
#     y_epi_max_list.append(epi_max)
    
#     belt_min_idx = np.argmin((sig_rip_chst + sig_rip_abd)[df_breaths['Ins start'][i] : df_breaths['Exp end'][i]]) + df_breaths['Ins start'][i]
#     belt_min = (sig_rip_chst + sig_rip_abd)[belt_min_idx]
#     x_belt_min_list.append(belt_min_idx)
#     y_belt_min_list.append(belt_min)
    
#     vimax_idx = np.argmax(sig_flow[df_breaths['Ins start'][i] : df_breaths['Exp end'][i]]) + df_breaths['Ins start'][i]
#     x_vi_max_list.append(vimax_idx)
    
#     ax[0].plot(epi_max_idx - df_breaths['Ins start'][start_b-1] , epi_max, 'gx', markersize=12)
#     ax[1].plot(belt_min_idx - df_breaths['Ins start'][start_b-1] , belt_min, 'yx', markersize=12)
    
# z1 = np.polyfit(x_epi_max_list, y_epi_max_list, 1)
# p1 = np.poly1d(z1)

# z2 = np.polyfit(x_belt_min_list, y_belt_min_list, 1)
# p2 = np.poly1d(z2)

# ax[0].plot(x_epi_max_list - df_breaths['Ins start'][start_b-1] , p1(x_epi_max_list), ls = '--', color = 'g', lw = 1)
# ax[1].plot(x_belt_min_list - df_breaths['Ins start'][start_b-1] , p2(x_belt_min_list ), ls = '--', color = 'y', lw = 1)
# ax[0].set_ylabel('Epi')
# ax[1].set_ylabel('Belts')
# fig.tight_layout()
# plt.show()



# fig, ax1 = plt.subplots(nrows=1, ncols=1)
# ax2 = ax1.twinx()
   
# ax1.scatter(x_epi_max_list, y_epi_max_list, marker = 'x', s = 10, color = 'g')   
# ax1.plot(x_epi_max_list, p1(x_epi_max_list), linestyle = '--', color = 'g', lw = 1, label = 'Epi' ) 
# ax1.set_ylim([-1, 1])

# ax2.scatter(x_belt_min_list, y_belt_min_list, marker = 'o', s = 10, color = 'y')       
# ax2.plot(x_belt_min_list, p2(x_belt_min_list), linestyle = '--', color = 'y', lw = 1, label = 'Belts' ) 


# fig.legend(loc="lower right", bbox_to_anchor=(0.2,0.1), bbox_transform=ax1.transAxes)

# plt.show()    
    



# ''' plot Pepi Pv and Pn'''

# df_breaths.columns


# def plot_exp_p(df_breaths):

#     df_temp = df_breaths.loc[(df_breaths['Airway status final'] == 'NFL')
#                                & (df_breaths['Stim mode final'] == 'No stim') 
#                              ].reset_index(drop=True)
#     lgtxt = []
#     plt.style.use('default')
#     for outcome, color, marker in zip(['Exp pv mean', 'Exp pepi mean', 'CPAP mean'], ['g', 'b', 'r'], ['o', '*', 'v']):
#         x = df_temp['CPAP mean']
#         y = df_temp[outcome]
#         z1 = np.polyfit(x, y, 1)
#         p1 = np.poly1d(z1)   
#         plt.scatter(x, y, color = color, marker = marker)
#         plt.plot([min(x), max(x)], p1([min(x), max(x)]), linestyle = '--', color = color, lw = 1, label = '_nolegend_')
#         lgtxt.append('={:.2f}x{:+.2f}'.format(z1[0], z1[1]))
#         if outcome == 'Exp pv mean':
#             df_breaths['compensate_pv'] =  (1-z1[0]) *  df_breaths['CPAP mean'] - z1[1]
#         elif outcome == 'Exp pepi mean':
#             df_breaths['compensate_pepi'] =  (1-z1[0]) *  df_breaths['CPAP mean'] - z1[1]
#     # plt.plot(df_temp['CPAP mean'], df_temp['Exp pepi mean'], 'b*')
#     # plt.plot(df_temp['CPAP mean'], df_temp['CPAP mean'], 'rv')
    
#     plt.legend(["{}{}".format(a_,b_) for a_, b_ in zip(['Pv', 'Pepi', 'Pn'], lgtxt)])
#     plt.show()
    
#     # return df_breaths
    
# plot_exp_p(df_breaths)



# def plot_resistence(sig_p, sig_pn, sig_f, df_breaths, i):
      
#     # i = 19
#     res_start = df_breaths['Ins start'][i]
#     vimax_t = round((df_breaths['Flow max time'][i] - t_start)*sample_rate)
#     # ins_end = round((df_breaths['Pepi min time'][i] - t_start)*sample_rate) 
#     res_end = df_breaths['Exp end'][i]
    
#     sig_p_temp = sig_p[res_start : vimax_t]
#     # sig_p_temp2 = sig_p[vimax_t : ins_end]
#     sig_p_res = sig_p[res_start : res_end]
    
#     sig_flow_temp = sig_f[res_start : vimax_t]
#     # sig_flow_temp2 = sig_f[vimax_t : ins_end]
#     sig_flow_res = sig_f[res_start : res_end]
    
#     sig_pn_temp = sig_pn[res_start : vimax_t]
#     # sig_pn_temp2 = sig_pn[vimax_t : ins_end]
    
#     d_sig_p = sig_pn_temp-sig_p_temp     
#     # d_sig_p2 = sig_pn_temp2 - sig_p_temp2

#     img_arr1 = normalized_plot(sig_flow_res)
#     img_arr2 = normalized_plot(sig_p_res)
        
#     plt.style.use('default')
#     fig, ax = plt.subplots(nrows=1, ncols=1) 
    
#     ax.plot(sig_flow_temp, d_sig_p, 'go', alpha=0.5, markersize=4)
#     # ax.plot(d_sig_p2, sig_flow_temp2, 'b^', alpha=0.5, markersize=4)
    
#     x_ins = 0
#     y_ins = 0.75
    
#     ins1 = ax.inset_axes([x_ins,y_ins,0.2,0.2])
#     ins2 = ax.inset_axes([x_ins,y_ins - 0.25 ,0.2,0.2])
    
#     ins1.imshow(img_arr1)
#     ins2.imshow(img_arr2)
    
#     for ins in [ins1, ins2]:
#         # ins.set_xticklabels([])
#         ins.set_xticks([])
#         ins.set_yticks([])
#         ins.patch.set_alpha(0.1)
    
   
#     z = np.polyfit(sig_flow_temp, d_sig_p, 1)
#     p = np.poly1d(z)
         
#     ax.plot( [min(sig_flow_temp), max(sig_flow_temp)], p([min(sig_flow_temp), max(sig_flow_temp)]), linestyle = '--', color = 'g', lw = 1 )
    
#     ins1.text(0.8, 0.2, 'F', transform=ins1.transAxes)
#     ins2.text(0.8, 0.2, 'P', transform=ins2.transAxes)
#     ax.text(0.8, 0.1, "Ω = {:.2f}\nPn = {:.2f}".format(z[0], np.mean(sig_pn_temp)), transform=ax.transAxes)
    
#     ax.set_xlabel('Flow (L/min)')
#     ax.set_ylabel('Pressure gradient (cmH2O)')
#     ax.spines[['top', 'right']].set_visible(False)

#     # plt.gca().invert_xaxis()
#     plt.show()

# plot_resistence(sig_epi, sig_cpap, sig_flow, df_breaths, i=33)



# def plot_resistence_2ch(sig_p1, sig_p2, sig_pn, sig_f, df_breaths, i, show_fit = True, flow0 = True):
      
#     i = i-1
    
#     # i = 27
#     res_start = df_breaths['Ins start'][i]
#     vimax_t = round((df_breaths['Flow max time'][i] - t_start)*sample_rate)
    
#     res_end = df_breaths['Exp end'][i]
#     compensate_pv = df_breaths['compensate_pv'][i]
#     compensate_pepi = df_breaths['compensate_pepi'][i]
    
#     if not flow0:
#         ins_end = round((df_breaths['Pepi min time'][i] - t_start)*sample_rate) 
#     else:
#         ins_end = np.argmax(pd.Series(sig_f[vimax_t : res_end]).rolling(window=5).mean().values < 0) + vimax_t
    
#     sig_p1_temp = sig_p1[res_start : vimax_t] + compensate_pv
#     sig_p1_temp2 = sig_p1[vimax_t : ins_end] + compensate_pv
#     sig_p1_res = sig_p1[res_start : res_end] + compensate_pv
    
#     sig_p2_temp = sig_p2[res_start : vimax_t] + compensate_pepi
#     sig_p2_temp2 = sig_p2[vimax_t : ins_end] + compensate_pepi
#     sig_p2_res = sig_p2[res_start : res_end] + compensate_pepi
    
#     sig_flow_temp = sig_f[res_start : vimax_t]
#     sig_flow_temp2 = sig_f[vimax_t : ins_end]
#     sig_flow_res = sig_f[res_start : res_end]
    
#     sig_pn_temp = sig_pn[res_start : vimax_t]
#     sig_pn_temp2 = sig_pn[vimax_t : ins_end]
    
#     sig_p1_delta_1 = sig_pn_temp - sig_p1_temp     
#     sig_p1_delta_2 = sig_pn_temp2 - sig_p1_temp2
    
#     sig_p2_delta_1 = sig_pn_temp - sig_p2_temp     
#     sig_p2_delta_2 = sig_pn_temp2 - sig_p2_temp2

#     img_arr1 = normalized_plot(sig_flow_res)
#     img_arr2 = normalized_plot(sig_p1_res)
#     img_arr3 = normalized_plot(sig_p2_res)
        
#     plt.style.use('default')
#     fig = plt.figure(layout="constrained")
#     ax_dict  = fig.subplot_mosaic('''
#                                 AAAB
#                                 AAAC
#                                 AAAD
#                                 ''') 
    
#     ax_dict['A'].plot(sig_p1_temp, sig_flow_temp, color = 'g', linestyle='None', marker='o', alpha=0.5, markersize=4)
#     ax_dict['A'].plot(sig_p1_temp2, sig_flow_temp2, color = 'g', linestyle='None', marker='v', alpha=0.2, markersize=4, mfc='none')
#     ax_dict['A'].plot(sig_p2_temp, sig_flow_temp, color = 'b', linestyle='None', marker='o', alpha=0.5, markersize=4)
#     ax_dict['A'].plot(sig_p2_temp2, sig_flow_temp2, color = 'b', linestyle='None', marker='v', alpha=0.2, markersize=4, mfc='none')
    
#     ymin, _ = ax_dict['A'].get_ylim()
#     ax_dict['A'].annotate('Pn', 
#                           c = 'r',                         
#                           xy = (df_breaths['CPAP mean'][i], ymin), 
#                           annotation_clip=False,
#                           # xytext=(df_breaths['CPAP mean'][i], ymin - 0.5),  
#                           arrowprops = dict(facecolor='r', edgecolor='r', shrink=0.05, alpha = 0.5))
    
#     # ax_dict['A'].plot(df_breaths['CPAP mean'][i], ymin, 'rx', clip_on=False
#                      # )
      
#     ax_dict['B'].imshow(img_arr1)
#     ax_dict['C'].imshow(img_arr2)
#     ax_dict['D'].imshow(img_arr3)
    
#     for ins in ['B', 'C', 'D']:
#         # ins.set_xticklabels([])
#         ax_dict[ins].set_xticks([])
#         ax_dict[ins].set_yticks([])
#         # ax_dict[ins].patch.set_alpha(0.1)
      
#     ax_dict['B'].text(0.75, 0.2, 'F', transform=ax_dict['B'].transAxes)
#     ax_dict['C'].text(0.75, 0.2, 'Pv', transform=ax_dict['C'].transAxes, color = 'g')
#     ax_dict['D'].text(0.75, 0.2, 'Pepi', transform=ax_dict['D'].transAxes, color = 'b')
    
#     if show_fit:
#         z1 = np.polyfit(sig_flow_temp, sig_p1_delta_1, 1)
#         p1 = np.poly1d(z1)        
#         ax_dict['A'].plot(p1([min(sig_flow_temp), max(sig_flow_temp)]), [min(sig_flow_temp), max(sig_flow_temp)], 
#                           linestyle = '--', color = 'g', lw = 1 )
        
#         z2 = np.polyfit(sig_flow_temp, sig_p2_delta_1, 1)
#         p2 = np.poly1d(z2)        
#         ax_dict['A'].plot(p2([min(sig_flow_temp), max(sig_flow_temp)]), [min(sig_flow_temp), max(sig_flow_temp)], 
#                           linestyle = '--', color = 'b', lw = 1 )
           
#         ax_dict['A'].text(0.5, 0.1, "Ωv = {:.2f}\nΩepi = {:.2f}\nPn = {:.2f}".format(z1[0], z2[0], np.mean(sig_pn_temp)), transform=ax_dict['A'].transAxes)
        
#     ax_dict['A'].set_ylabel('Flow (L/min)')
#     ax_dict['A'].set_xlabel('Pressure (cmH2O)')
#     ax_dict['A'].spines[['top', 'right']].set_visible(False)
#     # plt.plot(df_breaths['CPAP mean'], df_breaths['Exp pepi mean'], 'b*')
#     # plt.plot(df_breaths['CPAP mean'], df_breaths['CPAP mean'], 'rv')
#     # plt.legend(['Pv', 'Pepi', 'Pn'])
#     # ax_dict['A'].invert_xaxis()
#     fig.suptitle(f'Breath #{i+1}')
#     # plt.tight_layout()
#     plt.show()

# plot_resistence_2ch(sig_pv, sig_epi, sig_cpap, sig_flow, df_breaths, i=45, show_fit = False, flow0 = True)





# def conv_smooth(y, box_pts):   
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth

# def get_spline(x,y):
#     tck, u  = interpolate.splprep( [x,y], s = 0 )
#     xnew, ynew = interpolate.splev(u, tck, der = 0)
#     return xnew, ynew





# def plot_resistence_multiple_b(sig_p1, sig_p2, sig_pn, sig_f, df_breaths, i_list, show_fit = True, flow0 = True, apply_fitler='lp',
#                                p_cutoff = 5, f_cutoff= 10):
#     plt.style.use('default')
#     fig = plt.figure(figsize = (12,6)) #layout="constrained"
#     ax_dict  = fig.subplot_mosaic('''
#                                 AAABEF
#                                 AAACGH
#                                 AAADIJ
#                                 ''')  
                            
#     if apply_fitler == 'lp':                        
#         sig_p1 =  low_pass_filter(sig_p1, sample_rate, cutoff=p_cutoff) 
#         sig_p2 =  low_pass_filter(sig_p2, sample_rate, cutoff=p_cutoff)
#         sig_pn = low_pass_filter(sig_pn, sample_rate, cutoff=p_cutoff)
#         sig_f = low_pass_filter(sig_f, sample_rate, cutoff=f_cutoff)
#     elif apply_fitler == 'ma': 
#         sig_p1 = conv_smooth(sig_p1, p_cutoff)
#         sig_p2 = conv_smooth(sig_p2, p_cutoff)
#         sig_pn = conv_smooth(sig_pn, p_cutoff)
#         sig_f = conv_smooth(sig_f, f_cutoff)
        
                                    
#     pn_list = []                    
#     for num, i in enumerate(i_list):
        
#         if num == 0:
#             inset_list = ['B', 'C', 'D']
#         elif num == 1:
#             inset_list = ['E', 'G', 'I']
#         else:
#             inset_list = ['F', 'H', 'J']
        
#         i = i-1
        
#         # i = 27
#         res_start = df_breaths['Ins start'][i] - 30
#         vimax_t = round((df_breaths['Flow max time'][i] - t_start)*sample_rate)
        
#         res_end = df_breaths['Exp end'][i]
#         compensate_pv = df_breaths['compensate_pv'][i]
#         compensate_pepi = df_breaths['compensate_pepi'][i]
#         pn_list.append(df_breaths['CPAP mean'][i])
#         if not flow0:
#             ins_end = round((df_breaths['Pepi min time'][i] - t_start)*sample_rate) 
#         else:
#             ins_end = np.argmax(pd.Series(sig_f[vimax_t : res_end]).rolling(window=5).mean().values < 0) + vimax_t
        
#         sig_p1_temp = sig_p1[res_start : vimax_t] + compensate_pv
#         sig_p1_temp2 = sig_p1[vimax_t : ins_end] + compensate_pv
#         sig_p1_temp3 = sig_p1[res_start : ins_end] + compensate_pv
#         sig_p1_res = sig_p1[res_start : res_end] + compensate_pv
        
#         sig_p2_temp = sig_p2[res_start : vimax_t] + compensate_pepi
#         sig_p2_temp2 = sig_p2[vimax_t : ins_end] + compensate_pepi
#         sig_p2_temp3 = sig_p2[res_start : ins_end] + compensate_pv
#         sig_p2_res = sig_p2[res_start : res_end] + compensate_pepi
        
#         sig_flow_temp = sig_f[res_start : vimax_t]
#         sig_flow_temp2 = sig_f[vimax_t : ins_end]
#         sig_flow_temp3 = sig_f[res_start : ins_end]
#         sig_flow_res = sig_f[res_start : res_end]
        
#         sig_pn_temp = sig_pn[res_start : vimax_t]
#         sig_pn_temp2 = sig_pn[vimax_t : ins_end]
#         sig_pn_temp3 = sig_pn[res_start : ins_end]
    
#         img_arr1 = normalized_plot(sig_flow_res)
#         img_arr2 = normalized_plot(sig_p1_res)
#         img_arr3 = normalized_plot(sig_p2_res)
        
#         sig_p1_temp_sp_b4, sig_flow_temp_sp_b4_1 = get_spline(sig_p1_temp, sig_flow_temp)
#         sig_p2_temp_sp_b4, sig_flow_temp_sp_b4_2 = get_spline(sig_p2_temp, sig_flow_temp)
#         sig_p1_temp_sp_af, sig_flow_temp_sp_af_1 = get_spline(sig_p1_temp2, sig_flow_temp2)
#         sig_p2_temp_sp_af, sig_flow_temp_sp_af_2 = get_spline(sig_p2_temp2, sig_flow_temp2)
            
#         # ax_dict['A'].plot(sig_p1_temp, sig_flow_temp, color = 'g', linestyle='None', marker='o', alpha=0.2, markersize=2)
#         # ax_dict['A'].plot(sig_p1_temp2, sig_flow_temp2, color = 'g', linestyle='None', marker='v', alpha=0.2, markersize=2, mfc='none')
#         # ax_dict['A'].plot(sig_p2_temp, sig_flow_temp, color = 'b', linestyle='None', marker='o', alpha=0.2, markersize=2)
#         # ax_dict['A'].plot(sig_p2_temp2, sig_flow_temp2, color = 'b', linestyle='None', marker='v', alpha=0.2, markersize=2, mfc='none')
        
#         ax_dict['A'].plot(sig_p1_temp_sp_b4, sig_flow_temp_sp_b4_1, color = 'g', linestyle='-', alpha=0.5)
#         ax_dict['A'].plot(sig_p2_temp_sp_b4, sig_flow_temp_sp_b4_2, color = 'b', linestyle='-', alpha=0.5)
#         ax_dict['A'].plot(sig_p1_temp_sp_af, sig_flow_temp_sp_af_1, color = 'g', linestyle='--', alpha=0.5)
#         ax_dict['A'].plot(sig_p2_temp_sp_af, sig_flow_temp_sp_af_2, color = 'b', linestyle='--', alpha=0.5)
        
#         ax_dict['A'].plot(sig_p1_temp2[0], sig_flow_temp2[0], color = 'y', marker ='o', alpha=0.5)
#         ax_dict['A'].plot(sig_p2_temp2[0], sig_flow_temp2[0], color = 'y', marker ='o', alpha=0.5)
        
#         ax_dict[inset_list[0]].imshow(img_arr1)
#         ax_dict[inset_list[1]].imshow(img_arr2)
#         ax_dict[inset_list[2]].imshow(img_arr3)
#         ax_dict[inset_list[0]].text(0.75, 0.2, 'F', transform=ax_dict[inset_list[0]].transAxes)
#         ax_dict[inset_list[1]].text(0.75, 0.2, 'Pv', transform=ax_dict[inset_list[1]].transAxes, color = 'g')
#         ax_dict[inset_list[2]].text(0.75, 0.2, 'Pepi', transform=ax_dict[inset_list[2]].transAxes, color = 'b')
        
#         ax_dict[inset_list[0]].text(0.5, 1.1, str(i+1), transform=ax_dict[inset_list[0]].transAxes)
            
#     ymin, _ = ax_dict['A'].get_ylim()
#     for num, pn in enumerate(pn_list):
#         ax_dict['A'].annotate(f'Pn{i_list[num]}', 
#                               c = 'r',                         
#                               xy = (pn, ymin), 
#                               annotation_clip=False,
#                               # xytext=(df_breaths['CPAP mean'][i], ymin - 0.5),  
#                               arrowprops = dict(facecolor='r', edgecolor='r', shrink=0.05, alpha = 0.5))
   
#     for ins in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
#         # ins.set_xticklabels([])
#         ax_dict[ins].set_xticks([])
#         ax_dict[ins].set_yticks([])
#         # ax_dict[ins].patch.set_alpha(0.1)
    
#     if show_fit:
#         sig_p1_delta_1 = sig_pn_temp - sig_p1_temp     
#         # sig_p1_delta_2 = sig_pn_temp2 - sig_p1_temp2
        
#         sig_p2_delta_1 = sig_pn_temp - sig_p2_temp     
#         # sig_p2_delta_2 = sig_pn_temp2 - sig_p2_temp2
        
#         z1 = np.polyfit(sig_flow_temp, sig_p1_delta_1, 1)
#         p1 = np.poly1d(z1)        
#         ax_dict['A'].plot(p1([min(sig_flow_temp), max(sig_flow_temp)]), [min(sig_flow_temp), max(sig_flow_temp)], 
#                           linestyle = '--', color = 'g', lw = 1 )
        
#         z2 = np.polyfit(sig_flow_temp, sig_p2_delta_1, 1)
#         p2 = np.poly1d(z2)        
#         ax_dict['A'].plot(p2([min(sig_flow_temp), max(sig_flow_temp)]), [min(sig_flow_temp), max(sig_flow_temp)], 
#                           linestyle = '--', color = 'b', lw = 1 )
           
#         ax_dict['A'].text(0.5, 0.1, "Ωv = {:.2f}\nΩepi = {:.2f}\nPn = {:.2f}".format(z1[0], z2[0], np.mean(sig_pn_temp)), transform=ax_dict['A'].transAxes)
    
#     # ax_dict['A'].set_xlim(xmin=0, xmax=8)
#     ax_dict['A'].set_ylabel('Flow (L/min)')
#     ax_dict['A'].set_xlabel('Pressure (cmH2O)')
#     ax_dict['A'].spines[['top', 'right']].set_visible(False)
#     # plt.plot(df_breaths['CPAP mean'], df_breaths['Exp pepi mean'], 'b*')
#     # plt.plot(df_breaths['CPAP mean'], df_breaths['CPAP mean'], 'rv')
#     # plt.legend(['Pv', 'Pepi', 'Pn'])
#     # ax_dict['A'].invert_xaxis()
#     fig.suptitle(f'Breaths #{i_list}')
#     fig.tight_layout()
#     plt.show()



# plot_resistence_multiple_b(sig_pv, sig_epi, sig_cpap, sig_flow, df_breaths, i_list = [14, 27, 51], show_fit = False, flow0 = True,
#                            apply_fitler='ma', p_cutoff = 10, f_cutoff= 10)
