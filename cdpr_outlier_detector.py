import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import traceback
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
import warnings
from scipy.spatial import distance
from sklearn.decomposition import PCA
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['agg.path.chunksize'] = 10000
M=5
DIMPCA = 4*M
SOLOUNA=True

# Regular expressions
rx_dict = {
    'tiempo': re.compile(r'[Tt]iempo:? (?P<tiempo>\d+) s\n'),
    'tcalibracion': re.compile(r'[Cc]alibracion:? (?P<tcalibracion>\d+) s\n'),
    'tsync': re.compile(r'[Ss]ync:? (?P<tsync>[-\d]+.\d+) s\n'),
    'label': re.compile(r'[Ll]abel:? (?P<label>[\d,\,]*)\n'),
    'record': re.compile(
        r'#(?P<num>\d+), starttimeE: (?P<abstime>.*), starttime: (?P<starttime>\d+.\d+)s, [Dd]ata:? (?P<data>\(.*\)), [Pp]osition:? (?P<position>\(.*\)), [Vv]elocity:? (?P<velocity>\(.*\))\n'
    ),
    'undetected': re.compile(r'[Uu]ndetected:? (?P<undetected>\d,\s*\d)\n')
}

def _parse_line(line):
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    return None, None

def parse_file(filepath):
    alldata = []
    label = [0]
    undetected = [0]
    tsync = 0.0

    with open(filepath, 'r') as file_object:
        line = file_object.readline()

        while line:
            key, match = _parse_line(line)

            if key == 'tiempo':
                tiempo = int(match.group('tiempo'))

            if key == 'tcalibracion':
                tcalibracion = int(match.group('tcalibracion'))

            if key == 'tsync':
                tsync = float(match.group('tsync'))

            if key == 'label':
                label = np.array(list(map(int, match.group('label').split(','))))

            if key == 'undetected':
                undetected = np.array(list(map(int, match.group('undetected').split(','))))

            if key == 'record':
                numsample = int(match.group('num'))
                starttime = float(match.group('starttime'))
                abstime = match.group('abstime')
                data = match.group('data')
                position = match.group('position')
                velocity = match.group('velocity')
                row = {
                    'starttime': starttime,
                    'abstime': abstime,
                    'data': data,
                    'position': position,
                    'velocity': velocity,
                }
                row['calibracion'] = starttime < tcalibracion

                alldata.append(row)

            line = file_object.readline()

        alldata = pd.DataFrame(alldata)
        cols_to_convert = ['data', 'position', 'velocity']
        for col in cols_to_convert:
            alldata[col] = alldata[col].apply(lambda s: np.array(literal_eval(s)))
        print(alldata.head())
        return alldata, tiempo, tcalibracion, tsync, label, undetected

def getpower(signal):
    numsens = signal[0].shape[0]
    return np.sqrt(np.concatenate(signal**2).reshape(-1, numsens).sum(axis=1))

def getrepeat(X, M, overlap=1):
    n, d = X.shape
    if M > n:
        raise ValueError("M cannot be greater than the number of samples in X.")

    if overlap == 0:
        overlap = M

    rows = (n - M) // overlap + 1
    output = np.empty((rows, d * M))

    for i, j in enumerate(range(0, n - M + 1, overlap)):
        output[i] = X[j:j+M].ravel()

    return output

def bestGMM(X):
    Kmin = 1
    bicmin = float('inf')
    aicmin = float('inf')
    bgmm = None
    for K in range(1, 6):
        gmm = GaussianMixture(n_components=K).fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        if bic < bicmin:
            Kmin = K
            bicmin = bic
            aicmin = aic
            bgmm = gmm
    #print(f'GMM --> Kmin: {Kmin}, BICmin: {bicmin}, AICmin: {aicmin}')
    return bgmm

def mahalanobis_to_gmm_centers(X, gmm):
    means = gmm.means_
    covariances = gmm.covariances_

    all_distances = np.zeros((X.shape[0], means.shape[0]))
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        inv_covar = np.linalg.inv(covar)
        d = [distance.mahalanobis(x, mean, inv_covar) for x in X]
        all_distances[:, i] = d

    distances = np.min(all_distances, axis=1)

    return distances

def calculamodelo(X, potencias):
    pca = PCA(n_components=DIMPCA)
    X = pca.fit_transform(X)
    gmm = bestGMM(X)
    dMAX = np.max(mahalanobis_to_gmm_centers(X, gmm))
    pMAX = np.max(potencias)
    return gmm, pca, dMAX, pMAX

def decimal_a_hora(decimal_hora):
    horas = int(decimal_hora)
    minutos_decimales = (decimal_hora - horas) * 60
    minutos = int(minutos_decimales)
    segundos = int((minutos_decimales - minutos) * 60)
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

def processfile(archivo, M=1, fastmode=True, DIMPCA=5, WINDOWSIZE=100):
    pd.options.mode.chained_assignment = None
    archivobase = os.path.splitext(archivo)[0]
    picklefile = f'{archivobase}/data.pickle'
    epsilon = 1.0
    gamma = 0.15

    try:
        with open(picklefile, 'rb') as f:
            data, tiempo, tcalibracion, tsync, labels, undetected = pickle.load(f)
            print(f'Data already available for {archivo}')
    except:
        print(f'Parsing {archivo}')
        if not os.path.exists(f'{archivobase}'):
            os.mkdir(f'{archivobase}')

        data, tiempo, tcalibracion, tsync, labels, undetected = parse_file(archivo)
        with open(picklefile, 'wb') as f:
            pickle.dump([data, tiempo, tcalibracion, tsync, labels, undetected], f)

    #print(data.head())

    # Apply fastmode
    if fastmode:
        num_samples = len(data)
        limit = int(num_samples * 0.1)
        data = data.iloc[:limit]
        print(f"Fast mode active: processing the first {limit} samples out of {num_samples}")

    print(f'Applying model. ', end='')
    numsens = len(data['data'].iloc[0])
    print(f'numsens: {numsens}')

    # Time in hours
    data['time_hours'] = data['starttime'] / 3600.0
    time = np.array(data['time_hours'])

    data['power'] = getpower(data['data'])

    data['drel'] = 1
    data['prel'] = 1
    data['drelupdate'] = 1
    data['prelupdate'] = 1
    data['umbralmodelo'] = 0

    # FIRST MODEL: COMPARISON WITH CALIBRATION ZONE
    X_calib_data = data[data['calibracion'] == True]
    X = np.vstack(data[data['calibracion'] == True].data.values)
    XX = np.vstack(data.data.values)
    X = getrepeat(X, M, overlap=1)
    XX = getrepeat(XX, M, overlap=1)

    # DIMENSIONALITY REDUCTION
    pca = PCA(n_components=DIMPCA)
    X = pca.fit_transform(X)
    XX = pca.transform(XX)

    #print(f'#CALIBRATION DATA: {X.shape[0]}')

    # Gaussian Model
    gmm = bestGMM(X)

    # Distances
    dMAX = np.max(mahalanobis_to_gmm_centers(X, gmm))
    distancias = np.concatenate([[0]*(M-1), mahalanobis_to_gmm_centers(XX, gmm).ravel()])

    # Relative distances and power
    data['drelcalibracion'] = distancias / dMAX

    #potencias = np.vstack(data['power'].values).ravel()
    #pMAX = np.max(data[data['calibracion'] == True].power.values)
    potencias = np.vstack(data['power'].rolling(window=WINDOWSIZE, min_periods=1).mean().values).ravel() 
    pMAX = np.max(X_calib_data['power'].rolling(window=WINDOWSIZE, min_periods=1).mean().values[WINDOWSIZE:])
    data['prelcalibracion'] = potencias / pMAX

    # Define separate SAMPLESW for calibration data
    SAMPLESW_calib = X.shape[0]

    # Subfigure 1: Moving Average of Relative Power
    data['moving_avg_prelcalibracion'] = data['prelcalibracion'].rolling(window=SAMPLESW_calib, min_periods=1).mean()
    data['anomalia_power_moving_avg'] = data['moving_avg_prelcalibracion'][SAMPLESW_calib:] > 1.0

    # Subfigure 2: Moving Average of Relative Mahalanobis Distance (Model Without Updates)
    data['moving_avg_drelcalibracion'] = data['drelcalibracion'].rolling(window=SAMPLESW_calib, min_periods=1).mean()
    data['anomalia_drelcalibracion_moving_avg'] = data['moving_avg_drelcalibracion'] > 1.0

    # SECOND MODEL: MODEL UPDATES WITH NEW EVIDENCE
    SAMPLESW = np.vstack(data[data['calibracion'] == True].data.values).shape[0]-M+1
    DELTAW = 6*SAMPLESW

    #print(f'SAMPLESW: {SAMPLESW}, DELTAW: {DELTAW}')
    pca = PCA(n_components=DIMPCA)

    index = 0
    inicio = 1
    step = DELTAW
    tcalculo = []
    while index+DELTAW-1 < len(data.index):
        print(f'Processing from {index} to {index+DELTAW-1}, step: {step}')
        X = np.vstack(data['data'].loc[index-M+1:index+DELTAW-1].values)

        X = getrepeat(X, M, overlap=1)

        if inicio:
            print(f'Calculating updated model')
            Xcalib = np.vstack(data['data'].loc[index:index+SAMPLESW-1].values)
            Xcalib = getrepeat(Xcalib, M, overlap=1)
            potencias = np.vstack(data['power'].loc[index:index+SAMPLESW-1].values).ravel()
            [gmm, pca, dMAX, pMAX] = calculamodelo(Xcalib, potencias)
            inicio = 0
            step = DELTAW

        X = pca.transform(X)

        if index == 0:
            distancias = np.concatenate([[0]*(M-1), mahalanobis_to_gmm_centers(X, gmm).ravel()])
        else:
            distancias = mahalanobis_to_gmm_centers(X, gmm).ravel()

        data['drelupdate'].loc[index:index+DELTAW-1] = distancias[:DELTAW] / dMAX
        data['umbralmodelo'].loc[index:index+DELTAW-1] = np.mean(distancias[:SAMPLESW] / dMAX)
        potencias = np.vstack(data['power'].loc[index:index+DELTAW-1].values).ravel()
        data['prelupdate'].loc[index:index+DELTAW-1] = potencias[:DELTAW] / pMAX

        if np.max(distancias / dMAX) > (1.0+epsilon):
            print(f'Anomaly detected in window starting at {index}')
            step = DELTAW
        else:
            if np.mean(distancias[:SAMPLESW] / dMAX) > (1.0 - gamma):
                inicio = 1
                tcalculo.append(time[index])
                #step = 0 -> Hay situaciones donde con esto no avanza
                step = WINDOWSIZE

        index = index+step

    # Anomalies for second model
    data['anomalia_update'] = (data['drelupdate'] > (1+epsilon))
    #print(f'SECOND MODEL RESULT:')

    # NEW ANOMALY DETECTION USING MOVING AVERAGE
    data['moving_avg_drelupdate'] = data['drelupdate'].rolling(window=SAMPLESW, min_periods=1).mean()
    data['anomalia_moving_avg'] = data['moving_avg_drelupdate'] > 1.0

    # Plots
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 22}
    plt.rc('font', **font)
    params = {'axes.labelsize': 22, 'axes.titlesize': 22, 'xtick.labelsize': 22, 'ytick.labelsize': 22}
    matplotlib.rcParams.update(params)
    if SOLOUNA:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    else:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 20))

    moving_avg_powercalib = np.array(data['moving_avg_prelcalibracion'])
    moving_avg_distcalib = np.array(data['moving_avg_drelcalibracion'])
    moving_avg_distupdate = np.array(data['moving_avg_drelupdate'])

    anomaly_threshold = 1.0
    threshold_color = 'red'

    if SOLOUNA:
        # Subfigure 3: Moving Average of Mahalanobis Distance (Model With Updates)
        ax.plot(time[SAMPLESW:], moving_avg_distupdate[SAMPLESW:], color='tab:cyan', linewidth=1) #, label=f'M: {M}, \gamma: {gamma}')
        ax.hlines(1.0, np.min(time), np.max(time), color=threshold_color, linestyles='dashed', linewidth=2, label='Anomaly Threshold')
        ax.set_ylabel('Relative Distance')
        ax.set_xlabel('Time (hours)')
        ax.set_ylim(0,16)
        #ax[2].set_title(f'Moving Average over last {SAMPLESW} Samples')
        #ax.set_title(f'M: {M}, $\\gamma$: {gamma}')

        primera=True
        for t in tcalculo:
            if primera: 
                ax.scatter(t, 1.2, color='red', s=120, marker='v', label='Model update') 
                primera=False
            else:
                ax.scatter(t, 1.2, color='red', s=120, marker='v') 
        
        # Highlight anomalies in Subfigure 3
        #import datetime
        
        # Detecta zonas anómalas y las une si están a menos de 1 minuto entre ellas
        anomalies = data['anomalia_moving_avg']
        in_anomaly = False
        start_anomaly = None
        end_anomaly = None
        count = 0
        time_threshold = 2/60  # Umbral de 1 minuto para unir anomalías
        
        with open(f'{archivo}-anomalias.log', 'a') as file:
            file.write('MODELO CON ACTUALIZACIONES\n')
        
            i = 0
            while i < len(anomalies):
                if anomalies.iloc[i]:
                    if not in_anomaly:
                        # Inicio de una nueva anomalía
                        in_anomaly = True
                        start_anomaly = time[i]
                        count += 1
                        print(f'{archivo}--> Modelo con actualizaciones. Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-', end='')
                        file.write(f'Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-')
                    end_anomaly = time[i]  # Actualiza el fin de la anomalía
                    i += 1
                else:
                    if in_anomaly:
                        # Posible fin de anomalía, verificar si se une con la siguiente
                        next_i = i + 1
                        while next_i < len(anomalies) and not anomalies.iloc[next_i]:
                            next_i += 1
                        if next_i < len(anomalies):
                            # Hay una siguiente anomalía
                            time_diff = time[next_i] - end_anomaly
                            if time_diff <= time_threshold:
                                # La siguiente anomalía está dentro del umbral, no cerrar aún
                                i = next_i
                                continue
                        # Cierra la anomalía actual
                        in_anomaly = False
                        print(f'{decimal_a_hora(end_anomaly)} h')
                        file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                        ax.axvspan(start_anomaly - 0.01, end_anomaly + 0.01,
                                      color='orange', alpha=0.1, linewidth=0)
                        start_anomaly = None
                        end_anomaly = None
                    i += 1
        
            # Si hay una anomalía abierta al finalizar
            if in_anomaly:
                print(f'{decimal_a_hora(end_anomaly)} h')
                file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                ax.axvspan(start_anomaly - 0.01, end_anomaly + 0.01,
                              color='orange', alpha=0.1, linewidth=0)
        

        #ax.legend(loc='upper left')
    else:
        # Subfigure 1: Moving Average of Relative Power
        ax[0].plot(time[SAMPLESW:], moving_avg_powercalib[SAMPLESW:], color='tab:blue', linewidth=1, label='Moving Average Relative Power')
        ax[0].hlines(anomaly_threshold, np.min(time), np.max(time), color=threshold_color, linestyles='dashed', linewidth=2, label='Anomaly Threshold')
        ax[0].set_xlabel('Time (hours)')
        ax[0].set_ylabel('Relative Power')
        ax[0].set_title(f'Moving Average Relative Power')
        ax[0].set_ylim([np.min(moving_avg_powercalib[SAMPLESW:])*0.95, np.max(moving_avg_powercalib[SAMPLESW:])*1.05])
        ax[0].legend(loc='upper left')

        # Highlight anomalies in Subfigure 1
        anomalies = data['anomalia_power_moving_avg']
        in_anomaly = False
        start_anomaly = None
        end_anomaly = None
        count = 0
        time_threshold = 2/60  # Umbral de 1 minuto para unir anomalías
        
        with open(f'{archivo}-anomalias.log', 'w') as file:
            file.write('MODELO POTENCIA\n')
        
            i = 0
            while i < len(anomalies):
                if anomalies.iloc[i]:
                    if not in_anomaly:
                        # Inicio de una nueva anomalía
                        in_anomaly = True
                        start_anomaly = time[i]
                        count += 1
                        print(f'{archivo}--> Modelo potencia. Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-', end='')
                        file.write(f'Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-')
                    end_anomaly = time[i]  # Actualiza el fin de la anomalía
                    i += 1
                else:
                    if in_anomaly:
                        # Posible fin de anomalía, verificar si se une con la siguiente
                        next_i = i + 1
                        while next_i < len(anomalies) and not anomalies.iloc[next_i]:
                            next_i += 1
                        if next_i < len(anomalies):
                            # Hay una siguiente anomalía
                            time_diff = time[next_i] - end_anomaly
                            if time_diff <= time_threshold:
                                # La siguiente anomalía está dentro del umbral, no cerrar aún
                                i = next_i
                                continue
                        # Cierra la anomalía actual
                        in_anomaly = False
                        print(f'{decimal_a_hora(end_anomaly)} h')
                        file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                        ax[0].axvspan(start_anomaly - 0.01, end_anomaly + 0.01, color='orange', alpha=0.1, linewidth=0)
                        start_anomaly = None
                        end_anomaly = None
                    i += 1
        
            # Si hay una anomalía abierta al finalizar
            if in_anomaly:
                print(f'{decimal_a_hora(end_anomaly)} h')
                file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                ax[0].axvspan(start_anomaly - 0.01, end_anomaly + 0.01, color='orange', alpha=0.1, linewidth=0)

            #file.write(f'\n')

        # Subfigure 2: Moving Average Mahalanobis Distance (Model Without Updates)
        ax[1].plot(time[SAMPLESW:], moving_avg_distcalib[SAMPLESW:], color='tab:green', linewidth=1, label='Moving Average Relative Mahalanobis Distance')
        ax[1].hlines(anomaly_threshold, np.min(time), np.max(time), color=threshold_color, linestyles='dashed', linewidth=2, label='Anomaly Threshold')
        ax[1].set_ylabel('Relative Distance')
        ax[1].set_xlabel('Time (hours)')
        ax[1].set_title(f'Moving Average Mahalanobis Distance (Model Without Updates)')
        ax[1].legend(loc='upper left')

        # Highlight anomalies in Subfigure 2
        anomalies = data['anomalia_drelcalibracion_moving_avg']
        in_anomaly = False
        start_anomaly = None
        end_anomaly = None
        count = 0
        time_threshold = 2/60  # Umbral de 1 minuto para unir anomalías
        
        with open(f'{archivo}-anomalias.log', 'a') as file:
            file.write('MODELO SIN ACTUALIZACIONES\n')
        
            i = 0
            while i < len(anomalies):
                if anomalies.iloc[i]:
                    if not in_anomaly:
                        # Inicio de una nueva anomalía
                        in_anomaly = True
                        start_anomaly = time[i]
                        count += 1
                        print(f'{archivo}--> Modelo sin actualizaciones. Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-', end='')
                        file.write(f'Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-')
                    end_anomaly = time[i]  # Actualiza el fin de la anomalía
                    i += 1
                else:
                    if in_anomaly:
                        # Posible fin de anomalía, verificar si se une con la siguiente
                        next_i = i + 1
                        while next_i < len(anomalies) and not anomalies.iloc[next_i]:
                            next_i += 1
                        if next_i < len(anomalies):
                            # Hay una siguiente anomalía
                            time_diff = time[next_i] - end_anomaly
                            if time_diff <= time_threshold:
                                # La siguiente anomalía está dentro del umbral, no cerrar aún
                                i = next_i
                                continue
                        # Cierra la anomalía actual
                        in_anomaly = False
                        print(f'{decimal_a_hora(end_anomaly)} h')
                        file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                        ax[1].axvspan(start_anomaly - 0.01, end_anomaly + 0.01, color='orange', alpha=0.1, linewidth=0)
                        start_anomaly = None
                        end_anomaly = None
                    i += 1
        
            # Si hay una anomalía abierta al finalizar
            if in_anomaly:
                print(f'{decimal_a_hora(end_anomaly)} h')
                file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                ax[1].axvspan(start_anomaly - 0.01, end_anomaly + 0.01, color='orange', alpha=0.1, linewidth=0)

        # Subfigure 3: Moving Average of Mahalanobis Distance (Model With Updates)
        ax[2].plot(time[SAMPLESW:], moving_avg_distupdate[SAMPLESW:], color='tab:cyan', linewidth=1, label='Moving Average Relative Mahalanobis Distance')
        ax[2].hlines(1.0, np.min(time), np.max(time), color=threshold_color, linestyles='dashed', linewidth=2, label='Anomaly Threshold')
        ax[2].set_ylabel('Relative Distance')
        ax[2].set_xlabel('Time (hours)')
        #ax[2].set_title(f'Moving Average over last {SAMPLESW} Samples')
        ax[2].set_title(f'Moving Average Mahalanobis Distance (Model With Updates)')

        primera=True
        for t in tcalculo:
            if primera: 
                ax[2].scatter(t, 1.2, color='red', s=120, marker='v', label='Model update') 
                primera=False
            else:
                ax[2].scatter(t, 1.2, color='red', s=120, marker='v') 
        
        ax[2].legend(loc='upper left')
            

        # Highlight anomalies in Subfigure 3
        #import datetime
        
        # Detecta zonas anómalas y las une si están a menos de 1 minuto entre ellas
        anomalies = data['anomalia_moving_avg']
        in_anomaly = False
        start_anomaly = None
        end_anomaly = None
        count = 0
        time_threshold = 2/60  # Umbral de 1 minuto para unir anomalías
        
        with open(f'{archivo}-anomalias.log', 'a') as file:
            file.write('MODELO CON ACTUALIZACIONES\n')
        
            i = 0
            while i < len(anomalies):
                if anomalies.iloc[i]:
                    if not in_anomaly:
                        # Inicio de una nueva anomalía
                        in_anomaly = True
                        start_anomaly = time[i]
                        count += 1
                        print(f'{archivo}--> Modelo con actualizaciones. Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-', end='')
                        file.write(f'Anomalía {count}. Time: {decimal_a_hora(start_anomaly)}-')
                    end_anomaly = time[i]  # Actualiza el fin de la anomalía
                    i += 1
                else:
                    if in_anomaly:
                        # Posible fin de anomalía, verificar si se une con la siguiente
                        next_i = i + 1
                        while next_i < len(anomalies) and not anomalies.iloc[next_i]:
                            next_i += 1
                        if next_i < len(anomalies):
                            # Hay una siguiente anomalía
                            time_diff = time[next_i] - end_anomaly
                            if time_diff <= time_threshold:
                                # La siguiente anomalía está dentro del umbral, no cerrar aún
                                i = next_i
                                continue
                        # Cierra la anomalía actual
                        in_anomaly = False
                        print(f'{decimal_a_hora(end_anomaly)} h')
                        file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                        ax[2].axvspan(start_anomaly - 0.01, end_anomaly + 0.01,
                                      color='orange', alpha=0.1, linewidth=0)
                        start_anomaly = None
                        end_anomaly = None
                    i += 1
        
            # Si hay una anomalía abierta al finalizar
            if in_anomaly:
                print(f'{decimal_a_hora(end_anomaly)} h')
                file.write(f'{decimal_a_hora(end_anomaly)} h\n')
                ax[2].axvspan(start_anomaly - 0.01, end_anomaly + 0.01,
                              color='orange', alpha=0.1, linewidth=0)
        
    # anomalies = data['anomalia_moving_avg']
    # in_anomaly = False
    # start_anomaly = None
    # count = 0
    # with open(f'{archivo}-anomalias.log','w') as file:
    #     file.write('MODELO CON ACTUALIZACIONES\n')
    #     for i in range(len(anomalies)):
    #         if anomalies.iloc[i] and not in_anomaly:
    #             in_anomaly = True
    #             start_anomaly = time[i]
    #             count += 1
    #             print(f'{archivo}--> Modelo con actualizaciones. Anomalía {count}. Time: {decimal_a_hora(time[i])}-', end='')
    #             file.write(f'Anomalía {count}. Time: {decimal_a_hora(time[i])}-')
    #         elif not anomalies.iloc[i] and in_anomaly:
    #             in_anomaly = False
    #             end_anomaly = time[i]
    #             print(f'{decimal_a_hora(time[i])} h')
    #             file.write(f'{decimal_a_hora(time[i])} h\n')
    #             ax[2].axvspan(start_anomaly-0.01, end_anomaly+0.01, color='orange', alpha=0.1, linewidth=0)
    #     if in_anomaly:
    #         ax[2].axvspan(start_anomaly-0.01, time[-1], color='orange', alpha=0.1, linewidth=0)

    fig.tight_layout()

    fig.savefig(f'{archivobase}/{archivobase}-SUBFIG.pdf', dpi=300)
    plt.close(fig)

    return archivobase

''' ---------------------------------------------------------------------
                                   MAIN
--------------------------------------------------------------------- '''

def main():
    current_dir = os.getcwd()  # Obtiene el directorio actual
    #ficheros = [f for f in os.listdir(current_dir) if f.endswith('.txt')]   # Filtra los archivos .txt
    ficheros = {'experiment1.csv'}
    plt.close('all')

    for fichero in ficheros:
        # Envía cada llamada de processfile a un hilo independiente
        try:
            processfile(fichero, M=M, fastmode=False, WINDOWSIZE=100, DIMPCA=DIMPCA)
        except Exception as e: 
            # Imprimir el tipo de excepción
            print(f"Tipo de excepción: {type(e).__name__}")
            # Imprimir el mensaje de la excepción
            print(f"Mensaje: {e}")

if __name__ == "__main__":
    main()
