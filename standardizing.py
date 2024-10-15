import numpy as np
from sklearn.preprocessing import StandardScaler

def Diabetes_Normalize(data):
    #Re-create the scaler with the same parameters
    new_scaler = StandardScaler()

    mean = [  3.8990228 , 121.93648208 , 69.06026059 , 20.25895765 , 82.04885993,31.81042345,   0.4754544 ,  33.21335505]
    scale = [  3.37835113 , 31.80325621 , 19.49988601,  16.17845441 ,120.07301344,7.81549589 ,  0.34034154 , 11.5988213 ]

    new_scaler.mean_ = mean
    new_scaler.scale_ = scale

    #Transforming
    data = np.array(data).reshape(1,-1)
    x_new_scaled = new_scaler.transform(data)

    return (np.array(x_new_scaled).reshape(1,-1))


def Breast_Normalize(data):
    #Re-create the scaler with the same parameters
    new_scaler = StandardScaler()

    mean = [3.22485602e+07,1.41176352e+01, 1.91850330e+01, 9.18822418e+01, 6.54377582e+02, 9.57440220e-02, 1.03619319e-01, 888981451e-02, 4.82798703e-02, 1.81098681e-01, 6.27567692e-02, 4.02015824e-01, 1.20268681e+00, 2.85825341e+00, 4.00712989e+01, 6.98907473e-03,2.56354484e-02, 3.28236723e-02, 1.18939407e-02, 2.05735121e-02,
 3.82045560e-03, 1.62351033e+01, 2.55356923e+01, 1.07103121e+02,
 8.76987033e+02, 1.31532132e-01, 2.52741802e-01, 2.74594569e-01,
 1.14182222e-01, 2.90502198e-01, 8.38678462e-02]
    scale = [1.32372484e+08, 3.53192761e+00, 4.26131404e+00, 2.42952845e+01,
 3.54552925e+02, 1.39076981e-02, 5.24128055e-02, 7.93805091e-02,
 3.80183541e-02, 2.74570850e-02, 7.20178506e-03, 2.82849558e-01,
 5.41151676e-01, 2.06893139e+00, 4.71843820e+01, 3.05347371e-03,
 1.85862970e-02, 3.21102454e-02, 6.28718721e-03, 8.16296642e-03,
 2.78406874e-03, 4.80597715e+00, 6.05843964e+00, 3.33379686e+01,
 5.67048681e+02, 2.30571257e-02, 1.54843847e-01, 2.09167861e-01,
 6.52542583e-02, 6.30817958e-02, 1.78282760e-02]

    new_scaler.mean_ = mean
    new_scaler.scale_ = scale

    #Transforming
    data = np.array(data).reshape(1,-1)
    x_new_scaled = new_scaler.transform(data)

    return (np.array(x_new_scaled).reshape(1,-1))
