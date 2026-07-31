
"""
Preprocessing utilities for IPIM-T.

This module contains the data preparation workflow:
- loading environmental features
- loading InSAR deformation features
- normalization
- saving training arrays

The complete KKH raster preprocessing can be connected here.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib



def normalize_features(
    train_data,
    save_path=None
):

    shape=train_data.shape

    flat=train_data.reshape(
        shape[0],
        -1
    )

    scaler=StandardScaler()

    flat=scaler.fit_transform(flat)

    output=flat.reshape(shape)


    if save_path:
        joblib.dump(
            scaler,
            save_path
        )


    return output, scaler



def save_numpy_dataset(
    env,
    insar,
    labels,
    output_dir
):

    np.save(
        output_dir+"/X_env.npy",
        env
    )

    np.save(
        output_dir+"/X_insar.npy",
        insar
    )

    np.save(
        output_dir+"/y.npy",
        labels
    )
