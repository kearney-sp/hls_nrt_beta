import pickle
import pandas as pd
import xarray as xr
from src.hls_funcs.bands import *
from src.hls_funcs.indices import *

func_dict = {
    "blue": blue_func,
    "ndvi": ndvi_func,
    "dfi": dfi_func,
    "ndti": ndti_func,
    "satvi": satvi_func,
    "ndii7": ndii7_func,
    "nir": nir_func,
    "swir1": swir1_func,
    "swir2": swir2_func,
    "bai_126": bai_126_func,
    "bai_136": bai_136_func,
    "bai_146": bai_146_func,
    "bai_236": bai_236_func,
    "bai_246": bai_246_func,
    "bai_346": bai_346_func
}


def predict_biomass(dat, model, se=True):
    """ Predict biomass (kg/ha) and standard error of prediction from existing linear model
        dat (xarray dataset) = new data in xarray Dataset format
        model (object) = opened existing model using pickle
        se (boolean) """

    model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]

    new_df = pd.DataFrame()
    for v in model_vars:
        new_df[v] = func_dict[v](dat).values.flatten()
    new_df['bm'] = np.exp(model.predict(new_df))

    if se:
        new_df.loc[~new_df.bm.isnull(), 'bm_se_log'] = model.get_prediction(new_df.loc[~new_df.bm.isnull()]).se_obs
        return [xr.DataArray(data=new_df['bm'].values.reshape(dat[list(dat.keys())[0]].shape),
                        coords=dat.coords),
                xr.DataArray(data=new_df['bm'].values.reshape(dat[list(dat.keys())[0]].shape),
                             coords=dat.coords)]
    else:
        return xr.DataArray(data=new_df['bm'].values.reshape(dat[list(dat.keys())[0]].shape),
                        coords=dat.coords)