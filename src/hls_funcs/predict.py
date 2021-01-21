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


def pred_bm(dat, model, dim):
    model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]

    dat_masked = dat.where(dat.notnull)

    dims_list = [[dim] for v in model_vars]

    def pred_func(*args, mod_vars_np):
        vars_dict_np = {}
        for idx, v in enumerate(mod_vars_np):
            vars_dict_np[v] = args[idx]
        #print(vars_dict_np)
        bm_np = np.ones_like(args[0]) * np.nan
        mask = np.any(np.isnan(args), axis=0)
        bm_np[~mask] = np.exp(model.predict(vars_dict_np))
        #print(bm_np)
        return bm_np.astype('int16')

    def pred_func_xr(dat_xr, model_vars_xr, dims):
        vars_list_xr = []
        for v in model_vars_xr:
            vars_list_xr.append(func_dict[v](dat_xr))
        bm_xr = xr.apply_ufunc(pred_func,
                               *vars_list_xr,
                               kwargs=dict(mod_vars_np=np.array(model_vars_xr)),
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=dims,
                               output_core_dims=[dims[0]],
                               output_dtypes=['int16'])
        return bm_xr

    bm_out = pred_func_xr(dat_masked, model_vars, dims_list)

    return bm_out
