import xarray as xr
import pickle
import os
from copy import deepcopy
import re
import numpy as np
import scipy.stats as st
from tqdm import tqdm
from src.hls_funcs.masks import shp2mask
from src.hls_funcs.predict import predict_biomass
from src.hls_funcs.predict import pred_bm, pred_cov, pred_bm_se, pred_bm_thresh
from src.objects.charts import gauge_obj
import param
import panel as pn
pn.extension('echarts')
import datetime as dt
import geoviews as gv
from cartopy import crs
import affine
import holoviews as hv
from holoviews import streams
from bokeh.models.formatters import PrintfTickFormatter
import numpy as np
#import rioxarray
import hvplot.pandas
import hvplot.xarray
hv.extension('bokeh', logo=False)
import datetime
from pyproj import Proj
from distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=5, threads_per_worker=2, dashboard_address=":9898")
client = Client(cluster)


inDIR = "D:/LMF_STARFM_netcdf/"
all_files = [f for f in os.listdir(inDIR) if f.endswith('.nc')]
#dat = xr.open_dataset(os.path.join(inDIR, all_files[-1]))
yrs = np.arange(2013, 2021)
sub_files = [f for f in all_files if any(str(n) in f for n in yrs)]
da_list = []
for yr in yrs:
    f_path = os.path.join(inDIR, [f for f in all_files if str(yr) in f][0])
    dat_tmp = xr.open_dataset(f_path).chunk(dict(DOY=1, x=-1, y=-1))
    date_list = [datetime.datetime(yr, 1, 1) +
                 datetime.timedelta(days=d) for d in (dat_tmp['DOY'].values.astype('float') - 1)]
    dat_tmp = dat_tmp.rename(dict(DOY='date'))
    dat_tmp.coords['date'] = [d.date() for d in date_list]
    dat_tmp.coords['date'] = dat_tmp.coords['date'].astype('datetime64[ns]')
    da_list.append(dat_tmp)
dat = xr.concat(da_list, dim='date').sortby('date')

#dat = dat.chunk(dict(DOY=1, x=-1, y=-1))

bm_mod = pickle.load(open('C:/SPK_local/models/CPER_HLS_to_VOR_biomass_model_lr_simp.pk', 'rb'))
cov_mod = pickle.load(open('C:/SPK_local/models/CPER_HLS_to_LPI_cover_pls_binned_model.pk', 'rb'))

vars_list_xr = []
for idx, v in enumerate(band_list):
    vars_list_xr.append(func_dict[v](dat))
    vars_list_xr[idx].name = v

dat_xr = xr.merge(vars_list_xr)
dat_xr

da_cov = pred_cov(dat_xr, cov_mod)
from rasterio.plot import show
show(da_cov['SD'].sel(date='2020-10-01'))


ends_dict = {
    'SD': {
        'ndvi': 0.30,
        'dfi': 16,
        'bai_126': 155},
    'GREEN': {
        'ndvi': 0.55,
        'dfi': 10,
        'bai_126': 160},
    'BARE': {
        'ndvi': 0.10,
        'dfi': 8,
        'bai_126': 140}}



class App(param.Parameterized):
    date_init = datetime.datetime(int(dat['YEAR'].max()), 6, 15)
    thresh_init = 500
    bm_mean1 = 0

    da_bm = pred_bm(dat, bm_mod)
    da_bm.name = 'Biomass'
    da_bm = da_bm.where(da_bm > 0)
    da_bm_sel = da_bm.sel(date=date_init).persist()
    da_cov = pred_cov(dat, ends_dict)
    da_cov = da_cov.to_array(dim='type')
    da_cov = da_cov.where((da_cov < 1.0) | (da_cov.isnull()), 1.0)
    da_cov_sel = da_cov.sel(date=date_init).persist()
    da_se = pred_bm_se(dat, bm_mod)
    da_se = da_se.where(da_bm.notnull())
    da_thresh = pred_bm_thresh(da_bm, da_se, thresh_init)
    da_thresh.name = 'Threshold'
    da_thresh_sel = da_thresh.sel(date=date_init).persist()

    datCRS = crs.UTM(13)
    mapCRS = crs.GOOGLE_MERCATOR
    datProj = Proj(datCRS.proj4_init)
    mapProj = Proj(mapCRS.proj4_init)
    map_args = dict(crs=datCRS, rasterize=True, project=False, dynamic=True)
    map_opts = dict(projection=mapCRS, responsive=False, xticks=None, yticks=None, width=900, height=700,
                         padding=0, tools=['pan', 'wheel_zoom', 'box_zoom'],
                         active_tools=['pan', 'wheel_zoom'], toolbar='left')
    poly_opts = dict(fill_color=['', ''], fill_alpha=[0.0, 0.0], line_color=['#1b9e77', '#d95f02'],
                     line_width=[3, 3])
    gauge_opts = dict(height=200, width=300)

    bg_col='#ffffff'

    css = '''
    .bk.box1 {
      background: #ffffff;
      border-radius: 5px;
      border: 1px black solid;
    }
    '''
    pn.extension(raw_css=[css])

    basemap = param.ObjectSelector(default="Satellite", objects=["Satellite", "Map"])
    alpha = param.Number(default=1.0)
    date = param.CalendarDate(default=date_init)
    thresh = param.Integer(default=thresh_init)
    #action = param.Action(lambda self: self.param.trigger('action'), 'Compute')

    action = pn.widgets.Button(name='Save regions and \ncompute stats',
                               width=200)
    #action.on_click(lambda self: self.param.trigger('action'))

    action_val = action


    cov_map = da_cov_sel.hvplot.rgb(x='x', y='y', bands='type',
                                              **map_args).opts(**map_opts)
    bm_map = da_bm_sel.hvplot(x='x', y='y',
                                        cmap='Viridis', clim=(100, 1000), colorbar=False,
                                        **map_args).opts(**map_opts)
    thresh_map = da_thresh_sel.hvplot(x='x', y='y',
                                                cmap='YlOrRd', clim=(0.05, 0.95), colorbar=False,
                                                **map_args).opts(**map_opts)

    tiles = gv.tile_sources.EsriImagery.opts(projection=mapCRS, backend='bokeh')

    polys = hv.Polygons([])

    def __init__(self, **params):
        super(App, self).__init__(**params)

        self.gauge_obj = deepcopy(gauge_obj)

        self.poly_stream = streams.PolyDraw(source=self.polys, drag=True, num_objects=2,
                                       show_vertices=True, styles=self.poly_opts)
        self.edit_stream = streams.PolyEdit(source=self.polys, shared=True)
        self.select_stream = streams.Selection1D(source=self.polys)

        self.startX, self.endX = (float(self.da_bm['x'].min().values), float(self.da_bm['x'].max().values))
        self.startY, self.endY = (float(self.da_bm['y'].min().values), float(self.da_bm['y'].max().values))
        self.cov_stats = ''
        self.bm_stats = ''
        self.thresh_stats = ''
        #self.stats_dict = {0: self.cov_stats,
        #         1: self.bm_stats,
        #         2: self.thresh_stats}
        #self.stats = self.stats_dict[self.active_tab]
        #self.text2.jscallback(args={'gauge1': self.gauge_pane1}, code="""
        #gauge1.data.series[0].data[0].value = cb_obj.value
        #gauge1.properties.data.change.emit()
        #""")
        self.all_maps = pn.Tabs(('Cover', pn.Row(self.tiles * self.cov_map * self.polys, self.cov_stats)),
                           ('Biomass', pn.Row(self.tiles * self.bm_map * self.polys, self.bm_stats)),
                           ('Threshold', pn.Row(self.tiles * self.thresh_map * self.polys, self.thresh_stats)))
        self.active_tab = self.all_maps.active

    def keep_zoom(self, x_range, y_range):
        map_x_range, map_y_range = self.mapProj(x_range, y_range, inverse=True)
        (self.startX, self.endX), (self.startY, self.endY) = self.datProj(map_x_range, map_y_range, inverse=False)

    @param.depends('basemap')
    def map_base(self):
        if self.basemap == "Satellite":
            self.tiles = gv.tile_sources.EsriImagery(projection=self.mapCRS, backend='bokeh')
        elif self.basemap == "Map":
            self.tiles = gv.tile_sources.Wikipedia(projection=self.mapCRS, backend='bokeh')
        return self.tiles

    @param.depends('date', watch=True)
    def bm_date(self):
        self.date_init = np.datetime64(self.date)
        self.da_bm_sel = self.da_bm.sel(date=self.date_init).persist()
        self.bm_map = self.da_bm_sel.hvplot(x='x', y='y',
                                                               xlim=(self.startX, self.endX),
                                                               ylim=(self.startY, self.endY),
                                                               cmap='Viridis', clim=(100, 1000),
                                                               colorbar=False,
                                                               **self.map_args).opts(alpha=self.alpha,
                                                                                     **self.map_opts)
        self.bm_map.streams[-1].add_subscriber(self.keep_zoom)
        return self.bm_map

    @param.depends('date', watch=True)
    def cov_date(self):
        self.date_init = np.datetime64(self.date)
        self.da_cov_sel = self.da_cov.sel(date=self.date_init).persist()
        self.cov_map = self.da_cov_sel.hvplot.rgb(x='x', y='y',
                                                    xlim=(self.startX, self.endX),
                                                    ylim=(self.startY, self.endY),
                                                    bands='type',
                                                    **self.map_args).opts(alpha=self.alpha,
                                                                          **self.map_opts)
        return self.cov_map

    @param.depends('date', 'thresh', watch=True)
    def thresh_date(self):
        self.date_init = np.datetime64(self.date)
        self.thresh_init = self.thresh
        self.da_thresh = pred_bm_thresh(self.da_bm, self.da_se, self.thresh_init)
        self.da_thresh_sel = self.da_thresh.sel(date=self.date_init).persist()
        self.thresh_map = self.da_thresh_sel.hvplot(x='x', y='y',
                                           xlim=(self.startX, self.endX),
                                           ylim=(self.startY, self.endY),
                                           cmap='YlOrRd', clim=(0.05, 0.95), colorbar=False,
                                           **self.map_args).opts(alpha=self.alpha,
                                                                 **self.map_opts)
        self.bm_map.streams[-1].add_subscriber(self.keep_zoom)
        return self.thresh_map

    @param.depends('alpha', watch=True)
    def cov_alpha(self):
        return self.cov_map.opts(alpha=self.alpha,
                                 xlim=self.bm_map.streams[-1].x_range,
                                 ylim=self.bm_map.streams[-1].y_range,
                               **self.map_opts)
    @param.depends('alpha', watch=True)
    def bm_alpha(self):
        self.bm_map = self.bm_map.opts(alpha=self.alpha,
                                 xlim=self.bm_map.streams[-1].x_range,
                                 ylim=self.bm_map.streams[-1].y_range,
                               **self.map_opts)
        self.bm_map.streams[-1].add_subscriber(self.keep_zoom)
        return self.bm_map

    @param.depends('alpha', watch=True)
    def thresh_alpha(self):
        return self.thresh_map.opts(alpha=self.alpha,
                                 xlim=self.bm_map.streams[-1].x_range,
                                 ylim=self.bm_map.streams[-1].y_range,
                               **self.map_opts)

    @param.depends('action.param.value', watch=True)
    def show_hist(self):
        if self.poly_stream.data is None:
            self.cov_stats = ''
            self.bm_stats = ''
            self.thresh_stats = ''
        else:
            self.bm_stats = 'Yes'
            thresh_list = []
            bm_list = []
            cov_list = []
            ts_yr_list = []
            ts_avg_list = []
            for idx, ps_c in enumerate(self.poly_stream.data['line_color']):
                xs_map, ys_map = self.mapProj(self.poly_stream.data['xs'][idx],
                                              self.poly_stream.data['ys'][idx], inverse=True)
                xs_dat, ys_dat = self.datProj(xs_map, ys_map, inverse=False)
                geometries = {
                    "type": "Polygon",
                    "coordinates": [
                        list(map(list, zip(xs_dat, ys_dat)))
                    ]
                }
                ta = affine.Affine(30.0, 0.0, float(self.da_bm_sel['x'].min()),
                                   0.0, -30.0, float(self.da_bm_sel['y'].max()))
                poly_mask = shp2mask([geometries], self.da_bm_sel,
                                     transform=ta, outshape=self.da_bm_sel.shape, default_value=1)
                da_bm_tmp = self.da_bm_sel.where(poly_mask == 1)
                bm_hist_tmp = da_bm_tmp.hvplot.hist('Biomass', xlim=(0, 2000),
                                                    bins=np.arange(0, 10000, 20))\
                    .opts(height=200, width=300, fill_color=ps_c, fill_alpha=0.6,
                          line_color='black', line_width=0.5, line_alpha=0.6,
                          bgcolor=self.bg_col).options(toolbar=None)
                markdown = pn.pane.Markdown('## Region stats', height=50,
                                            style={'font-family': "serif",
                                                   'color': ps_c})
                thresh_pct = round(float(da_bm_tmp.where(da_bm_tmp < self.thresh_init).count())/
                                   float(da_bm_tmp.count()) * 100, 0)
                thresh_text = pn.pane.Markdown(f'**{thresh_pct}%** of the region is estimated to have biomass ' +
                                               f'less than {self.thresh_init} kg/ha.',
                                               style={'font-family': "Helvetica"})
                thresh_list.append(pn.Column(pn.Row(pn.layout.HSpacer(), markdown, pn.layout.HSpacer()),
                                             bm_hist_tmp * hv.VLine(x=self.thresh_init).opts(line_color='black'),
                                             thresh_text,
                                             css_classes=['box1'], margin=5))
                bm_gauge_obj = deepcopy(self.gauge_obj)
                bm_gauge_obj['series'][0]['data'][0]['value'] = int(da_bm_tmp.mean().values)
                bm_gauge_pane = pn.pane.ECharts(bm_gauge_obj, **self.gauge_opts)
                bm_list.append(pn.Column(pn.Row(pn.layout.HSpacer(),markdown, pn.layout.HSpacer()),
                                         bm_gauge_pane,
                                         css_classes=['box1'], margin=5))
                #yr = int(self.da_bm_sel.YEAR.values)
                #ts_bm_yr_tmp = self.da_bm.where(poly_mask == 1).sel(date=slice(datetime.datetime(yr, 5, 1),
                #                                                                datetime.datetime(yr, 10, 31)))
                #ts_yr_list.append(ts_bm_yr_tmp.mean(dim=['x', 'y']).hvplot.line(x='date',
                #                                                               y='Biomass').opts(line_color=ps_c))
                #ts_bm_avg_tmp = self.da_bm.where(poly_mask == 1).groupby(da_bm.date.dt.dayofyear).mean(dim=['x', 'y'])
                #ts_avg_list.append(ts_bm_avg_tmp.hvplot.line(x='date', y='Biomass').opts(line_color=ps_c))
                da_cov_tmp = self.da_cov_sel.where(poly_mask == 1)
                cov_factors = list(da_cov_tmp.type.values)
                cov_vals = [round(float(da_cov_tmp.sel(type=f).mean().values), 2) for f in cov_factors]
                from bokeh.models import NumeralTickFormatter
                pct_fmt = NumeralTickFormatter(format="0%")
                cov_colors = hv.Cycle(['red', 'green', 'blue'])
                cov_scatter_tmp = hv.Overlay([hv.Scatter(f) for f in list(zip(cov_factors, cov_vals))]) \
                    .options({'Scatter': dict(xformatter=pct_fmt,
                                              size=15,
                                              fill_color=cov_colors,
                                              line_color=cov_colors,
                                              ylim=(0, 1))})
                cov_spike_tmp = hv.Overlay([hv.Spikes(f) for f in cov_scatter_tmp])\
                    .options({'Spikes': dict(color=cov_colors, line_width=4,
                                             labelled=[], invert_axes=True, color_index=None,
                                             ylim=(0, 1))})
                cov_list.append(pn.Column(pn.Row(pn.layout.HSpacer(), markdown, pn.layout.HSpacer()),
                                          (cov_spike_tmp * cov_scatter_tmp).options(height=200,
                                                                                    width=300,
                                                                                    bgcolor=self.bg_col,
                                                                                    toolbar=None),
                                          css_classes=['box1'], margin=5))


            self.polys=self.poly_stream.element.opts(xlim=(self.startX, self.endX),
                                                     ylim=(self.startY, self.endY))
            self.poly_stream = streams.PolyDraw(source=self.polys, drag=True, num_objects=2,
                                           show_vertices=True, styles=self.poly_opts)
            self.edit_stream = streams.PolyEdit(source=self.polys, shared=True)
            #self.gauge_pane1.object['series'][0]['data'][0]['value'] = int(da_bm_tmp.mean().values)
            #self.gauge_pane2.object['series'][0]['data'][0]['value'] = int(da_bm_tmp.mean().values) + 100
            self.thresh_stats = pn.Column(*thresh_list)
            self.bm_stats = pn.Column(*bm_list)
            self.cov_stats = pn.Column(*cov_list)


    def view_all(self):
        #self.da_bm_sel.name = 'Biomass'
        self.active_tab = self.all_maps.active
        self.bm_map.streams[-1].add_subscriber(self.keep_zoom)
        #self.stats = self.stats_dict[self.active_tab]
        base = hv.DynamicMap(self.map_base)
        cov = self.cov_map
        bm = self.bm_map
        thresh = self.thresh_map
        self.all_maps = pn.Tabs(('Cover', pn.Row(base * cov * self.polys, self.cov_stats)),
                                ('Biomass', pn.Row(base * bm * self.polys, self.bm_stats)),
                                ('Threshold', pn.Row(base * thresh * self.polys, self.thresh_stats)),
                                active=self.active_tab)
        return pn.Column(self.all_maps)



viewer = App()
pn.serve(pn.Row(pn.Column(pn.Param(viewer.param,
                                      widgets={'date': pn.widgets.DatePicker(name='Calendar',
                                                                             value=datetime.datetime(int(dat['YEAR'].max()), 6, 15).date(),
                                                                             width=200),
                                               'alpha': pn.widgets.FloatSlider(name='Map transparency',
                                                                               value=1.0,
                                                                               start=0.0, end=1.0,
                                                                               step=0.1,
                                                                               width=200),
                                               'thresh': pn.widgets.IntSlider(name='Threshold',
                                                                              start=200, end=2000,
                                                                              step=25, value=500,
                                                                              format=PrintfTickFormatter(
                                                                                  format='%d kg/ha'),
                                                                              width=200),
                                               'basemap': pn.widgets.Select(name="Change basemap",
                                                                            options=["Satellite", "Map"],
                                                                            width=200),
                                               #'action': pn.widgets.Button(name='Save regions and \ncompute stats',
                                               #                            width=200)
                                               }), viewer.action_button),
                viewer.view_all))



layout = pn.template.ReactTemplate(
    site="Shortgrass Explorer",
    title="HoloViews",
    theme=pn.template.react.DarkTheme,
    row_height=100,
)

layout.sidebar[:] = [pn.Column(pn.Param(viewer.param,
                                      widgets={'date': pn.widgets.DatePicker(name='Calendar',
                                                                             value=datetime.datetime(int(dat['YEAR'].max()), 6, 15).date()),
                                               'alpha': pn.widgets.FloatSlider(name='Map transparency',
                                                                               value=1.0,
                                                                               start=0.0, end=1.0,
                                                                               step=0.1),
                                               'thresh': pn.widgets.IntSlider(name='Threshold',
                                                                              start=200, end=2000,
                                                                              step=25, value=500,
                                                                              format=PrintfTickFormatter(
                                                                                  format='%d kg/ha')),
                                               'basemap': pn.widgets.Select(name="Change basemap",
                                                                            options=["Satellite", "Map"]),
                                               'action': pn.widgets.Button(name='Save regions and \ncompute stats')}))]
layout.main[:, :] = viewer.view_all
pn.serve(layout)



import param
import panel as pn

class A(param.Parameterized):
    action = param.Event(label="param mapped")

class B(param.Parameterized):

    def __init__(self, a, **params):
        super().__init__(**params)
        a.param.watch(self.compute, 'action')

    def compute(self, event):
        print(event)

a = A()
b = B(a)

pn.serve(pn.Column(a.param), show=True)


factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
x =  [50, 40, 65, 10, 25, 37, 80, 60]
scatter = hv.Scatter((factors, x)).opts(size=15, fill_color="orange", line_color="green")
spikes = hv.Spikes(scatter).opts(color='green', line_width=4, labelled=[], invert_axes=True, color_index=None)
plot = (spikes * scatter).options(toolbar=None, bgcolor='#f0f0f0')

css = '''
.bk.box1 {
  background: #f0f0f0;
  border-radius: 5px;
  border: 1px black solid;
  margin-top: 10% 5% 10%;
  margin-bottom:10%
}
'''
markdown = pn.pane.Markdown('## Area info',
                                style={'font-family': "serif",
                                       'text-align': 'center',
                                       'color': '#1b9e77'})
pn.extension(raw_css=[css])
layout = pn.template.ReactTemplate(
    site="Awesome Panel",
    title="HoloViews",
    theme=pn.template.react.DarkTheme,
    row_height=200,
)
layout.main[0:2, 0:4] = pn.Column(pn.Row(pn.layout.HSpacer(), markdown, pn.layout.HSpacer()),
                                  plot, css_classes=['box1'], margin=5, sizing_mode='stretch_both')
layout.main[3:4, 0:3] =  pn.Column(plot, css_classes=['box1'], margin=5,sizing_mode='stretch_both')
pn.serve(layout)









da_bm = pred_bm(dat, bm_mod)
da_bm = da_bm.where(da_bm > 0)
da_bm.name = 'Biomass'
da_cov = pred_cov(dat, ends_dict)
da_cov = da_cov.to_array(dim='type')
da_cov = da_cov.where((da_cov < 1.0) | (da_cov.isnull()), 1.0)
da_se = pred_bm_se(dat, bm_mod)
da_se = da_se.where(da_bm.notnull())

hv.config.image_rtol = 0.01
datCRS = crs.UTM(13)
mapCRS = crs.GOOGLE_MERCATOR
datProj = Proj(datCRS.proj4_init)
mapProj = Proj(mapCRS.proj4_init)
#map_args = dict(crs=datCRS, rasterize=True, project=True, dynamic=True)
map_args = dict(rasterize=True, dynamic=True)
map_opts = dict(projection=mapCRS, responsive=False, width=900, height=500,
                padding=0, tools=['pan', 'wheel_zoom', 'box_zoom', 'poly_edit'],
                    active_tools=['pan', 'wheel_zoom'])
x, y = datProj(da_bm.x.values, da_bm.y.values, inverse=True)
bm_sub = da_bm.isel(date=150)
bm_sub['x'] = x
bm_sub['y'] = y
bm_map = bm_sub.hvplot(x='x', y='y', geo=True,
                       **map_args).opts(**map_opts)
img_stream = streams.RangeXY(source=bm_map)
tiles = gv.tile_sources.EsriImagery.opts(projection=mapCRS, backend='bokeh')
pn.serve(pn.Pane(tiles * bm_map))

class Test(param.Parameterized):
    bm_sub = da_bm.isel(date=150)
    bm_map = bm_sub.hvplot(x='x', y='y',
                                         **map_args).opts(**map_opts)
    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=4,
                                   show_vertices=True, styles={
            'fill_color': ['red', 'green', 'blue'],
            'fill_alpha': [0.6, 0.6, 0.6]
        })
    poly_edit = streams.PolyEdit(source=poly, shared=True)
    select_stream = streams.Selection1D(source=poly)
    point_holder = hv.Points([10])
    point_holder_hv = pn.pane.HoloViews(point_holder)
    #slider = pn.widgets.IntSlider(start=0, end=1000)
    gauge1 = gauge_obj
    gauge_pane = pn.pane.ECharts(gauge1, width=400, height=400)
    #slider.jscallback(args={'gauge': gauge_pane}, value="""
    #    gauge.data.series[0].data[0].value = cb_obj.value
    #    gauge.properties.data.change.emit()
    #    """)
    point_holder_hv.jscallback(args={'gauge': gauge_pane}, value="""
        gauge.data.series[0].data[0].value = y_range.start
        gauge.properties.data.change.emit()
        """)

    def __init__(self, **params):
        super(Test, self).__init__(**params)

    def watch_poly(self, index):
        if len(index) > 0:
            data = self.poly_stream.data
            xs_map, ys_map = mapProj([data['xs'][i] for i in index][0],
                                          [data['ys'][i] for i in index][0], inverse=True)
            xs_dat, ys_dat = datProj(xs_map, ys_map, inverse=False)
            geometries = {
                "type": "Polygon",
                "coordinates": [
                    list(map(list, zip(xs_dat, ys_dat)))
                ]
            }
            ta = affine.Affine(30.0, 0.0, float(self.bm_sub['x'].min()),
                               0.0, -30.0, float(self.bm_sub['y'].max()))
            poly_mask = shp2mask([geometries], self.bm_sub,
                                 transform=ta, outshape=self.bm_sub.shape, default_value=1)
            da_bm_tmp = self.bm_sub.where(poly_mask == 1)
            #self.slider.value = int(da_bm_tmp.mean().values)
            self.point_holder = hv.Points([int(da_bm_tmp.mean().values)])

    #@param.depends('slider')
    def show(self):
        self.select_stream.add_subscriber(self.watch_poly)
        #gauge_pane = pn.pane.ECharts(self.gauge1, width=400, height=400)
        return pn.Row(pn.Pane(self.bm_map*self.poly), pn.Column(self.gauge_pane))

Tester = Test()
pn.serve(Tester.show)
pn.serve(pn.Row(pn.Pane(bm_map*poly_out), pn.Column(gauge_pane, slider)))

gauge1 = gauge_obj
gauge_pane = pn.pane.ECharts(gauge1, width=200, height=250)
pn.serve(gauge_pane)

x=1
test = pn.widgets.Button(name='Click')
def button_f(event):
    print(event)
    x+=1
test.on_click(button_f)

class Tester2(param.Parameterized):
    def __init__(self):
        super(Tester2, self)
        #self.action = param.Action(lambda x: x.param.trigger('action'), label='Compute')
        self.button = pn.widgets.Button(name='Click')
        self.text = pn.widgets.TextInput(value='Ready')
        self.a=5
        self.button.on_click(self.b)

    def b(self, event):
        super(Tester2, self)
        self.text.value = 'Clicked {0} times'.format(self.button.clicks)
        self.a = self.a + self.button.clicks
        print(self.a)

test = Tester2()
pn.serve(pn.Row(test.button, test.text))

pn.serve(test)

gauge = {
    'tooltip': {
        'formatter': '{a} <br/>{b} : {c}%'
    },
    'series': [
        {
            'name': 'Gauge',
            'type': 'gauge',
            'min': 0,
            'max': 1000,
            'detail': {'formatter': '{value}%'},
            'data': [{'value': 50, 'name': 'Value'}]
        }
    ]
}
slider = pn.widgets.IntSlider(start=0, end=1000)
gauge_pane = pn.pane.ECharts(gauge, width=400, height=400)
slider.jscallback(args={'gauge': gauge_pane}, value="""
    gauge.data.series[0].data[0].value = cb_obj.value
    gauge.properties.data.change.emit()
    """)
pn.serve(pn.Column(gauge_pane, slider))

def load_bm(date):
    bm_map = da_bm.sel(DOY=date).hvplot(x='x', y='y', tiles='EsriImagery', crs=crs.UTM(13),
                                         cmap='Viridis', clim=(200, 1800), colorbar=False,
                                        data_aspect=0.6).opts(responsive=True,
                                                              xticks=None,
                                                              yticks=None)
    print(bm_map.range(dimension='x'))
    return bm_map

def load_cov(date):
    cv_map = da_cov.sel(DOY=date).hvplot.rgb(x='x', y='y',
                                              bands='type', tiles='EsriImagery',
                                             crs=crs.UTM(13),
                                             data_aspect=0.6).opts(responsive=True,
                                                              xticks=None,
                                                              yticks=None)
    return cv_map

def load_thresh(date, thresh):
    da_thresh = pred_bm_thresh(da_bm, da_se, thresh)
    thresh_map = da_thresh.sel(DOY=date).hvplot(x='x', y='y', tiles='EsriImagery', crs=crs.UTM(13),
                                         cmap='YlOrRd', clim=(0.05, 0.95), colorbar=False,
                                        data_aspect=0.6).opts(responsive=True,
                                                              xticks=None,
                                                              yticks=None)
    return thresh_map


date_picker = pn.widgets.DatePicker(name='Calendar',
                                    value=datetime.datetime(2016, 6, 15).date())
thresh_picker = pn.widgets.IntSlider(name='Threshold', start=200, end=2000, step=25, value=500,
                                     format=PrintfTickFormatter(format='%d kg/ha'))

@pn.depends(date=date_picker.param.value)
def load_map_bm(date):
    p = load_bm(date.timetuple().tm_yday)
    return p
@pn.depends(date=date_picker.param.value)
def load_map_cov(date):
    return load_cov(date.timetuple().tm_yday)
@pn.depends(date=date_picker.param.value, thresh=thresh_picker.param.value)
def load_map_thresh(date, thresh):
    return load_thresh(date.timetuple().tm_yday, thresh)

pn.serve(pn.Row(pn.WidgetBox('# Interactive tools:',
                             date_picker, thresh_picker, width=250),
                pn.Tabs(('Cover', load_map_cov), ('Biomass', load_map_bm), ('Threshold', load_map_thresh)),
                        sizing_mode='stretch_both'))




geometries = \
    {'type': 'Polygon', 'coordinates': [
[[515775.58844153915, 4523045.2408434935],
 [518012.1479929752, 4522208.065559676],
 [515722.8805455199, 4519434.631583949],
 [513485.08678470645, 4520633.901147823]]
    ]}
da_bm_sel = da_bm.isel(date=150)
ta = affine.Affine(30.0, 0.0, float(da_bm_sel['x'].min()),
                   0.0, -30.0, float(da_bm_sel['y'].max()))
poly_mask = shp2mask([geometries], da_bm_sel,
                     transform=ta, outshape=da_bm_sel.shape, default_value=1)
show(poly_mask)
da_bm_sel.where(poly_mask == 1)

class App(param.Parameterized):
    date_init = datetime.datetime(dat['YEAR'].values, 6, 15).date()
    doy_init = date_init.timetuple().tm_yday
    thresh_init = 500

    da_bm = pred_bm(dat, bm_mod)
    da_bm = da_bm.where(da_bm > 0)
    da_cov = pred_cov(dat, ends_dict)
    da_cov = da_cov.to_array(dim='type')
    da_cov = da_cov.where((da_cov < 1.0) | (da_cov.isnull()), 1.0)
    da_se = pred_bm_se(dat, bm_mod)
    da_se = da_se.where(da_bm.notnull())

    da_thresh = pred_bm_thresh(da_bm, da_se, thresh_init)

    bm_map = da_bm.sel(DOY=doy_init).hvplot(x='x', y='y',   rasterize=True,
                                         cmap='Viridis', clim=(100, 1000), colorbar=False).opts(responsive=False,
                                                              xticks=None,
                                                              yticks=None)
    cv_map = da_cov.sel(DOY=doy_init).hvplot.rgb(x='x', y='y',
                                              bands='type', rasterize=True).opts(responsive=False,
                                                              xticks=None,
                                                              yticks=None)
    thresh_map = da_thresh.sel(DOY=doy_init).hvplot(x='x', y='y',  rasterize=True,
                                         cmap='YlOrRd', clim=(0.05, 0.95), colorbar=False).opts(responsive=False,
                                                              xticks=None,
                                                              yticks=None)


    def keep_zoom(self, x_range, y_range):
        self.startX, self.endX = x_range
        self.startY, self.endY = y_range

    #rangexy = streams.RangeXY(source = bm_map, x_range=(startX,endX), y_range=(startY,endY))

    #date_picker = pn.widgets.DatePicker(name='Calendar',
    #                                    value=datetime.datetime(da_bm['YEAR'].values, 6, 15).date())
    #thresh_picker = pn.widgets.IntSlider(name='Threshold', start=200, end=2000, step=25, value=500,
    #                                     format=PrintfTickFormatter(format='%d kg/ha'))
    date = param.CalendarDate(default=date_init)
    thresh = param.Integer(default=thresh_init)
    # create a button that when pushed triggers 'button'
    button = param.Action(lambda x: x.param.trigger('button'), label='Get XY')

    def __init__(self, **params):
        super(App, self).__init__(**params)
        self.startX, self.endX = (508305.0, 538305.0)
        self.startY, self.endY = (4504575.0, 4534575.0)
        self.xr = (508305.0, 538305.0)
        self.yr = (4504575.0, 4534575.0)
        self.t = datetime.datetime.now()

    @param.depends('button')
    def view_xy(self):
        x = self.bm_map.streams[-1].x_range
        y = self.bm_map.streams[-1].y_range
        return str(x) + ',' + str(y)

    @param.depends('button')
    def view_xy2(self):
        self.t = datetime.datetime.now()
        self.bm_map.streams[-1].add_subscriber(self.keep_zoom)
        x = (self.startX, self.endX)
        y = (self.startY, self.endY)
        return str(x) + ',' + str(y)

    @param.depends('button')
    def view_t(self):
        return self.t

    @param.depends('date')
    def view_bm(self):
        self.doy_init = self.date.timetuple().tm_yday
        self.bm_map = self.da_bm.sel(DOY=self.doy_init).hvplot(x='x', y='y',
                                                               #xlim=(self.startX, self.endX),
                                                               #ylim=(self.startY, self.endY),
                                                                       cmap='Viridis', clim=(100, 1000),
                                                                       colorbar=False,
                                                               rasterize=True).opts(responsive=False,
                                                                                             xticks=None,
                                                                                             yticks=None)
        self.bm_map.streams[-1].add_subscriber(self.keep_zoom)
        self.bm_map = self.bm_map.redim.range(x=(self.startX, self.endX), y=(self.startY, self.endY))
        return self.bm_map

    @param.depends('date')
    def view_cov(self):
        self.doy_init = self.date.timetuple().tm_yday
        self.cv_map = self.da_cov.sel(DOY=self.doy_init).hvplot.rgb(x='x', y='y',
                                                                    #xlim=(self.startX, self.endX),
                                                                    #ylim=(self.startY, self.endY),
                                                     bands='type', rasterize=True).opts(responsive=False,
                                                                                             xticks=None,
                                                                                             yticks=None)
        self.cv_map.streams[-1].add_subscriber(self.keep_zoom)
        self.cv_map = self.cv_map.redim.range(x=(self.startX, self.endX), y=(self.startY, self.endY))
        return self.cv_map
    @param.depends('date', 'thresh')
    def view_thresh(self):
        self.doy_init = self.date.timetuple().tm_yday
        self.thresh_init = self.thresh
        self.da_thresh = pred_bm_thresh(self.da_bm, self.da_se, self.thresh_init)
        self.thresh_map = self.da_thresh \
            .sel(DOY=self.doy_init).hvplot(x='x', y='y',
                                           #xlim=(self.startX, self.endX),
                                           #ylim=(self.startY, self.endY),
                                      cmap='YlOrRd', clim=(0.05, 0.95),
                                      colorbar=False, rasterize=True).opts(responsive=False,
                                                            xticks=None,
                                                            yticks=None)
        self.thresh_map.streams[-1].add_subscriber(self.keep_zoom)
        self.thresh_map = self.thresh_map.redim.range(x=(self.startX, self.endX), y=(self.startY, self.endY))
        return self.thresh_map

viewer = App()
pn.serve(pn.Row(pn.Column(pn.Param(viewer.param,
                                      widgets={'date': pn.widgets.DatePicker(name='Calendar',
                                        value=datetime.datetime(dat['YEAR'].values, 6, 15).date()),
                                               'thresh': pn.widgets.IntSlider(name='Threshold',
                                                                              start=200, end=2000,
                                                                              step=25, value=500,
                                                                              format=PrintfTickFormatter(
                                                                                  format='%d kg/ha'))}),
                          viewer.view_xy,
                          viewer.view_xy2,
                          viewer.view_t),
                pn.Tabs(('Biomass', viewer.view_bm),
                        ('Cover', viewer.view_cov),
                        ('Threshold', viewer.view_thresh))))


test = da_bm.sel(DOY=100).hvplot(x='x', y='y',  rasterize=True,
                                 #xlim=xr,
                                 #ylim=yr,
                                         cmap='Viridis', clim=(200, 1800), colorbar=False).opts(responsive=False,
                                                              xticks=None,
                                                              yticks=None)
pn.Row(test).servable()

test = da_bm.sel(DOY=100).hvplot(x='x', y='y', tiles='EsriImagery', crs=crs.UTM(13), rasterize=True,
                                    cmap='Viridis', clim=(200, 1800), colorbar=False,
                                    data_aspect=0.6).opts(responsive=True,
                                                          xticks=None,
                                                          yticks=None)
test


import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas
from holoviews import streams
import param

class AppTest(param.Parameterized):
    data = 10*np.random.rand(100,4)
    df = pd.DataFrame(data=data,columns=['x','y','z1' ,'z2'])
    color_val = param.Selector(default='z1',objects=['z1','z2'])
    plot = df.hvplot.scatter(x='x',y='y',c='z1')
    startX,endX = plot.range('x')
    startY,endY = plot.range('y')

    def keep_zoom(self,x_range,y_range):
        self.startX,self.endX = x_range
        self.startY,self.endY = y_range

    @param.depends('color_val')
    def view(self):
        self.plot = self.df.hvplot.scatter(x='x',y='y',c=self.color_val)
        self.plot = self.plot.redim.range(x=(self.startX,self.endX), y=(self.startY,self.endY))
        rangexy = streams.RangeXY(source = self.plot, x_range=(self.startX,self.endX), y_range=(self.startY,self.endY))
        rangexy.add_subscriber(self.keep_zoom)
        return self.plot

viewer = AppTest()
pn.serve(pn.Column(pn.Param(viewer.param,widgets={'color_val':pn.widgets.RadioButtonGroup}),viewer.view))


import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

hv.extension('bokeh')
renderer = hv.renderer('bokeh')

x_range_fmt = "({:.3f}, {:.3f})"

# Create dummy dataset
num_pts = 1000

# Create scatterplots
hv_plots = [da_bm.sel(DOY=100).hvplot().opts(axiswise=True) for i in range(2)]

txt_xrange_doc = [pn.widgets.StaticText(name="x_range using get_root", value='x_range: (,)') for i in range(2)]
txt_xrange_renderer = [pn.widgets.StaticText(name="x_range using get_plot", value='x_range: (,)') for i in range(2)]
txt_xrange_stream = [pn.widgets.StaticText(name="x_range using streams", value='x_range: (,)') for i in range(2)]

# Create streams linked to x_range
rng = [hv.streams.RangeX(source=hv_plot) for hv_plot in hv_plots]

def query_x_range(event):
    for i in range(2):
        # Query x_range by using get_root on panel to obtain bokeh document
        xrng_doc = panel.get_root().children[i].children[0].x_range
        xrng_str = x_range_fmt.format(xrng_doc.start, xrng_doc.end)
        txt_xrange_doc[i].value = xrng_str

        # Query x_range by using holoviews renderer to get plot handles
        xrng_renderer = renderer.get_plot(panel[i][0].object).handles['x_range']
        xrng_str = x_range_fmt.format(xrng_renderer.start, xrng_renderer.end)
        txt_xrange_renderer[i].value = xrng_str

        # Query x_range stream object
        xrng_stream = rng[i].x_range
        xrng_str = x_range_fmt.format(xrng_stream[0], xrng_stream[1])
        txt_xrange_stream[i].value = xrng_str


update_button = pn.widgets.Button(name='Query figure x_range', button_type='primary')
update_button.on_click(query_x_range)

# Create panel
panel = pn.Row(pn.Column(hv_plots[0], txt_xrange_doc[0], txt_xrange_renderer[0], txt_xrange_stream[0]),
               pn.Column(hv_plots[1], txt_xrange_doc[1], txt_xrange_renderer[1], txt_xrange_stream[1], update_button))
pn.serve(panel)















def load_bm(date):
    bm_map = da_bm.sel(DOY=date).hvplot(x='x', y='y', tiles='EsriImagery', crs=ccrs.UTM(13),
                                         cmap='Viridis', clim=(100, 1000), colorbar=False,
                                        data_aspect=0.6).opts(responsive=True,
                                                              xticks=None,
                                                              yticks=None)
    print(bm_map.range(dimension='x'))
    return bm_map

def load_cov(date):
    cv_map = da_cov.sel(DOY=date).hvplot.rgb(x='x', y='y',
                                              bands='type', tiles='EsriImagery',
                                             crs=ccrs.UTM(13), data_aspect=0.6).opts(responsive=True,
                                                              xticks=None,
                                                              yticks=None)
    return cv_map

def load_thresh(date, thresh):
    da_thresh = pred_bm_thresh(dat, bm_mod, thresh)
    thresh_map = da_thresh.sel(DOY=date).hvplot(x='x', y='y', tiles='EsriImagery', crs=ccrs.UTM(13),
                                         cmap='YlOrRd', clim=(0.05, 0.95), colorbar=False,
                                        data_aspect=0.6).opts(responsive=True,
                                                              xticks=None,
                                                              yticks=None)
    return thresh_map


def keep_zoom(self,x_range,y_range):
    self.startX,self.endX = x_range
    self.startY,self.endY = y_range

# date_slider = pn.widgets.IntSlider(name='Date Slider',
#                                    start=0, end=len(da_bm.time), value=0)

date_picker = pn.widgets.DatePicker(name='Calendar',
                                    value=datetime.datetime(da_bm['YEAR'].values, 6, 15).date())
thresh_picker = pn.widgets.IntSlider(name='Threshold', start=200, end=2000, step=25, value=500,
                                     format=PrintfTickFormatter(format='%d kg/ha'))

@pn.depends(date=date_picker.param.value)
def load_map_bm(self):
    p = load_bm(self.date.timetuple().tm_yday)
    startX, endX = p.range('x')
    startY, endY = p.range('y')
    self.plot = self.plot.redim.range(x=(self.startX, self.endX), y=(self.startY, self.endY))
    rangexy = streams.RangeXY(source=self.plot, x_range=(self.startX, self.endX), y_range=(self.startY, self.endY))
    rangexy.add_subscriber(self.keep_zoom)
    return p
@pn.depends(date=date_picker.param.value)
def load_map_cov(date):
    return load_cov(date.timetuple().tm_yday)
@pn.depends(date=date_picker.param.value, thresh=thresh_picker.param.value)
def load_map_thresh(date, thresh):
    return load_thresh(date.timetuple().tm_yday, thresh)

def query_x_range(event):
    for i in range(2):
        # Query x_range by using get_root on panel to obtain bokeh document
        xrng_doc = panel.get_root().children[i].children[0].x_range
        xrng_str = x_range_fmt.format(xrng_doc.start, xrng_doc.end)
        txt_xrange_doc[i].value = xrng_str

        # Query x_range by using holoviews renderer to get plot handles
        xrng_renderer = renderer.get_plot(panel[i][0].object).handles['x_range']
        xrng_str = x_range_fmt.format(xrng_renderer.start, xrng_renderer.end)
        txt_xrange_renderer[i].value = xrng_str

        # Query x_range stream object
        xrng_stream = rng[i].x_range
        xrng_str = x_range_fmt.format(xrng_stream[0], xrng_stream[1])
        txt_xrange_stream[i].value = xrng_str


update_button = pn.widgets.Button(name='Query figure x_range', button_type='primary')
update_button.on_click(query_x_range)

pn.serve(pn.Row(pn.WidgetBox('# Interactive tools:',
                             date_picker, thresh_picker, width=250),
                pn.Tabs(('Biomass', load_map_bm), ('Cover', load_map_cov), ('Threshold', load_map_thresh)),
                        sizing_mode='stretch_both'))



dat_stacked = dat.loc[dict(DOY=slice(100, 125))].stack(z=('y', 'x')).chunk(dict(DOY=1, z=-1))
from src.hls_funcs.predict import pred_bm
t0 = datetime.datetime.now()
dat_bm = pred_bm(dat_stacked, bm_mod, dim='z').compute()
t1 = datetime.datetime.now()
print(t1-t0)

dat_sub = dat.loc[dict(DOY=slice(100, 125))].chunk(dict(DOY=1, x=-1, y=-1))
from src.hls_funcs.predict import pred_bm2
t0 = datetime.datetime.now()
dat_bm2 = pred_bm2(dat_sub, bm_mod).compute()
t1 = datetime.datetime.now()
print(t1-t0)


dat_stacked = dat.loc[dict(DOY=slice(100, 115))].stack(z=('y', 'x')).chunk(dict(DOY=1, z=-1))
from src.hls_funcs.predict import pred_cov
t0 = datetime.datetime.now()
dat_cov = pred_cov(dat_stacked, ends_dict, dim='z').compute()
t1 = datetime.datetime.now()
print(t1-t0)

dat_sub = dat.loc[dict(DOY=slice(100, 115))].chunk(dict(DOY=1, x=-1, y=-1))
from src.hls_funcs.predict import pred_cov2
t0 = datetime.datetime.now()
dat_cov2 = pred_cov2(dat_sub, ends_dict).compute()
t1 = datetime.datetime.now()
print(t1-t0)


class BiomassExplorer(param.Parameterized):
    date = param.CalendarDate(default=datetime.datetime(da_bm['YEAR'].values, 6, 15).date(),
                              bounds=(datetime.datetime(da_bm['YEAR'].values, 1, 1).date(),
                                      datetime.datetime(da_bm['YEAR'].values, 12, 31).date()))

    @param.depends('date')
    def load_map_bm(self):
        return da_bm.sel(DOY=self.date).hvplot(x='x', y='y', tiles='EsriImagery', crs=ccrs.UTM(13), responsive=True,
                                         cmap='inferno', clim=(100, 1000), colorbar=False)

calendar = pn.Param(
    BiomassExplorer.param,
    widgets={'date': pn.widgets.DatePicker}
)
bmxplore = BiomassExplorer()
map = hv.DynamicMap(bmxplore.load_map_bm)
pn.serve(pn.Row(pn.Column(calendar), pn.Column(map, sizing_mode='stretch_both')))






blue = flist_to_xr(flist=blue_files, band_str_in='blue', yr_str='2016', band_str_out='BLUE', chunks=chunks)
green = flist_to_xr(flist=green_files, band_str_in='green', yr_str='2016', band_str_out='GREEN', chunks=chunks)

test = xr.merge([blue, green], compat='broadcast_equals')



blue

idx = 0
for f in blue_files:



blue = os.path.join(geotiff_dir, 'HLS_MODIS_STARFM.T13TEF.2016200.blue.bin')
green = os.path.join(geotiff_dir, 'HLS_MODIS_STARFM.T13TEF.2016200.green.bin')
red = os.path.join(geotiff_dir, 'HLS_MODIS_STARFM.T13TEF.2016200.red.bin')
swir1 = os.path.join(geotiff_dir, 'HLS_MODIS_STARFM.T13TEF.2016200.swir1.bin')
swir2 = os.path.join(geotiff_dir, 'HLS_MODIS_STARFM.T13TEF.2016200.swir2.bin')
nir = os.path.join(geotiff_dir, 'HLS_MODIS_STARFM.T13TEF.2016200.nir.bin')


dat_blue = xr.open_rasterio(blue, chunks=chunks)
dat_green = xr.open_rasterio(blue, chunks=chunks)
dat_red = xr.open_rasterio(blue, chunks=chunks)
dat_swir2 = xr.open_rasterio(swir2, chunks=chunks)
dat_swir1 = xr.open_rasterio(swir1, chunks=chunks)
dat_nir = xr.open_rasterio(nir, chunks=chunks)

#dat = xr.Dataset(dict(BLUE=dat_blue, GREEN=dat_green, RED=dat_red, NIR1=dat_nir, SWIR1=dat_swir1, SWIR2=dat_swir2))
dat_stacked = dat.sel(dict(DOY=100)).stack(z=('y', 'x')).chunk(dict(z=-1))

from src.hls_funcs.predict import pred_bm
t0 = datetime.datetime.now()
## NEED TO STACK WITHIN THE FUNCTION?!
dat_bm = pred_bm(dat_stacked, bm_mod, dim='z').compute()
t1 = datetime.datetime.now()
print(t1-t0)



model_vars = [n for n in bm_mod.params.index if ":" not in n and "Intercept" not in n]

#bmask = bolton_mask(dat, 'band')
#dat = dat.where(bmask == 0)
#ndii7 = ndii7_func(dat)
#nir = nir_func(dat)
#bai_236 = bai_236_func(dat)
dat['B03'] = dat_blue
dat['B04'] = dat_nir

dat_bm, dat_bm_se = predict_biomass(dat, bm_mod, se=True)

from src.hls_funcs.predict import pred_cov
ends_dict = {
    'BARE': {
        'ndvi': 0.10,
        'dfi': 8,
        'bai_126': 140},
    'SD': {
        'ndvi': 0.30,
        'dfi': 16,
        'bai_126': 155},
    'GREEN': {
        'ndvi': 0.55,
        'dfi': 10,
        'bai_126': 160}}
dat_cov = pred_cov(dat_stacked, ends_dict, dim='z')
dat_cov.assign_coords(type='SD')