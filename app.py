"""SolarScope: local solar research workspace."""
import base64
import hashlib
from datetime import date, timedelta
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from core import demo_data
from research import (clean_history, import_kaggle, fetch_nasa, fit_models, forecast,
                      future_scenario, causal_features, validate_capacity, report_markdown, KAGGLE_URL)
from model_io import export_bundle

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='SolarScope | Group 8', page_icon='☀️', layout='wide')
alt.data_transformers.disable_max_rows()
st.markdown('<style>'+ROOT.joinpath('style.css').read_text()+'</style>', unsafe_allow_html=True)
st.markdown('<div class="topline"><span>WORKSPACE &nbsp; / &nbsp; ENERGY INTELLIGENCE</span><span class="status-pill">Oct 22, 2026</span></div>', unsafe_allow_html=True)
asset = ROOT/'assets/solar-field.png'
if not asset.exists(): asset = ROOT/'solar-field.png'
if asset.exists():
    encoded = base64.b64encode(asset.read_bytes()).decode()
    st.markdown(f'''<section class="cosmic-hero" style="background-image:url(data:image/png;base64,{encoded})">
    <div class="cosmic-label">SOLARSCOPE &nbsp; / &nbsp; GROUP 8</div><h1>A new perspective<br>on solar energy.</h1>
    <p>From the sun’s first light to tomorrow’s forecast.<br>Explore, compare, and understand your energy.</p>
    <div class="cosmic-footer">DATA &nbsp; · &nbsp; INSIGHT &nbsp; · &nbsp; FORECAST</div></section>''', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">☼</span>SolarScope</div><div class="side-label">Plant workspace</div>', unsafe_allow_html=True)
    page = st.radio('Workspace navigation', ['Overview','Forecast','Data','Report'], label_visibility='collapsed',key='workspace_nav')
    st.markdown('<div class="side-label">Data source</div>', unsafe_allow_html=True)
    source = st.radio('Historical data', ['Demo plant','Upload CSV','Kaggle plant','NASA POWER'])
    st.caption('Generation: kW · Radiation: W/m² · Temperature: °C')
    st.markdown('<div class="side-label">Forecast settings</div>', unsafe_allow_html=True)
    epochs = st.slider('Maximum training epochs', 5, 100, 20)
    hours = st.select_slider('Forecast horizon (hours)', [24,48,72,168], value=24)
    st.download_button('Download sample history', demo_data().to_csv(index=False), 'sample_history.csv','text/csv')
    st.markdown('<div class="side-foot">GROUP 8 · SOLAR ENERGY PROJECT<br>Analyze. Compare. Forecast.</div>', unsafe_allow_html=True)

st.header({'Overview':'Plant performance overview','Forecast':'Forecast your solar generation','Data':'Manage your plant data','Report':'Your solar energy report'}[page])
st.caption({'Overview':'Explore generation trends and the effect of weather.','Forecast':'Use city weather for an estimate, or train a model with plant history.','Data':'Select a source in the sidebar, inspect your records, and download data.','Report':'Read and download a report based on your current dataset and model results.'}[page])
if page == 'Forecast':
    from accuweather import weather_panel
    with st.container(border=True):
        weather_panel()
    st.divider()

source_description = source
try:
    if source == 'Demo plant':
        source_description = 'Synthetic demonstration data (100 kW demo plant)'
        df, cleaning = clean_history(demo_data())
        st.caption('DEMO PLANT · 100 kW · Synthetic data. Accuracy does not represent real-world performance.')
    elif source == 'Upload CSV':
        uploaded = st.file_uploader('Hourly or 15-minute history CSV',type='csv',key='history_upload')
        st.caption('Required: timestamp, irradiance_wm2, temperature_c, power_kw. Optional: humidity_pct, cloud_cover_pct, dc_power_kw, module_temperature_c.')
        if uploaded is None:
            st.info('Upload history, or choose Demo plant to explore.')
            st.stop()
        df, cleaning = clean_history(pd.read_csv(uploaded))
    elif source == 'Kaggle plant':
        st.subheader('Connect your Kaggle plant')
        st.link_button('Open Solar Power Generation Data',KAGGLE_URL)
        sample_dir = ROOT/'sample_data'
        samples = sorted(p.name.split('_Generation')[0] for p in sample_dir.glob('*_Generation_Data.csv'))
        use_sample = st.checkbox('Use included Kaggle sample',value=True) if samples else False
        if use_sample:
            sample = st.selectbox('Sample plant',samples)
            generation = pd.read_csv(sample_dir/f'{sample}_Generation_Data.csv')
            weather = pd.read_csv(sample_dir/f'{sample}_Weather_Sensor_Data.csv')
        else:
            left,right = st.columns(2)
            gen_file = left.file_uploader('Generation CSV',type='csv',key='kaggle_generation')
            met_file = right.file_uploader('Weather sensor CSV',type='csv',key='kaggle_weather')
            if gen_file is None or met_file is None:
                st.info('Add the original Kaggle generation/weather CSV pair for one or both plants.')
                st.stop()
            generation,weather = pd.read_csv(gen_file),pd.read_csv(met_file)
        plants = sorted(set(generation.PLANT_ID.astype(str)) & set(weather.PLANT_ID.astype(str)))
        plant = st.selectbox('Plant ID',plants)
        divisor = st.number_input('DC unit correction divisor',min_value=.001,value=1.,help='Verify the source DC scale. 10 divides raw DC by 10; the AC forecast target is never altered.')
        df,cleaning = import_kaggle(generation,weather,plant,divisor)
        source_description = f'Kaggle plant {plant}'
        st.caption('Fixed inverter fleet summed per interval; complete quarter-hours averaged to hours. Humidity/cloud cover are absent and are not invented. Verify DC units before interpreting AC/DC ratios.')
    else:
        st.subheader('NASA POWER · historical weather')
        st.caption('Ranchi, Jharkhand is the default. NASA supplies historical weather, not measured power or future forecasts. Timestamps are UTC.')
        left,right = st.columns(2)
        latitude = left.number_input('Latitude',-90.,90.,23.3441,format='%.4f')
        longitude = right.number_input('Longitude',-180.,180.,85.3096,format='%.4f')
        start = left.date_input('Start date',date(2025,1,1))
        end = right.date_input('End date',date(2025,1,31),max_value=date.today()-timedelta(days=1))
        key = (latitude,longitude,str(start),str(end))
        if st.button('Fetch NASA weather',type='primary'):
            with st.spinner('Retrieving NASA weather…'):
                st.session_state.nasa_weather = (key,fetch_nasa(latitude,longitude,start,end))
        saved = st.session_state.get('nasa_weather')
        if saved is None or saved[0] != key:
            st.info('Choose a location and dates, then fetch weather.')
            st.stop()
        nasa = saved[1]
        st.line_chart(nasa.set_index('timestamp')[['irradiance_wm2']])
        st.download_button('Download NASA weather CSV (UTC)',nasa.to_csv(index=False),'nasa_weather_utc.csv','text/csv')
        power_file = st.file_uploader('Optional measured power CSV in UTC (timestamp, power_kw)',type='csv',key='nasa_power')
        if power_file is None:
            st.info('Weather is ready. Add measured power to train. Convert IST to UTC by subtracting 5 hours 30 minutes and aggregate into UTC hourly bins before upload.')
            st.stop()
        power = pd.read_csv(power_file)
        power['timestamp'] = pd.to_datetime(power.timestamp,errors='raise')
        df,cleaning = clean_history(nasa.merge(power[['timestamp','power_kw']],on='timestamp',how='left',validate='one_to_one'))
        source_description = f'NASA POWER UTC weather + measured power ({latitude}, {longitude})'
except Exception as error:
    st.error(f'Data could not be loaded: {error}')
    st.stop()

fingerprint = hashlib.sha256((df.to_csv(index=False)+source_description).encode()).hexdigest()
if st.session_state.get('data_key') != fingerprint:
    st.session_state.pop('comparison',None)
    st.session_state.data_key = fingerprint
if page == 'Overview':
    cols = st.columns(3)
    cols[0].metric('Observed energy',f'{df.power_kw.sum():,.0f} kWh')
    cols[1].metric('Peak generation',f'{df.power_kw.max():.1f} kW')
    cols[2].metric('Average daily energy',f'{df.power_kw.sum()/(len(df)/24):,.0f} kWh')
    
    st.caption('Hourly mean kW × 1 hour gives kWh. Missing hours are excluded; daily averages divide by the full calendar span and may underestimate yield when coverage is incomplete.')

if page == 'Overview':
    with st.container(border=True):
        title_col,filter_col = st.columns([4,1])
        title_col.subheader('Power generation')
        window = filter_col.selectbox('Show recent history',[1,7,30,120],format_func=lambda n:f'{n} days',label_visibility='collapsed')
        chart = alt.Chart(df.tail(window*24)).encode(x=alt.X('timestamp:T',title=None),y=alt.Y('power_kw:Q',title='Power (kW)'),tooltip=['timestamp:T',alt.Tooltip('power_kw:Q',format='.1f')])
        st.altair_chart(chart.mark_area(line={'color':'#dea119','strokeWidth':2},color='#f7dda0').properties(height=250),width='stretch')
    left,right = st.columns(2)
    left.subheader('Daily energy · kWh')
    left.bar_chart(df.set_index('timestamp').power_kw.resample('D').sum(min_count=1),color='#dfa94d',height=220)
    right.subheader('Monthly energy · kWh')
    right.bar_chart(df.set_index('timestamp').power_kw.resample('MS').sum(min_count=1),color='#248d70',height=220)
    daylight_eda = st.checkbox('Daylight-only exploratory analysis',value=True,help='Excludes irradiance ≤10 W/m² from weather analysis. Energy charts retain nights.')
    eda = df[df.irradiance_wm2 > 10] if daylight_eda else df
    corr_cols = [c for c in ['irradiance_wm2','power_kw','dc_power_kw','temperature_c','module_temperature_c','humidity_pct','cloud_cover_pct'] if c in eda and eda[c].notna().any()]
    correlation = eda[corr_cols].corr()
    left,right = st.columns(2)
    left.metric('Irradiance / AC correlation',f'{correlation.loc["irradiance_wm2","power_kw"]:.3f}')
    peak = int(df.groupby(df.timestamp.dt.hour).power_kw.mean().idxmax())
    right.metric('Highest mean generation hour',f'{peak:02d}:00')
    st.subheader('Correlation heatmap')
    friendly = {'irradiance_wm2':'Irradiance','power_kw':'AC power','dc_power_kw':'DC power','temperature_c':'Ambient °C','module_temperature_c':'Module °C','humidity_pct':'Humidity','cloud_cover_pct':'Cloud cover'}
    matrix = correlation.rename(index=friendly,columns=friendly).rename_axis('Variable').reset_index().melt('Variable',var_name='Compared with',value_name='Correlation')
    heat = alt.Chart(matrix).encode(x=alt.X('Variable:N',title=None,axis=alt.Axis(labelAngle=-35)),y=alt.Y('Compared with:N',title=None))
    tiles = heat.mark_rect().encode(color=alt.Color('Correlation:Q',scale=alt.Scale(domain=[-1,1],scheme='redblue'),legend=alt.Legend(orient='top',direction='horizontal',gradientLength=100)),tooltip=['Variable:N','Compared with:N',alt.Tooltip('Correlation:Q',format='.3f')])
    labels = heat.mark_text(fontSize=11).encode(text=alt.Text('Correlation:Q',format='.2f'),color=alt.condition('abs(datum.Correlation) > 0.65',alt.value('white'),alt.value('#163c36')))
    st.altair_chart((tiles+labels).properties(height=300),width='stretch')
    left,right = st.columns(2)
    left.subheader('Radiation & generation')
    left.altair_chart(alt.Chart(eda).mark_circle(color='#dfa94d',size=22,opacity=.5).encode(x=alt.X('irradiance_wm2:Q',title='Irradiance (W/m²)'),y=alt.Y('power_kw:Q',title='AC power (kW)'),tooltip=[alt.Tooltip('irradiance_wm2:Q',format='.1f'),alt.Tooltip('power_kw:Q',format='.1f')]).properties(height=260),width='stretch')
    right.subheader('Temperature & normalized output')
    temp = 'module_temperature_c' if 'module_temperature_c' in eda else 'temperature_c'
    normalized = eda[eda.irradiance_wm2 > 200].copy()
    normalized['kw_per_kwm2'] = normalized.power_kw/(normalized.irradiance_wm2/1000)
    right.altair_chart(alt.Chart(normalized).mark_circle(color='#248d70',size=22,opacity=.5).encode(x=alt.X(f'{temp}:Q',title=friendly[temp],scale=alt.Scale(zero=False)),y=alt.Y('kw_per_kwm2:Q',title='Normalized output'),tooltip=[alt.Tooltip(f'{temp}:Q',format='.1f'),alt.Tooltip('kw_per_kwm2:Q',format='.1f')]).properties(height=260),width='stretch')
    right.caption('kW per kW/m² at radiation >200 W/m². Observational proxy, not panel efficiency; panel area and plane-of-array radiation are required for that.')
    if 'cloud_cover_pct' in eda:
        cloud = eda.copy()
        cloud['Cloud cover'] = pd.cut(cloud.cloud_cover_pct,[0,25,50,75,100],include_lowest=True).astype(str)
        st.subheader('Power by cloud cover · mean kW')
        st.bar_chart(cloud.groupby('Cloud cover').power_kw.mean(),color='#408e9b')
        st.caption('These associations may be confounded by time of day, season, and irradiance.')

if page == 'Forecast':
    st.subheader('Compare forecasting models')
    st.write('Compare a linear baseline, tree models, and optional neural models. Earliest 70% of usable rows: training. Next 15%: validation and model selection. Final 15%: held-out test.')
    names = st.multiselect('Models',['Linear Regression','Random Forest','XGBoost','TensorFlow Dense','TensorFlow LSTM'],default=['Linear Regression','Random Forest','XGBoost'])
    daylight = st.checkbox('Train and evaluate on daylight hours only',value=False)
    st.caption('Features: hour, month, day of year, cyclical time, previous-hour/previous-day power, and shifted 3-hour/24-hour rolling means. Imputation and scaling fit training rows only.')
    if st.button('Train selected models',type='primary',disabled=not names,key='train_models'):
        try:
            with st.spinner('Training and evaluating unseen hours…'):
                st.session_state.comparison = fit_models(df,names,daylight,epochs)
            st.success('Comparison complete. Forecasts and exports are ready.')
        except Exception as error:
            st.error(f'Training could not complete: {error}')
    result = st.session_state.get('comparison')
    if result:
        st.caption(f"Rows: {result['splits']['train']} train / {result['splits']['validation']} validation / {result['splits']['test']} test. Selection uses validation RMSE, not test scores.")
        st.dataframe(result['metrics'],hide_index=True,width='stretch')
        st.success('Selected by validation performance: '+result['best'])
        st.subheader('Actual vs. predicted · held-out hours')
        st.line_chart(result['predictions'].set_index('timestamp').drop(columns='irradiance_wm2'))
        st.caption('Rolling one-hour-ahead test: observed weather and observed earlier power are available for each test hour. This does not establish day-ahead recursive accuracy. R² is undefined for zero target variance.')
        for name,loss in result['losses'].items():
            st.subheader(name+' · learning curves')
            st.line_chart(loss.rename(columns={'loss':'Training','val_loss':'Validation'}))

if page == 'Forecast':
    st.subheader('Future power generation')
    result = st.session_state.get('comparison')
    scenario = future_scenario(df,hours)
    st.download_button('Download future weather template',scenario.to_csv(index=False),'future_weather_template.csv','text/csv')
    if result is None:
        st.info('Choose models above and click Train selected models to unlock predictions from your plant history. City weather estimates are available at the top of this page.')
    else:
        choice = st.selectbox('Forecast model',list(result['bundles']),index=list(result['bundles']).index(result['best']))
        bundle = result['bundles'][choice]
        future_file = st.file_uploader('Future hourly weather CSV',type='csv',key='future')
        st.caption('Use the same time standard as history. Module-temperature features also require future estimates if included during training.')
        try:
            future = pd.read_csv(future_file) if future_file else scenario
            if len(future) < hours:
                raise ValueError(f'Supply at least {hours} future weather rows.')
            output = forecast(bundle,df,future.head(hours))
            if future_file is None:
                st.warning('Scenario only: prior-seven-day hourly weather averages, not a weather forecast. NASA POWER supplies history, not future forecasts.')
            st.metric('Forecast energy',f'{output.predicted_power_kw.sum():,.1f} kWh')
            st.line_chart(output.set_index('timestamp')[['predicted_power_kw']],color=['#248d70'])
            st.download_button('Download forecast CSV',output.to_csv(index=False),'solar_forecast.csv','text/csv')
        except Exception as error:
            st.error(str(error))
        with st.expander('Try your own weather · next-hour prediction'):
            st.caption('Uses measured recent generation for lag features. Prediction time: '+str(df.timestamp.iloc[-1]+pd.Timedelta(hours=1)))
            with st.form('manual_weather'):
                inputs = {}
                for col in bundle['weather_columns']:
                    default = float(scenario[col].iloc[0])
                    bounds = {'min_value':0.,'max_value':100.} if col in ['humidity_pct','cloud_cover_pct'] else ({'min_value':0.} if col == 'irradiance_wm2' else {})
                    inputs[col] = st.number_input(col,value=default,**bounds)
                submitted = st.form_submit_button('Predict next hour')
            if submitted:
                try:
                    row = pd.DataFrame([dict(timestamp=df.timestamp.iloc[-1]+pd.Timedelta(hours=1),**inputs)])
                    instant = forecast(bundle,df,row)
                    st.metric('Predicted output',f'{instant.predicted_power_kw.iloc[0]:.2f} kW')
                except Exception as error:
                    st.error(str(error))

if page == 'Data':
    st.subheader('Cleaned data & traceability')
    st.json(cleaning)
    st.caption('Complete hourly means. Missing power remains missing. Weather forward-fill is limited to two hours. Night-time rows are preserved in the calendar series.')
    st.dataframe(df,hide_index=True,width='stretch')
    st.download_button('Download cleaned history',df.to_csv(index=False),'cleaned_history.csv','text/csv')
    st.download_button('Download engineered features',pd.concat([df[['timestamp','power_kw']],causal_features(df)],axis=1).to_csv(index=False),'features.csv','text/csv')
    result = st.session_state.get('comparison')
    if result:
        st.download_button('Download evaluation metrics',result['metrics'].to_csv(index=False),'model_metrics.csv','text/csv')
        st.download_button('Download test predictions',result['predictions'].to_csv(index=False),'test_predictions.csv','text/csv')
        model_name = st.selectbox('Model to export',list(result['bundles']))
        if st.button('Prepare model download'):
            try:
                st.download_button('Download model + preprocessing',export_bundle(result['bundles'][model_name]),'solar_model.zip','application/zip')
            except Exception as error:
                st.error(f'Could not export: {error}')

if page == 'Data':
    st.subheader('State-wise renewable capacity')
    st.write('Compare sourced installed solar, wind, and other renewable capacity. Capacity (MW) is separate from plant energy (kWh) and is not a model input.')
    left,right = st.columns(2)
    left.link_button('MNRE capacity reports','https://mnre.gov.in/installed-capacity/')
    right.link_button('CEA power reports','https://cea.nic.in/reports/external/daily-generation?lang=en')
    st.download_button('Download capacity CSV template','state,solar_mw,wind_mw,other_re_mw,as_of,source_url\n','capacity_template.csv','text/csv')
    st.caption('Transcribe one report date. Include source URLs; define non-overlapping categories in other_re_mw.')
    capacity_file = st.file_uploader('State capacity CSV',type='csv',key='capacity')
    if capacity_file:
        try:
            cap = validate_capacity(pd.read_csv(capacity_file))
            st.caption('Report date: '+str(cap.as_of.iloc[0].date()))
            st.bar_chart(cap.set_index('state')[['solar_mw','wind_mw','other_re_mw']],color=['#dfa94d','#248d70','#408e9b'])
            st.dataframe(cap,hide_index=True,width='stretch')
            st.download_button('Download capacity analysis',cap.to_csv(index=False),'capacity_analysis.csv','text/csv')
        except Exception as error:
            st.error(str(error))
    else:
        st.info('Upload sourced state figures to generate a comparison. No fabricated capacity values are prefilled.')

if page == 'Report':
    st.subheader('Your project report')
    st.caption('All requested sections, with current data, cleaning summary, measured results, limitations, and references.')
    document = report_markdown(df,st.session_state.get('comparison'),source_description,cleaning)
    st.download_button('Download project report · Markdown',document,'SolarScope_Project_Report.md','text/markdown')
    with st.expander('Read the report'):
        st.markdown(document)
st.markdown('<div class="footer">SOLARSCOPE &nbsp; / &nbsp; Renewable energy, understood. &nbsp; · &nbsp; Group 8</div>', unsafe_allow_html=True)
