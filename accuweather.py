"""AccuWeather integration. Credentials stay on the server."""
import math
import requests
import pandas as pd
BASE='https://dataservice.accuweather.com'
def get_weather(path,key,**params):
    try:
        r=requests.get(BASE+path,headers={'Authorization':'Bearer '+key,'Accept-Encoding':'gzip,deflate'},params=params,timeout=25)
    except requests.RequestException:
        raise ValueError('Weather service could not be reached. Please try again.') from None
    if r.status_code != 200:
        raise ValueError({401:'API key is invalid or expired.',403:'Your subscription does not include this forecast.',429:'Weather request limit reached. Please try later.'}.get(r.status_code,'Weather service is temporarily unavailable.'))
    return r.json()
def solar_estimate(rows,latitude,capacity,performance):
    out=[]
    for row in rows:
        t=pd.Timestamp(row['DateTime']); temp=float(row['Temperature']['Value']); cloud=row.get('CloudCover')
        if cloud is None: raise ValueError('This response has no cloud cover. Detailed hourly weather is required for the solar estimate.')
        # Solar geometry approximation: local clock, not a measured irradiance sensor.
        decl=math.radians(23.44*math.sin(2*math.pi*(284+t.dayofyear)/365))
        lat=math.radians(latitude); angle=math.radians(15*(t.hour+t.minute/60-12))
        elevation=max(0,math.sin(lat)*math.sin(decl)+math.cos(lat)*math.cos(decl)*math.cos(angle))
        radiation=1000*elevation*(1-.75*(float(cloud)/100)**3.4) if row.get('IsDaylight',True) else 0
        cell=temp+.03*radiation
        power=max(0,min(capacity,capacity*radiation/1000*performance*(1-.004*(cell-25))))
        out.append({'Local time':t,'Temperature (°C)':temp,'Cloud cover (%)':cloud,'Estimated radiation (W/m²)':radiation,'Estimated power (kW)':power})
    return pd.DataFrame(out)
def weather_panel():
    import os
    import streamlit as st
    st.header('Plan your solar energy')
    st.write('Choose your city and date to see the weather and estimated solar generation.')
    key=os.getenv('ACCUWEATHER_API_KEY','')
    secret=__import__('pathlib').Path(__file__).parent/'.streamlit/secrets.toml'
    if not key and secret.exists():
        import tomllib
        key=tomllib.loads(secret.read_text(encoding='utf-8-sig')).get('ACCUWEATHER_API_KEY','')
    if not key: key=st.text_input('AccuWeather API key',type='password',help='Used only by this server to request AccuWeather data.')
    city=st.text_input('City or town',value='Ranchi, India')
    if st.button('Find location',type='primary'):
        if not key: st.info('Enter an API key to continue.')
        else:
            try: st.session_state.weather_places=get_weather('/locations/v1/cities/search',key,q=city)
            except ValueError as e: st.error(str(e))
    places=st.session_state.get('weather_places',[])
    if not places:
        st.caption('Start by finding your location. Available dates depend on the forecast returned by your subscription.')
        return
    place=st.selectbox('Choose the correct location',places,format_func=lambda p:f"{p['LocalizedName']}, {p['AdministrativeArea']['LocalizedName']}, {p['Country']['LocalizedName']}")
    a,b=st.columns(2)
    capacity=a.number_input('Solar system size (kW)',min_value=.1,value=100.)
    performance=b.slider('System performance (%)',50,100,80,help='Allows for inverter, wiring, shading and other losses.')/100
    if st.button('Load available weather'):
        try:
            rows=get_weather('/forecasts/v1/hourly/12hour/'+place['Key'],key,details='true',metric='true')
            st.session_state.weather_result=(place['Key'],rows)
        except ValueError as e: st.error(str(e))
    saved=st.session_state.get('weather_result')
    if not saved or saved[0]!=place['Key']: return
    try:
        df=solar_estimate(saved[1],place['GeoPosition']['Latitude'],capacity,performance)
        days=list(dict.fromkeys(t.date() for t in df['Local time']))
        day=st.selectbox('Date to analyse',days,format_func=lambda d:d.strftime('%A, %d %B %Y'))
        selected=df[df['Local time'].map(lambda t:t.date()==day)]
        st.caption('AccuWeather · next 12 hourly forecasts · local time at your selected city. Only dates supplied by the API can be selected.')
        a,b,c=st.columns(3)
        a.metric('Energy in available hours',f"{selected['Estimated power (kW)'].sum():.1f} kWh")
        b.metric('Peak estimated output',f"{selected['Estimated power (kW)'].max():.1f} kW")
        c.metric('Hours available',len(selected))
        st.line_chart(selected.set_index('Local time')[['Estimated power (kW)']],color=['#dfa94d'])
        st.info('Solar output is a weather-based engineering estimate, not measured generation or a trained-model accuracy result. Radiation is approximated from latitude, local clock and cloud cover; panel tilt, solar-time correction and site shading are not modeled. Partial days are not full-day totals.')
        with st.expander('Weather and calculation details'): st.dataframe(selected,hide_index=True)
        st.download_button('Download this analysis',selected.to_csv(index=False),'weather_solar_analysis.csv','text/csv')
        st.link_button('Weather source: AccuWeather','https://www.accuweather.com/')
    except (ValueError,KeyError,TypeError) as e:
        st.error(str(e) if isinstance(e,ValueError) else 'The weather response is missing required details.')
