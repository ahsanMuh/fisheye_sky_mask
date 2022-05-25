import numpy as np
from numpy import sin
from numpy import cos
from numpy import degrees as deg
from numpy import radians as rad

from datetime import datetime
import pandas as pd

import cv2
from PIL import Image

from mask_generator import img_to_maskpath
from circle_detection import detect_fish_circle


# required constants
dEarthMeanRadius = 6371008.7714
dAstronomicalUnit = 149597870700

#########################################
f = 0.0016 #camera fisheye focal length (m)
Lc = 0.00489 #camera sensor length (m)
Wc = 0.00367 #camera sensor width (m)
dx = 0.0000015 #pixel width (m/px)
dy = 0.0000015 #pixel height (m/px)
SVF_f = 1

diffuse_path = './CSV Weather Data/CSV Weather Data/Diffuse_2019.csv'
beam_path = './CSV Weather Data/CSV Weather Data/Direct_2019.csv'

def calculate_azimuth_altitude(dLatitude, dLongitude):
    """ 
    Calculates azimuth, altitude and the time based on the 
    longitude, latitude and the year
     """
    iYear = datetime.now().year 

    timeList = []
    azimuthList = []
    altitudeList = []
    
    for iMonth in range(1, 13):
        if iMonth == 4 or iMonth == 6 or iMonth == 9 or iMonth == 11:
            day = 30
        elif iMonth == 2:
            if iYear % 4 == 0:
                day = 29
            else:
                day = 28
        else:
            day = 31
        for iDay in range(1, day+1):
            for hour in range(0, 24):
                for dMinutes in range(0, 60, 15):
                    ## universal time zone
                    dHours = hour - 8
                    dSeconds = 0
                    # Calculate time of the day in UT decimal hours
                    ## breaking down a day into 10 intervals
                    dDecimalHours = dHours + (dMinutes + dSeconds / 60) / 60
                    # Calculate current Julian Day
                    ## Julian day is the continuous count of days since the beginning of the Julian period
                    ## Julian period is a chronological interval of 7980 years; year 1 of the Julian Period was 4713 BC
                    liAux1 = (iMonth - 14) / 12
                    liAux2 = (1461 * (iYear + 4800 + liAux1)) / 4 + (367 * (iMonth - 2 - 12 * liAux1)) / 12 - \
                             (3 * ((iYear + 4900 + liAux1) / 100)) / 4 + iDay - 32075
                    dJulianDate = liAux2 - 0.5 + dDecimalHours / 24.0
                    # Calculate difference between current Julian Day and JD 2451545.0
                    dElapsedJulianDays = dJulianDate - 2451545.0

                    # Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
                    # ecliptic in radians but without limiting the angle to be less than 2*Pi
                    # (i.e., the result may be greater than 2*Pi)
                    ## Obliquity of the ecliptic is the term used by astronomers for the 
                    ## inclination of Earth's equator with respect to the ecliptic, or of Earth's rotation axis 
                    ## to a perpendicular to the ecliptic. 
                    dOmega = 2.1429 - 0.0010394594 * dElapsedJulianDays
                    dMeanLongitude = 4.8950630 + 0.017202791698 * dElapsedJulianDays # Radians
                    dMeanAnomaly = 6.2400600 + 0.0172019699 * dElapsedJulianDays
                    dEclipticLongitude = dMeanLongitude + 0.03341607 * np.sin(dMeanAnomaly) \
                                         + 0.00034894 * np.sin(2 * dMeanAnomaly) - 0.0001134 \
                                         - 0.0000203 * np.sin(dOmega)
                    dEclipticObliquity = 0.4090928 - 6.2140e-9 * dElapsedJulianDays\
                                         + 0.0000396 * np.cos(dOmega)

                    # Calculate celestial coordinates(right ascension and declination ) in radians
                    # but without limiting the angle to be less than 2*Pi(i.e., the result may be greater than 2*Pi)
                    dSin_EclipticLongitude = np.sin(dEclipticLongitude)
                    dY = np.cos(dEclipticObliquity) * dSin_EclipticLongitude
                    dX = np.cos(dEclipticLongitude)
                    dRightAscension = np.arctan2(dY,dX)
                    if dRightAscension < 0:
                        dRightAscension = dRightAscension + (2 * np.pi)
                    dDeclination = np.arcsin(np.sin(dEclipticObliquity)*dSin_EclipticLongitude)

                    # Calculate local coordinates ( azimuth and zenith angle ) in degrees
                    ## alttitude is the vertical angle with sun
                    ## azimuth is the horizontal angle starting 
                    dGreenwichMeanSiderealTime = 6.6974243242 + 0.0657098283*dElapsedJulianDays + dDecimalHours
                    dLocalMeanSiderealTime = np.radians(dGreenwichMeanSiderealTime * 15 + dLongitude)
                    dHourAngle = dLocalMeanSiderealTime - dRightAscension
                    dLatitudeInRadians = np.radians(dLatitude)
                    dCos_Latitude = np.cos(dLatitudeInRadians)
                    dSin_Latitude = np.sin(dLatitudeInRadians)
                    dCos_HourAngle = np.cos(dHourAngle)
                    dZenithAngle = (np.arccos(dCos_Latitude*dCos_HourAngle*np.cos(dDeclination) + np.sin(dDeclination)*dSin_Latitude))
                    dY = -np.sin(dHourAngle)
                    dX = np.tan(dDeclination)*dCos_Latitude - dSin_Latitude*dCos_HourAngle
                    dAzimuth = np.arctan2(dY,dX)
                    if dAzimuth<0:
                        dAzimuth = dAzimuth + 2*np.pi
                    dAzimuth = np.degrees(dAzimuth)
                    # Parallax Correction
                    dParallax = (dEarthMeanRadius/dAstronomicalUnit)*np.sin(dZenithAngle)
                    dZenithAngle = np.degrees(dZenithAngle + dParallax)

                    dAltitude = np.degrees(np.arcsin(np.cos(np.radians(dZenithAngle))))


                    # Stack the array of parameters including time, azimuth and altitude
                    timeList.append(
                        pd.Timestamp(datetime(iYear, iMonth, iDay, hour, dMinutes))
                    )
                    azimuthList.append(dAzimuth)
                    altitudeList.append(dAltitude)
                
    data = pd.DataFrame({
            'time' : timeList,
            'azimuth' : azimuthList,
            'altitude': altitudeList
            })
    
    return data 

def grey_outer(img, centre, radius):
    """ Greys out the part of the image outside of the fisheye circle using the img, centre, radius """
    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)
    _ = cv2.circle(mask, centre, radius, (255, 255, 255), thickness=-1)
    
    masked_data = cv2.bitwise_and(img, img, mask=mask)
    # convert outside pixels to gray
    for x in range(width):
        for y in range(height):
            if masked_data[y, x].all() == 0:
                masked_data[y, x] = [192, 192, 192]
    
    return masked_data


def getCameraCoordinates(altitude, azimuth):
    """ Converts altitude and azimuth into image coordinates """
    xc = cos(altitude) * sin(azimuth)
    yc = cos(altitude) * cos(azimuth)
    zc = sin(altitude)
    return xc, yc, zc


def draw_sun(data, imgTest):
    """ draws the sun using altitude and azimuth lists """
    sunCoordinateX = []
    sunCoordinateY = []
    x0 = imgTest.shape[0] / 2
    y0 = imgTest.shape[1] / 2
    # Select the time of interest, and calculate the corresponding sun coordinates in camera coordinates
    for altitude, azimuth in zip(data['altitude'], data['azimuth']):
        altitude = rad(altitude)
        azimuth = rad(azimuth)
        if deg(altitude) > 0:
            xc, yc, zc = getCameraCoordinates(altitude, azimuth)
            x = xc / zc
            y = yc / zc

            Xi = (f/dx) * x + x0
            Yi = (f/dy) * y + y0

            # Stack the calculated coordinates in two arrays
            sunCoordinateX.append(int(Xi))
            sunCoordinateY.append(int(Yi))
        else:
            sunCoordinateX.append(-1)
            sunCoordinateY.append(-1)

    data['sunCoordinateX'] = sunCoordinateX
    data['sunCoordinateY'] = sunCoordinateY
    
    return data

def sky_work(data, imgTest):
    # calculating fsh_b and sky_pixels:
    fsh_b = []
    for x, y in zip(data['sunCoordinateX'], data['sunCoordinateY']):
        if x > 0 and y > 0 and x < imgTest.shape[0] and y < imgTest.shape[1]:
            if imgTest[x, y].all():
                fsh_b.append(0)
            else:
                fsh_b.append(1)
        else:
            fsh_b.append(-1)
    
    data['fsh_b'] = fsh_b
    return data

def preprocess_weather_data(path):
    """ Prerpocess the weather data as per the requirements """
    df = pd.read_csv(path)
    cur_year = str(datetime.now().year)

    df['date'] = df['yyyymmdd'].map(str).map(
                            lambda x: cur_year + x[4:] if x.startswith('2019') else x)
    df['time'] = df['hhmm'].map(str).map(
                                    lambda x:('0' * (4 - len(x))) + x)
    df['datetime'] = pd.to_datetime(df['date'] + df['time'].replace('2400', '0000'), 
                                        format='%Y%m%d%H%M')
    
    return df[['datetime', 'r']].set_index('datetime')

def img_to_shading_factor(org_im, dLatitude, dLongitude):
    """ Takes image and location and returns a DF with shading factor for each month """

    data = calculate_azimuth_altitude(dLatitude, dLongitude)

    print('Read the image, getting the mask ...')
    sky_area = img_to_maskpath(org_im)
    print('Got the mask, reading the mask ...')
    mask = Image.open('mask.png')

    mask = np.array(mask)
    org = np.array(org_im)

    centre, radius = detect_fish_circle(org)
    org = grey_outer(org, centre, radius)

    if org_im.width > org_im.height:
        org = np.rot90(org, 3)

    alpha3 = np.stack([mask]*3, axis=2)
    imgTest = cv2.subtract(org, alpha3)
    print('Subtracted the images, about to plot sun ...')

    data = draw_sun(data, imgTest)
    data = sky_work(data, imgTest)

    circle_area = np.pi * radius ** 2
    SVF = sky_area / circle_area
    fsh_d = SVF / SVF_f
    data['fsh_d'] = fsh_d

    diffuse_df = preprocess_weather_data(diffuse_path)
    beam_df = preprocess_weather_data(beam_path)

    data = data.assign(
    H_b = lambda y: y['time'].apply(
        lambda x: float(beam_df.loc[x, ['r']]) if x in beam_df.index else -1),
    H_d = lambda y: y['time'].apply(
        lambda x: float(diffuse_df.loc[x, ['r']]) if x in diffuse_df.index else -1)
    )

    results = data.groupby(pd.Grouper(key='time', freq='1M')).sum()[['H_b', 'H_d']]
    results = results.assign(
        H_b_H_d = lambda x: x['H_b'] + x['H_d'],
        v_b = lambda y: y.apply(
                lambda x: x['H_b'] / x['H_b_H_d'] if x['H_b'] > 0  else x['H_b'], axis=1),
        v_d = lambda y: y.apply(
            lambda x: x['H_d'] / x['H_b_H_d'] if x['H_d'] > 0  else x['H_d'], axis=1)
        )

    filtered = data[(data['H_b'] != 0) & (data['fsh_b'] != -1)]
    results['fsh_b'] = filtered.groupby(pd.Grouper(key='time', freq='1M')).mean()['fsh_b']
    results['fsh_d'] = fsh_d
    results['fsh'] = results.apply(lambda x: (x['fsh_b'] * x['v_b']) + (x['fsh_d'] * x['v_d']), axis=1)

    data.to_csv('v2_data_test.csv', index=False)
    results.to_csv('v2_results_test.csv', index=False)

    return results
