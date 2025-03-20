#!/usr/bin/env python

"""
Author: Lucile Gaultier & Severine Fournier

include
- bissextile(year) -- return 1 if leap year
- daysinmonth(year, month) -- return the number of days in month for year
- date2jj(year, month, day) -- return the corresponding Julian Day
- jj2date(jj) -- return the year month and day corresponding to the Julian day jj
- doy2date(year, yday) -- return the year, month, day corresponding to the year and day of year
- date2doy(year, month, day) -- return the corresponding day of year
- jj2doy(jj) -- return the corresponding day of year and year
- doy2jj(year,doy) -- return the corresponding Julian day

"""

import numpy
import sys

def bissextile(year):
  biss=0
  if numpy.mod(year,400)==0:
    biss=1
  if (numpy.mod(year,4)==0 and numpy.mod(year,100)!=0):
    biss=1
  return biss
  
  
def daysinyear(year):
  biss = bissextile(year)
  if biss:
      daysinyear = 366
  else:
      daysinyear = 365
  return daysinyear


def daysinmonth(year, month):
  if month>12 :
    sys.exit("wrong month in daysinmonth")
  biss = bissextile(year)
  if (month==4 or month==6 or month==9 or month==11):
    dinm = 30
  elif month==2:
    if biss:
      dinm = 29
    else:
      dinm = 28
  else:
    dinm = 31
  return dinm


def date2jj(syear, smonth, sday): 
  year = int(syear)
  month = int(smonth)
  day = int(sday)
  refyear = 1950
  jday = day

  if day>31:
    sys.exit("wrong day") 

  for imonth in numpy.arange(1,month):
    dinm = daysinmonth(year,imonth)
    jday = jday + dinm
  
  for iyear in numpy.arange(refyear,year):
    biss=bissextile(iyear)
    if biss:
      daysinyear = 366
    else:
      daysinyear = 365      
    jday = jday + daysinyear
  
  return jday


def datehh2jj(syear, smonth, sday, hh, mm , ss): 
  import numpy 
  
  year = int(syear)
  month = int(smonth)
  day = int(sday)
  refyear = 1950
  jday = day

  if day>31:
    sys.exit("wrong day") 

  for imonth in numpy.arange(1,month):
    dinm = daysinmonth(year,imonth)
    jday = jday + dinm
  
  for iyear in numpy.arange(refyear,year):
    biss=bissextile(iyear)
    if biss:
      daysinyear = 366
    else:
      daysinyear = 365      
    jday = jday + daysinyear
  
  jday =  numpy.float(jday) + ss/60./60./24 + mm/60./24. + hh/24.
  
  return jday


def date2jj0(syear, smonth, sday): 
  year = int(syear)
  month = int(smonth)
  day = int(sday)
  refyear = 0
  jday = day

  if day>31:
    sys.exit("wrong day") 

  for imonth in numpy.arange(1,month):
    dinm = daysinmonth(year,imonth)
    jday = jday + dinm
  
  for iyear in numpy.arange(refyear,year):
    biss=bissextile(iyear)
    if biss:
      daysinyear = 366
    else:
      daysinyear = 365      
    jday = jday + daysinyear
  
  return jday


def jj2date(sjday):
  jday=int(sjday)
  year=1950
  month=1
  day=0

  for iday in numpy.arange(1,jday+1):
    day+=1
    dinm = daysinmonth(year, month)
    if (day > dinm):
      day = 1
      month = month + 1
    if (month > 12):
      month = 1
      year = year + 1
  return year, month, day


def jj2datehh(sjday):
  import numpy
  
  year=1950
  month=1
  day=0

  jday = numpy.floor(sjday)
  res = sjday-jday #en jours
  hh = numpy.floor(res*24)
  mm = numpy.floor((res*24-hh)*60)
  ss = numpy.floor(((res*24-hh)*60-mm)*60)
  for iday in numpy.arange(1,jday+1):
    day+=1
    dinm = daysinmonth(year, month)
    if (day > dinm):
      day = 1
      month = month + 1
    if (month > 12):
      month = 1
      year = year + 1
  return year, month, day, hh, mm, ss


def jj2date0(sjday):
  jday=int(sjday)
  year=0
  month=1
  day=0

  for iday in numpy.arange(1,jday+1):
    day+=1
    dinm = daysinmonth(year, month)
    if (day > dinm):
      day = 1
      month = month + 1
    if (month > 12):
      month = 1
      year = year + 1
  return year, month, day


def doy2date(syear, syday):
  year = int(syear)
  yday = int(syday)
  if yday>366: sys.exit('Year day should be less than 366')
# Leap year
  biss = bissextile(year)
  if biss:
    days_in_prev_month = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
  else:
    days_in_prev_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

  for i in range(0,len(days_in_prev_month)):
      if yday>days_in_prev_month[i] and yday<=days_in_prev_month[i+1]:
          day = yday-days_in_prev_month[i]
          month = i+1
  return year, month, day


def date2doy(syear, smonth, sday):
  year = int(syear)
  month = int(smonth)
  day = int(sday)
  if day>31:
    sys.exit("wrong day")
  if month>12:
    sys.exit("wrong month")

#leap year
  biss = bissextile(year)
  if biss: 
    day_prev_month = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
  else:
    day_prev_month=[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
  yday = day_prev_month[month-1]+day
  return yday



def jj2doy(jday):
  [yy,mm,dd] = jj2date(jday)
  doy = date2doy(yy,mm,dd)
  
  return doy



def doy2jj(year,doy):
  [yy,mm,dd] = doy2date(year,doy)
  jday = date2jj(yy,mm,dd)
  
  return jday