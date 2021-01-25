# Art C Gan v0.1

Music Visualising tool using python

## Process



## Features
Feature breakdown:
``peak_num``: Number of peaks above threshold. A higher number of peaks signifies a more saturated sound.
``means``: The mean of the spectrum location. Lower value of the mean signifies track is more 'bass heavy'
``loudness``: Measures the loudness of the track or 'power' of the spectrum [FIX: power]
``peaks_trend``: The rate of change of the number of peaks above threshold
``means_trend``: The rate of change of the peaks mean - signifies how fast the track is 'progressing' [FIX -second der.]

## Weight Function

## Rank
The ranking (currently) is a single digit as the CGAN is trained for only two 'energy levels'

## CGAN

## Note
- pip package ``simpleaudio`` requires linux package ``libasound2-dev``, [see here](https://github.com/pioneers/PieCentral/issues/710)

## TODO
- pywt instead of essentia
