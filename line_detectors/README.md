# Line detector list

## Pylene (Ours)

The entire source code is available on the following [gitlab](https://gitlab.lre.epita.fr/olena/pylene)

## AG3LINE

We use the source code available on the following [github](https://github.com/weidong-whu/AG3line).

## Cannylines

We use the source code available on the following [github](https://github.com/ludlows/CannyLine).

## Edlines

We use the source code available on the following [github](https://github.com/CihanTopal/ED_Lib).

## Elsed

We use the source code available on the following [github](https://github.com/iago-suarez/ELSED).

## Houghlines (OpenCV)

We use the opencv API available in [here](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a)

## LSD

We use the LSD implementation [version 1.6](http://www.ipol.im/pub/art/2012/gjmr-lsd/revisions/2012-03-24/lsd_1.6.zip).

**LSD_m** is LSD but with a filter on the length of the lines included.

# Adding line detectors

You can add new C++/C line detectors. It requires:
- to be named as `lsd_<name>`
- to have a opencv CLI with some specific arguments (input, output, help)

Look all currently available folder to have examples.



