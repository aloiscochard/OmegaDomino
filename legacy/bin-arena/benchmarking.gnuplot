# set term png
# set output 'history.png'
#
set terminal x11 background rgb 'black'
set border lc rgb 'white'
set key tc rgb 'white'

plot filename using 1:2 with lines, filename using 1:3 with lines, filename using 1:4 with lines, filename using 1:5 with lines, filename using 1:6 with lines, filename using 1:7 with lines, 0 title ""
pause 2
reread
