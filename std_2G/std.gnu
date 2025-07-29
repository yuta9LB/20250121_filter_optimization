set style data lines
set parametric
set ticslevel 0
set xlabel"X-direction"
set ylabel"Y-direction"
set zlabel"Z-direction"
set xrange [0:0.0727082341909408569]
set yrange [0:0.0449319742619991302]
set zrange [0:0.0189000051468610764]
splot"std.insu" title "Insulator" with l lt 2,"std.cond" title "Conductor" with l lt 1,"std.feeds" title "Feed" with p pt 3
