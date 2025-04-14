set style data lines
set parametric
set ticslevel 0
set xlabel"X-direction"
set ylabel"Y-direction"
set zlabel"Z-direction"
set xrange [0:0.0996677875518798828]
set yrange [0:0.0593646690249443054]
set zrange [0:0.0159600097686052322]
splot"std.insu" title "Insulator" with l lt 2,"std.cond" title "Conductor" with l lt 1,"std.feeds" title "Feed" with p pt 3
