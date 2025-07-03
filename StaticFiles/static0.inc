Display 0

gnuplot 0
Gerber 1 (-139, -109, 3) (158, 109, 3) 0

TimeSteps 0
TimeWeight 0.995

CellSize X -165 201 0.000272315
CellSize Y -109 109 0.000272315
CellSize Z  -30  33 0.000253333333

Feed 1 (-139, -3, 0) (-139, 3, 3) +Z 1 1 0 60e+9 3.16 0.0

ABC_Type UPML
PML_Parameters 8 4.0 -60.0

# Simulation Area
Area Vacuum (-165, -109, -30) (201, 109, 33)

# Board
RO4350B  Cube  (-165, -109, 0) (201, 109, 3)
Copper Plane (-165, -109, 0) (201, 109, 0)

# 50 ohm Line
Copper Plane (-165, -3, 3) (201, 3, 3)

# Define Parameters
Define Vacuum (1.0, 1.0, 0.0) 10.0e+9
Define Copper (1.0, 0.999991, 6.17e+7) 10.0e+9
Define RO4350B (3.48, 1.0, 0.0037) 1.0e+9

# Observe
Observe Vf (-100, 0, 0) (-100, 0, 3) +Z
Observe Vf (100, 0, 0) (100, 0, 3) +Z

# Marker
Copper Plane (-135, -93, 3) (-133, -91, 3)
Copper Plane (-135,  91, 3) (-133,  93, 3)

Copper Plane (149, -93, 3) (151, -91, 3)
Copper Plane (149,  91, 3) (151,  93, 3)
