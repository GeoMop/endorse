## More complex data vizualizations


## Bukov II URF map "georeferencing"
For origin in upper-left corner and X for horizontal axis, 
the six crosshair marks in the map has pixel coordinates
(from top-left to bottm right):
                JTSK shifted
(x,y)           (X,Y)
(527, 184)      
(1576, 184)     
(527, 1202)     (-100, 0)
(1576, 1202)    (0, 0)
(527, 2282)
(1576, 2282)
(527, 3331)
(1576, 3331)

sx = dX/dx = 100/1049 = 0.09533
sy = -sx = -0.09533
Tx = 0 - 2.04082 * 1576 = -150.24
Ty = 0 - 2.04082 * 1202 = -219.64
In Local coordinate system has origin at (-622600, -1127800) JTSK. This point should
