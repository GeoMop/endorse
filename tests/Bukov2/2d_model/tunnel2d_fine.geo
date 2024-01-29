L = 100;  // velikost oblasti kolem tunelu
W = 4;    // sirka tunelu
H1 = 3.5; // vyska nezaoblene casti tunelu
H2 = 1;   // vyska zaoblene (horni) casti tunelu
R = 0.1;  // polomer zakriveni dolnich rohu tunelu

cl_far = 10;   // velikost elementu na krali oblasti
cl_near = 0.1; // velikost elementu u steny tunelu

Point(1) = { -L/2, 0, -L/2 };
Point(2) = { L/2, 0, -L/2 };
Point(3) = { L/2, 0, L/2 };
Point(4) = { -L/2, 0, L/2 };

Point(5) = { -W/2+R, 0, 0 };
Point(6) = { W/2-R, 0, 0 };
Point(7) = { W/2, 0, R, 0 };
Point(8) = { W/2, 0, H1 };
Point(9) = { 0, 0, H1+H2 };
Point(10) = { -W/2, 0, H1 };
Point(11) = { -W/2, 0, R };
Point(12) = { -W/2+R, 0, R };
Point(13) = { W/2-R, 0, R };
Point(14) = { 0, 0, H1 };

Line(1) = { 1,2 };
Line(2) = { 2,3 };
Line(3) = { 3,4 };
Line(4) = { 4,1 };

Line(5) = { 5,6 };
Line(6) = { 7,8 };
Line(7) = { 10,11 };
Circle(8) = { 11,12,5 };
Circle(9) = { 6,13,7 };
Ellipse(10) = { 8,14,9 };
Ellipse(11) = { 9,14,10 };

Line Loop(1) = { 1:4 };
Line Loop(2) = { 5,9,6,10,11,7,8 };
Plane Surface(1) = { 1,2 };

Physical Surface("rock") = { 1 };
Physical Line(".tunnel") = { 5:11 };
Physical Line(".outer_boundary") = { 1:4 };


Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

Field[1] = Distance;
Field[1].NNodesByEdge = 100;
Field[1].EdgesList = { 5:11 };
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = cl_near;
Field[2].LcMax = cl_far;
Field[2].DistMin = cl_near;
Field[2].DistMax = L/4;
Field[3] = Distance;
Field[3].NNodesByEdge = 100;
Field[3].EdgesList = { 1:4 };
Field[4] = Threshold;
Field[4].IField = 3;
Field[4].LcMin = cl_far;
Field[4].LcMax = cl_far;
Field[4].DistMin = cl_far;
Field[4].DistMax = cl_far;
Field[5] = Min;
Field[5].FieldsList = { 2,4 };
Background Field = 5;