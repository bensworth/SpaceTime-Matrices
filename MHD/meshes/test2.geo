// Gmsh project created on Sun May 30 18:10:37 2021
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {-0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Field[1] = BoundaryLayer;
//+
Field[1].EdgesList = {1};
//+
Physical Line("North", 1) = {3};
//+
Physical Line("EastWest", 2) = {4, 2};
//+
Physical Line("South", 3) = {1};
