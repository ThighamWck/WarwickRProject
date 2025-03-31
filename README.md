# WarwickRProject
Code for Thomas Higham's fourth year research project on learning transition paths in turbulent domains with neural operators

The numbers 1 and 2 refer to whether the domain is square or not-square as detailed in my report. 

“Trig_polynomial_boundary.py" generates the boundary domains, implementing Algorithm 1 from the report.

“turbulent_velocity_field.py" generates random synthetic turbulence by implementing Algorithm 2 from the report.

These files, and the finite difference solvers, are used by the training data files to generate csv files.

The training data can be used in “FNO_1.py" and “FNO_2.py", for square and non-square boundaries respectively.

The FNOs can be tested in “FNO_1_Test.py" and “FNO_2_Test.py" respectively using a pth file for the neural operator and a csv file for the training data.
