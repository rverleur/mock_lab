function [Absorbance_Fit]    = Voigt_Approx_McLean_Vectorized_Fit_Demo(Free_Parameters,v,Absorbance_Exp)

global Absorbance_Fit  vd


% Parameters for Voigt profile algorithm (from McLean)
C1                  = [-1.2150 1.2359 -0.3085 0.0210];
C2                  = [-1.3509 0.3786 0.5906 -1.1858];
C3                  = [-1.2150 -1.2359 -0.3085 -0.0210];
C4                  = [-1.3509 -0.3786 0.5906 1.1858];



Phi_D_vo            = (2/vd)*sqrt(log(2)/pi);       %Calculate doppler line shape at vo
a1                  = sqrt(log(2))*vc1/vd;              %Calculate Voigt 'a' parameter
w1                  = 2*sqrt(log(2)) .* (v-vo1) ./ vd;   %Voigt 'w' parameter
Voigt1a             = ( C1(3) * (a1 - C1(1)) + C1(4) .* (w1 - C1(2)) ) ./ ( (a1 - C1(1))^2 + (w1 - C1(2)).^2 );
Voigt1b             = ( C2(3) * (a1 - C2(1)) + C2(4) .* (w1 - C2(2)) ) ./ ( (a1 - C2(1))^2 + (w1 - C2(2)).^2 );
Voigt1c             = ( C3(3) * (a1 - C3(1)) + C3(4) .* (w1 - C3(2)) ) ./ ( (a1 - C3(1))^2 + (w1 - C3(2)).^2 );
Voigt1d             = ( C4(3) * (a1 - C4(1)) + C4(4) .* (w1 - C4(2)) ) ./ ( (a1 - C4(1))^2 + (w1 - C4(2)).^2 );
Voigt1              = Voigt1a + Voigt1b + Voigt1c + Voigt1d;
Phi_V1              = Phi_D_vo .* Voigt1;
Absorbance_Fit1     = Aint1*Phi_V1;         %Calculate absorbance



a2                  = sqrt(log(2))*vc2/vd;              %Calculate Voigt 'a' parameter
w2                  = 2*sqrt(log(2)) .* (v-vo2) ./ vd;   %Voigt 'w' parameter
Voigt2a             = ( C1(3) * (a2 - C1(1)) + C1(4) .* (w2 - C1(2)) ) ./ ( (a2 - C1(1))^2 + (w2 - C1(2)).^2 );
Voigt2b             = ( C2(3) * (a2 - C2(1)) + C2(4) .* (w2 - C2(2)) ) ./ ( (a2 - C2(1))^2 + (w2 - C2(2)).^2 );
Voigt2c             = ( C3(3) * (a2 - C3(1)) + C3(4) .* (w2 - C3(2)) ) ./ ( (a2 - C3(1))^2 + (w2 - C3(2)).^2 );
Voigt2d             = ( C4(3) * (a2 - C4(1)) + C4(4) .* (w2 - C4(2)) ) ./ ( (a2 - C4(1))^2 + (w2 - C4(2)).^2 );
Voigt2              = Voigt2a + Voigt2b + Voigt2c + Voigt2d;
Phi_V2              = Phi_D_vo .* Voigt2;
Absorbance_Fit2     = Aint2*Phi_V2;         %Calculate absorbance

shift               = C1 + C2*v;

Absorbance_Fit      = Absorbance_Fit1 + Absorbance_Fit2 + shift;
