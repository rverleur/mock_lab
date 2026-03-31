function [Area]     = get_IntArea_guess(vd,vc,peak_abs)

%This function converts a peak absorbance guess to an integrated area guess
%that can be used to initiate Voigt profile fitting. It requires a Doppler
%width, collisional width, and peak absorbance guess values and returns an
%appropriate integrated area guess


%%
Phi_D_vo    = (2/vd)*sqrt(log(2)/pi);   %Calculate doppler line shape at vo
a           = sqrt(log(2))*vc/vd;       %Calculate Voigt 'a' parameter

V_vo        = exp(a^2)*erfc(a);         %Calculate Voigt function at linecenter
Voigt_vo    = Phi_D_vo*V_vo;            %Calculate the Voigt profile at linecenter

Area        = peak_abs/Voigt_vo;        %Calculate integrated area (SPiL) 