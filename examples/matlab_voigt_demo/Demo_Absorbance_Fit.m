clear all
close all
clc
global Absorbance_Fit  vd


load Example_Absorbance_Data.mat

Tguess              = 1400;
vo                  = 2060;
MW                  = 28;

% Determine guess values for spectroscopic parameters needed for
% least-squares fitting
vd                  = vo*(7.1623e-7)*(Tguess/MW)^(1/2);         % Doppler width (cm^-1)
vo_guesses          = [ -0.6  -0.18];                          % Linecenter frequency of each transition
peak_guesses        = [ 0.4    0.08];                           % Peak absorbance, used for determing guess of A
vc_guesses          = [ 0.04   0.04];                           % Collisional width guess
Aint1_guess         = get_IntArea_guess(vd,vc_guesses(1,1),peak_guesses(1,1));  % Convert peak absorbance guess to integrated absorbance guess
Aint2_guess         = get_IntArea_guess(vd,vc_guesses(1,2),peak_guesses(1,2));
Free_Parameters     = [vo_guesses(1,1) vc_guesses(1,1) Aint1_guess vo_guesses(1,2) vc_guesses(1,2) Aint2_guess]; % Bundle free-parameters

% Use nlinfit to find best-fit absorbance spectra
options             = statset('MaxIter',200);
estimates           = nlinfit(v,Absorbance_Exp,@Voigt_Approx_McLean_Vectorized_Fit_Demo,Free_Parameters,options);
Linecenters         = [estimates(1,1) estimates(1,4)]
Collisional_Widths  = [estimates(1,2) estimates(1,5)]
Integrated_Areas    = [estimates(1,3) estimates(1,6)]


figure(1)
set(gcf,'WindowStyle','docked')
axes1 = axes('Parent',figure(1),'FontName','Arial','FontSize',32);
hold(axes1,'all')
plot(v,Absorbance_Exp,'k','linewidth',2)
plot(v,Absorbance_Fit,'r--','linewidth',2)
xlabel('Relative Frequency, cm^{-1}','FontName','Arial','FontSize',32)
ylabel('Absorbance','FontName','Arial','FontSize',32)
legend('Data','Best Fit')
