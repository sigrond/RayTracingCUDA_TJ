function eff_app = effective_aperture(lens_radius,lens_barrel_thickness,FFL,lambda,temperature)
n_BK7=Calculate_m(temperature,lambda,'BK7'); % add tempereture dispersion 4 BK7
appfun = @(h3) (lens_radius-h3)*(lens_radius-h3)*((lens_barrel_thickness/h3)*(lens_barrel_thickness/h3)+1-n_BK7)-FFL*FFL*n_BK7*n_BK7;
eff_app = (lens_radius - fzero(appfun,0.1*lens_radius))*2;