% Smets and Wouters (AER, 2007)
% Script for Table1A and Table1B
% Runs sw_table1.mod in Dynare
% Jerome Williams, NYU
% June 2013

clear

structural_names = {'phi','sigma_c','lambda','xi_w','sigma_l','xi_p','iota_w','iota_p','psi','phi_p','r_pi','rho','r_y','r_dy','pi_bar','beta_const','l_bar','gamma_bar','alpha'};
shocks_names_1 = {'eta_a','eta_b','eta_g','eta_i','eta_r','eta_r','eta_p','eta_w'};
shocks_names_2 = {'rho_a','rho_b','rho_g','rho_i','rho_r','rho_p','rho_w','Mu_p','Mu_w','rho_ga'};

Table1a = zeros(19,4);
Table1b = zeros(16,4);

dynare sw_table1.mod noclearall;
	
to = oo_.posterior_mode.parameters;
te = oo_.posterior_mean.parameters;
t5 = oo_.posterior_hpdinf.parameters;
t95 = oo_.posterior_hpdsup.parameters;
% structural parameters
for i = 1:length(structural_names)
	param_name = char(structural_names(i));
	Table1a(i,:) = [to.(param_name) te.(param_name) t5.(param_name) t95.(param_name)];
end
% shock standard deviations
so = oo_.posterior_mode.shocks_std;
se = oo_.posterior_mode.shocks_std;
s5 = oo_.posterior_hpdinf.shocks_std;
s95 = oo_.posterior_hpdsup.shocks_std;
for i = 1:length(shocks_names_1)
	param_name = char(shocks_names_1(i));
	Table1b(i,:) = [so.(param_name) se.(param_name) s5.(param_name) s95.(param_name)];
end
offset = i;
% shock autocorrelation and cross-correlation
for i = 1:length(shocks_names_2)
	param_name = char(shocks_names_2(i));
	Table1b(offset+i,:) = [to.(param_name) te.(param_name) t5.(param_name) t95.(param_name)];
end

clear to te t5 t95 so se s5 s95 offset;

Table1a
Table1b
