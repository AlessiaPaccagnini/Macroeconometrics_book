// Replication of Smets and Wouters (AER, 2007)
// This file is included by
// - sw_table1.mod
// Jerome Williams, NYU, June 2013
// 
// This is essentially a rewrite of Smets&Wouters' own code, which is available at
// http://www.aeaweb.org/articles.php?doi=10.1257/aer.97.3.586

var
	// observed variables
	labobs robs pinfobs dy dc dinve dw
	// misc
	eta_w_ma eta_p_ma
	// flexible price quantites
	zcapf rkf kf pkf cf invef yf labf wf rrf
	// stick price quantities
	mc zcap rk k_s q c i y l pinf w r
	// innovations
	eps_a eps_b eps_g eps_i eps_r eps_p eps_w
	// capital-related
	kpf k;    
 
varexo
	eta_a eta_b eta_g  eta_i eta_r eta_p eta_w;  

parameters
	curv_w rho_ga curv_p l_bar pi_bar beta_const
	Mu_w Mu_p
	alpha psi phi delta sigma_c lambda
	phi_p iota_w xi_w iota_p xi_p sigma_l phi_w 
	r_pi r_dy r_y rho 
	rho_a
	rho_b rho_g
	rho_i
	rho_r rho_p rho_w  
	gamma_bar 
	G;

// fixed parameters
delta = 0.025;
phi_w = 1.5;
G = 0.18;
curv_p = 10;
curv_w = 10;

// initial values (are these necessary?)
shocks;
	var eta_a; stderr 0.4618;
	var eta_b; stderr 1.8513;
	var eta_g; stderr 0.6090;
	var eta_i; stderr 0.6017;
	var eta_r; stderr 0.2397;
	var eta_p; stderr 0.1455;
	var eta_w; stderr 0.2089;
end;