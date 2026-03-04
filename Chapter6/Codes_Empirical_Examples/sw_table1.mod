@#include "sw_defs.mod"
@#include "sw_model_def.mod"
@#include "sw_estimation_def.mod"

estimation(
	nograph,
	nodiagnostic,
	optim=('Algorithm','active-set'),
	datafile=usmodel_data,
	mode_compute=1,
	first_obs=71,
	presample=4,
	lik_init=2,
	prefilter=0,
	mh_replic=2000,
	mh_nblocks=2,
	mh_jscale=0.20,
	mh_drop=0.2
);
// Increase mh_replic to get a better estimation of the posterior distribution.

