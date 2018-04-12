Output images are in Output folder along with results of inference ie. Miss detection and False alarm
p1,p2,..,p6 images show ouput of predictions
f1,f2,..,f6 images show ouput of false alarm
m1,m2,..,m6 images show ouput of miss detection

1. Logistic Regression
	-is in file LogisticRegression.m
	-uses fit_logr.m and fit_logr_cost.m
	-change is made in fit_logr_cost.m for initialization
2. Bayesian Logistic Regression
	-is in file BayesianLogisticRegression.m
	-uses fit_blogr.m and fit_blogr_cost.m
	-change is made in fit_blogr_cost.m and eps is added
3. Dual Logistic Regression
	-is in file DualLogisticRegression.m
	-uses fit_dlogr.m and fit_dlogr_cost.m
	-change is made in fit_dlogr_cost.m and eps is added
4. Dual Bayesian Logistic Regression
	-is in file DualBayesianLogisticRegression.m
	-uses fit_dblogr.m and fit_dblogr_cost.m
	-change is made in fit_dblogr_cost.m and eps is added
Similar for other two files
