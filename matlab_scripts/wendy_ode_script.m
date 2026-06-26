%% add wsindy_obj_base to path
restart_run = false;

%% add wsindy_obj_base to path

fullPathToScript = mfilename('fullpath');
currentDir = fileparts(fullPathToScript);
parentDir = fileparts(currentDir);
addpath(genpath(parentDir))

if ~restart_run
    rng('shuffle')
    clear all;
    close all; 
end

set(0,'DefaultFigureWindowStyle','docked')

% %% generate data (Lorenz96)
% 
% nstates = 5;       % Number of variables
% F0 = 8;            % Forcing term
% self_damping = 1;  % strength of self-damping
% nn_coupling = 1;   % strength of nearest-neighbor forcing
% pnn_coupling = 1;  % strength of next-nearest-neighbor damping
% 
% M = 2048;
% tspan = linspace(0,10,M); % Time interval for integration
% x0 = F0 * ones(nstates, 1) + randn(nstates, 1) * 0.1; % ICs
% 
% ode_function = @(t, x) lorenz96ode(t, x, nstates, F0, self_damping, nn_coupling, pnn_coupling); 
% [t, x] = ode45(ode_function, tspan, x0); % clean data
% [true_nz_weights,w_true] = trueweights_Lorenz96(nstates, F0, self_damping, nn_coupling, pnn_coupling); % true terms / coefficients
% true_prod_tags = cellfun(@(w)w(:,1:end-1),true_nz_weights,'uni',0);
% 

%%

ode_num = 'Lorenz';                   % select ODE from list below
tol_ode = 1e-12;                      % set tolerance (abs and rel) of ode45

tspan = []; ode_params = {}; x0 = []; % ODE system parameters
ode_names = {'Linear','Logistic_Growth','Van_der_Pol','Duffing',... %1-4
             'Lotka_Volterra','Lorenz','Rossler','rational',...     %5-8
             'Oregonator','Hindmarsh-Rose','Pendulum','custom'};    %9-12
[true_nz_weights,x,t,x0,ode_name,ode_params,rhs] = gen_ode_data(ode_num,ode_params,tspan,x0,tol_ode);
true_nz_weights = true_nz_weights(:);
w_true = cell2mat(cellfun(@(eq) eq(:,end), true_nz_weights(:),'un',0));

%%% get library
true_prod_tags = cellfun(@(w)w(:,1:end-1),true_nz_weights,'uni',0);
lib = cellfun(@(tm)library('tags',tm),true_prod_tags);

%%% coarsen data
M_obs = 256; % number of observed timepoints
noise_ratio = 0.25; % noise ratio

%%% get wsindy_data object
Uobj = wsindy_data(x,t);
Uobj.coarsen(-M_obs); 

%%% apply noise
rng('shuffle')
rng_seed = rng().Seed; rng(rng_seed);
Uobj.addnoise(noise_ratio,'seed',rng_seed,'uniform',0);

%% wendy parameters

%%% test function params
tf_type = 'Cinf'; % 'pp'
rad_type = 'FFT'; % 'FFT','direct','timefrac'
toggle_SVD_tf = 0;
toggle_strong_form = 0;
subinds = -3; % subsample convolution for speed

%%% optimization alg params
wendy_params = {'maxits',100,'ittol',10^-4,'diag_reg',10^-inf,'trim_rows',1};

%%% viewing params 
toggle_compare = 1;
tol_dd = 10^-12;

%% WENDy

%%% get test function
eta = 9;
if isequal(tf_type,'Cinf')
    phifun = @(x) exp(-eta*(1-x.^2).^(-1)); 
elseif isequal(tf_type,'pp')
    phifun = @(x) (1-x.^2).^eta; 
end

if toggle_SVD_tf % get tf SVD
    K_max = 5000;
    eta = 9;
    subinds_svd = 1;
    center_scheme = 'uni';
    toggle_VVp_svd = 0.999; %NaN; % default NaN. 0, no SVD reduction; in (0,1), truncates Frobenious norm; NaN, truncates SVD according to cornerpoint of cumulative sum of singular values
    mt_params = 2.^(0:3);               % see get_rad.m:  default 2.^(0:3)
    K_min = length(w_true);
    mt_max = max(floor((Uobj.dims-1)/2)-K_min,1);
    
    tf = get_VVp_tf(Uobj,phifun,subinds_svd,rad_type,mt_params,[],K_min,toggle_VVp_svd);
else
    if toggle_strong_form==1
        phifun = 'delta';
        tf_meth = 'direct';
        tf_param = 1; % centered 2nd order FD
    else
        %%% FFT radius selection, piecewise-poly test functions
        if and(isequal(tf_type,'pp'), isequal(rad_type,'FFT'))
            phifun = 'pp';
            tau = 10^-10; tauhat = 2;
            tf_param = {[tau tauhat 1]};
        elseif isequal(rad_type,'timefrac')
            tf_param = [0.15];
        elseif isequal(rad_type,'FFT')
            tf_param = 2;
        elseif isequal(rad_type,'direct')
            tf_param = 15;
        end
    end
    tf = arrayfun(@(i)testfcn(Uobj,'phifuns',phifun,'subinds',subinds,...
        'meth',rad_type,'param',tf_param,'stateind',i),1:Uobj.nstates,'uni',0);
end
fprintf('\ntf rads=');fprintf('%u ',tf{1}.rads);fprintf('\n')

tic;
%%% instantiate WENDy model
wendy_model_class = [1 1];
WS = wendy_model(Uobj,lib,tf,wendy_model_class);

%% solve for coefficients
[WS,w_its,res,res_0,CovW,RT] = WS_opt().wendy(WS,wendy_params{:});
total_time_wendy = toc;

%% results

Str_mod = WS.disp_mod;
for j=1:WS.numeq
    fprintf('\n----------Eq %u----------\n',j)
    fprintf('%s = ',WS.lhsterms{j}.get_str)
    cellfun(@(s)fprintf('%s\n',s),Str_mod{j})
end

if exist('true_nz_weights','var')
    fprintf('\nRel. resid. =')
    fprintf('%1.2e ',norm(WS.res))
    E2 = norm(w_true-WS.weights)/norm(w_true);
    fprintf('\nCoeff err=%1.2e',E2)
end

errs = vecnorm(w_its-w_true,2,1)/norm(w_true);
disp(['-----------------'])
disp([' '])
disp(['rel L2 errs (OLS, WENDy)=',num2str(errs([1 end]))])
disp(['runtime(s)=',num2str(total_time_wendy)])
disp(['num its=',num2str(size(w_its,2))])

%% simulate learned and true reduced systems, display results
if toggle_compare==1
    w_plot = w_its(:,end);
    rhs_learned = WS.get_rhs('w',w_plot);
    options_ode_sim = odeset('RelTol',tol_dd,'AbsTol',tol_dd*ones(1,Uobj.nstates));

    t_train = Uobj.grid{1};
    x0_reduced = Uobj.get_x0([]);
    [t_learned,x_learned]=ode15s(@(t,x)rhs_learned(x),t_train,x0_reduced,options_ode_sim);
    figure(7);clf
    for j=1:Uobj.nstates
        subplot(Uobj.nstates,1,j)
        plot(Uobj.grid{1},Uobj.Uobs{j},'b-o',t_learned,x_learned(:,j),'r-.','linewidth',2)
        try
            title(['rel err=',num2str(norm(x_learned(:,j)-Uobj.Uobs{j})/norm(Uobj.Uobs{j}))])
        catch
        end
        legend({'data','learned'})
    end
    
    if exist('w_its','var')
        try
            figure(8)
            plot_wendy;
        catch
            w_true = w_plot*0;
        end
    end
end

%% Functions

function dXdt = lorenz96ode(t, X, n, F0, sd, nn, pnn)
    % This function defines the Lorenz-96 ODE system.
    dXdt = zeros(n, 1); % Initialize the derivative vector
    for j = 1:n
        % Implement the cyclic boundary conditions
        X_prev2 = X(mod(j-3+n, n) + 1); % X_{j-2}
        X_prev1 = X(mod(j-2+n, n) + 1); % X_{j-1}
        X_next1 = X(mod(j, n) + 1);    % X_{j+1}
    
        dXdt(j) = (nn*X_next1 - pnn*X_prev2) * X_prev1 - sd*X(j) + F0;
    end
end

function [true_nz_weights,w_true] = trueweights_Lorenz96(n, F0, sd, nn, pnn)

    true_nz_weights = cell(n,1);

    e = zeros(1,n);
    e0 = e; % forcing
    e1 = e; e1(1) = 1; % self damping
    e2 = e; e2(2) = 1; e2(end) = 1;  % nearest neighbor forcing
    e3 = e; e3(end) = 1; e3(end-1) = 1; % two previous damping
    E = [e0;e1;e2;e3];

    w = [F0;-sd;nn;-pnn];

    true_nz_weights{1} = [E,w];

    for j=2:n
        true_nz_weights{j} = [circshift(E,j-1,2),w];
    end

    w_true = repmat(w,n,1); 
    
end
