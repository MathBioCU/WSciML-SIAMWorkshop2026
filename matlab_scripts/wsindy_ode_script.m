%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script applies WSINDy to ODE data. Each of the ODEs in ode_names 
% below has associated defaults for 
% - tspan: time domain (1d array)
% - ode_params: ODE parameters (cell array)
% - x0: initial conditions 
% Check the file gen_ode_data for info on the parameters 
% dependence and dimensionality of the respective ODE

%% boiler plate

%%% add wsindy_obj_base to path
scriptsdir = fileparts(matlab.desktop.editor.getActiveFilename);
repodir = fileparts(scriptsdir);
addpath(genpath(repodir));

%%% restart with same rng seed or clear workspace and start from scratch
restart_run = false;
if ~restart_run
    rng('shuffle')
    close all; 
    clear;
end

%%% consolidate figures
set(0,'DefaultFigureWindowStyle','docked')

%% load data

ode_num = 'Lorenz';                   % select ODE from list below
tol_ode = 1e-12;                      % set tolerance (abs and rel) of ode45

tspan = []; ode_params = {}; x0 = []; % ODE system parameters
ode_names = {'Linear','Logistic_Growth','Van_der_Pol','Duffing',... %1-4
             'Lotka_Volterra','Lorenz','Rossler','rational',...     %5-8
             'Oregonator','Hindmarsh-Rose','Pendulum','custom'};    %9-12
[true_nz_weights,x,t,x0,ode_name,ode_params,rhs] = gen_ode_data(ode_num,ode_params,tspan,x0,tol_ode);

%% get wsindy_data object
Uobj = wsindy_data(x,t);

%%% Subsample data
subsample = -5000;
Uobj.coarsen(subsample);

%%% add noise data
noise_ratio = 0.0;
rng_seed = rng().Seed; rng(rng_seed);
Uobj.addnoise(noise_ratio,'seed',rng_seed);

%%% plot data
figure(1)
Uobj.plotDyn;
title(sprintf('Observed %s data: %i timepoints',ode_name,Uobj.dims))
xlabel('t')

%% select left-hand side

lhs_diff_ord = 1;
lhs_tags = get_tags(1,[],Uobj.nstates);
lhs = arrayfun(@(i)term('ftag',lhs_tags(i,:),'linOp',lhs_diff_ord),(1:Uobj.nstates)','uni',0);

%% get library

%%% define polynomial and trig orders
polys = 0:4;
trigs = [];

lib_tags = get_tags(polys,trigs,Uobj.nstates);
lib = library('tags',lib_tags);

%% get test function

%%% select test function family, radius selection method, spacing between tf
phifun = 'pp';
tau = 10^-10; tauhat = 1; max_deriv = 1;
tf_subinds = -4;

tf_param = {[tau tauhat max_deriv]};
tf_args = {'phifuns',phifun,'meth','FFT','param',tf_param,'subinds',tf_subinds};

tf = testfcn(Uobj,tf_args{:});

%% build WSINDy linear system

WS = wsindy_model(Uobj,lib,tf,'lhsterms',lhs);

%% optimize

lambdas = 10.^linspace(-4,0,200);
toggle_jointthresh = 4;

MSTLS_params = {'lambda',lambdas,'toggle_jointthresh',toggle_jointthresh};

tic;
[WS,loss_wsindy,its,G,b] = WS_opt().MSTLS_0(WS,MSTLS_params{:});
runtime = toc;

%% Diagnose

fprintf('\ndata dims =');
fprintf('%u ',Uobj.dims);
fprintf('\n')

Str_mod = WS.disp_mod;
resids = WS.res('sepcomp');
w_support = WS.get_supp;

for j=1:WS.numeq
    fprintf('\n----------Eq %i----------\n',j)
    fprintf('%s=\n',WS.lhsterms{j}.get_str)
    cellfun(@(s)fprintf('+%s\n',s),Str_mod{j})
    fprintf('\nrel resid =%1.3e\n', norm(resids{j}))
    fprintf('sparsity =%i\n',length(w_support{j}))
    fprintf('tf rad = %i\n',WS.tf{1}{j}.rads);
    [K,J] = size(WS.G{j});
    fprintf('size G = %d %d',K,J)
    fprintf('\ncond G = %1.1e\n',cond(WS.G{j}))
end

if ~isempty(loss_wsindy)
    figure(2);clf;
    f = min(loss_wsindy(1,:));
    g = min(loss_wsindy(2,loss_wsindy(1,:)==f));
    for j=1:size(loss_wsindy,1)-1
        loglog(loss_wsindy(end,:),loss_wsindy(j,:),'o-',g,f,'rx')
        hold on
    end
    hold off
    legend;
end

if exist('true_nz_weights','var')
    w_true = inject_true_weights(WS,true_nz_weights);
    Tps = tpscore(WS.weights,w_true);
    fprintf('\nTPR=%1.2f',Tps)
    E2 = norm(w_true-WS.weights)/norm(w_true);
    fprintf('\nCoeff err=%1.2e',E2)
end
fprintf('\nruntime=%1.2f(s)\n',runtime)


%% simulate learned and true reduced systems

toggle_compare = 1; 
toggle_display = 1;
if ~isempty(toggle_compare)

    tol_dd = 10^-12;
    x0_args = {[]};
    t_train = Uobj.grid{1};


    rhs_learned = WS.get_rhs;
    
    x0_data = Uobj.get_x0(x0_args{:});
    options_ode_sim = odeset('RelTol',tol_dd,'AbsTol',tol_dd*ones(1,Uobj.nstates));    
    [t_learned,x_learned]=ode15s(@(t,x)rhs_learned(x),t_train,x0_data,options_ode_sim);
    try
        rel_forward_err = arrayfun(@(j)norm(x_learned(:,j)-Uobj.Uobs{j})/norm(Uobj.Uobs{j}),1:Uobj.nstates);
        fprintf('\nrel forward err=')
        fprintf('%1.2e ',rel_forward_err)
    catch
        fprintf('\nsimulation and data are on different timegrids\n')
        rel_forward_err = NaN;
    end

    if toggle_display
        figure(3);clf
        for j=1:Uobj.nstates
            subplot(Uobj.nstates,1,j)
            plot(Uobj.grid{1},Uobj.Uobs{j},'b-o',t_learned,x_learned(:,j),'r-.','linewidth',2)
            try
                hold on
                plot(Uobj.grid{1},Uobj.Uobs{j}-Uobj.noise{j,1},'k-')
                legend({'data','learned','clean'})
            catch
                legend({'data','learned'})
            end
            title(['rel err=',num2str(rel_forward_err(j))])
        end
    end
end