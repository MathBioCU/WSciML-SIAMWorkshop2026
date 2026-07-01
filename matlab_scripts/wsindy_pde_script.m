%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script applies WSINDy to PDE data. Each of the PDEs files in pde_names 
% has the essential variables
% xs: cell array containing 1D grids which tensored together give the full computational grid
% U_exact: cell array containing state observations on the full computational grid
% lhs: list of integers of the form [p1 ... pn q1 ... qD] denote the left-hand-side term
% (d/dx1)^q1 ... (d/dxD)^dQ (u1^p1 ... un^pn) where u denotes state variables, x denotes grid coordinates
% in addition, the variable true_nz_weights contains information on the true WSINDy coefficients and terms
% used to generate the data

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

%%% choose PDE

dr = 'pde_data/';
% dr = 'C:/Users/385570/Desktop/data/WSINDy_PDE_zenodo/';
pde_names = {'burgers.mat',...          
             'KS.mat',...                
             'NLS.mat',...               
             'porous2.mat',...     
             'sod_exact.mat',...
             'Nav_stokes.mat',...
             'Sine_Gordon.mat',...
             'lin_schrod2.mat',...
             'rxn_diff.mat',...
             'wendy_hyperKS.mat'
         ...
    };

pde_num = 3; % set to 0 to run on pre-loaded dataset

if pde_num~=0
    pde_name = pde_names{pde_num};
    load([dr,pde_name],'U_exact','lhs','true_nz_weights','xs')
else
    pde_name = 'custom';
end

%% create data object
Uobj = wsindy_data(U_exact,xs);

%%% coarsen spacetime grid
subsample = 1;
Uobj.coarsen(subsample);

%%% add noise
noise_ratio = 0.3;
Uobj.addnoise(noise_ratio);

%%% set library
x_diffs = [0:5];%%% differential operators
polys = [0:5]; trigs = [];%%% poly/trig functions
custom_add =  {...  %%% custom terms using term algebra
        % term('fHandle',@(u,v) exp(sin(u+u.^2))),...                               % arbitrary term specified by function handle    
        % compterm(term('ftag',2), diffOp([1,0],'stateind',2)),...                   % term nonlinear in a derivative
        % prodterm(term('ftag',[-2i 2i]), diffOp([2,0],'stateind',1, 'nstates', 2)),...                 % product of two terms
        % addterm(diffOp([3,0],'stateind',1, 'nstates', 2), term('fHandle',@(u,v) tanh(u+v))),...              % sum of two terms
    };

custom_remove_f = {}; %{@(tag) all(tag(Uobj.nstates+1:Uobj.nstates+Uobj.ndims-1))};  % remove all cross derivatives
custom_remove_t = {}; %[1 0 0 1 0 0; 0 1 0 0 1 0];                              % remove tags for divergence terms

lib = get_lib(Uobj,polys,trigs,x_diffs,custom_add,custom_remove_f,custom_remove_t);

%%% set testfcn 
phifun = 'pp';
tau = 10^-10; tauhat = 1.5;
tf_param = {[tau tauhat max(x_diffs)]};
tf_args = {'phifuns',phifun,'meth','FFT','param',tf_param,'subinds',-4};
tf = testfcn(Uobj,tf_args{:});

%%% scale data
Uobj.set_scales([],'lib',lib,'tf',tf);
tf = testfcn(Uobj,tf_args{:}); % must recompute test function weights

%%% define wsindy_model
if isequal(class(lhs),'cell')
    lhs = [1,0,1];
end
WS = wsindy_model(Uobj,lib,tf,'lhsterms',lhs);

%% solve for coefficients

%%% get coefficient scale vector
Mscale = arrayfun(@(L)L.get_scales(Uobj.scales),WS.lib(:),'un',0);
lhs_scales = cellfun(@(t)t.get_scale(Uobj.scales),WS.lhsterms(:),'un',0);
Mscale = cellfun(@(M,L)M/L,Mscale,lhs_scales,'un',0);
Mscale_W = cell2mat(Mscale);

%%% optimization parameters
lambdas = 10.^linspace(-4,0,50);
threshold_scheme = 2;

tic;
[WS,loss_wsindy,its,G,b] = WS_opt().MSTLS_0(WS,'lambdas',lambdas,'M_diag',Mscale,'toggle_jointthresh',threshold_scheme,'alpha',[]);
runtime = toc;

%%% non-dimensionalized coefficients
W_nd = cellfun(@(w,m)w./m,WS.reshape_w,Mscale,'un',0); 

%% Diagnose

fprintf('\ndata dims=');fprintf('%u ',Uobj.dims);

%%% display model
Str_mod = WS.disp_mod;
for j=1:WS.numeq
    fprintf('----------Eq %i----------\n',j)
    fprintf('%s=\n',WS.lhsterms{j}.get_str)
    cellfun(@(s)fprintf('%s\n',s),Str_mod{j})
end

fprintf('\n')
resids = WS.res('sepcomp');
w_support = WS.get_supp;

cellfun(@(r)disp(['rel resid=',num2str(norm(r))]),resids);
cellfun(@(s)disp(['sparsity=',num2str(length(s))]),w_support)
fprintf('\ntf rads=');
cellfun(@(tf)fprintf('%u ',tf.rads),WS.tf{1});
fprintf('\nsize G =')
cellfun(@(G) fprintf('%d ',size(G)), WS.G)
fprintf('\ncond G =')
fprintf('%1.1e ', cellfun(@(G)cond(G),WS.G));

if exist('true_nz_weights','var')
    w_true = inject_true_weights(WS,true_nz_weights);
    Tps = tpscore(WS.weights,w_true);
    fprintf('\nTPR=%1.2f',Tps)
    E2 = norm(w_true-WS.weights)/norm(w_true);
    fprintf('\nCoeff err=%1.2e',E2)
end
fprintf('\nruntime=%1.2f(s)',runtime)

%%% display data
figure(1);clf;
n=1; % for 2D space problems 
subplot(2,1,1)
imagesc(Uobj.Uobs{n}(:,:,1))
title('observed')
subplot(2,1,2)
imagesc(U_exact{n}(:,:,1))
title('ground truth')

%%% plot MSTLS loss
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

figure(3);clf
%%% plot residual
for j=1:WS.numeq
    subplot(WS.numeq,1,j)
    plot([WS.bs{1}{j} WS.Gs{1}{j}*W_nd{j}])
    legend('b','G*w')
    title(['||G*w-b||/||b||=',num2str(norm(WS.Gs{1}{j}*W_nd{j}-WS.bs{1}{j})/norm(WS.bs{1}{j}))])
end

%% functions

function lib = get_lib(Uobj,polys,trigs,x_diffs,custom_add,custom_remove_f,custom_remove_t)    
    nstates = Uobj.nstates;
    ndims = Uobj.ndims;
    
    tags = get_tags(polys,trigs,nstates);
    lib = library('nstates',nstates);
    
    diff_tags = get_tags(x_diffs,[],ndims);
    diff_tags = diff_tags(diff_tags(:,end)==0,:);
    for j=1:size(tags,1)
        for i=1:size(diff_tags,1)
            if all([~and(sum(diff_tags(i,:))>0,...
                    isequal(tags(j,:),zeros(1,nstates))),...
                    ~cellfun(@(b)b([tags(j,:) diff_tags(i,:)]),custom_remove_f),...
                    ~ismember_rows([tags(j,:) diff_tags(i,:)],custom_remove_t)])
                lib.add_terms(term('ftag',tags(j,:),'linOp',diff_tags(i,:)));
            end
        end
    end
    lib.add_terms(custom_add);
end
