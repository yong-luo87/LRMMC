function [Z, best_mAP, best_mAUC] = MC_Simplex(W, Y_gt, option, para)
% function [best_mAP, best_mAUC, Z] = ...
%     MC_Simplex(W, Y_gt, lambda, red_factor, mu_thresh, shldNrm, g, bin_type, feat_sizes)
% -------------------------------------------------------------------------
% Matrix Completion for Multi-label Classification with positive constraint
% -------------------------------------------------------------------------

% train_nrm = (train)*spdiags(1./(max(abs(train),[],1))',0,size(train,2),size(train,2));
% L1 normalization
% train_nrm= spdiags(1./(sum(train,2)),0,size(train,1),size(train,1))*(train);

numclasses = size(Y_gt,1);
trainsize = size(W,2) - size(Y_gt,2);

Y_data = W(1:numclasses,:);
X_data = W(numclasses+1:end,:); featsize = size(X_data,1);
X_data = (X_data)*spdiags(1./(sum(X_data))',0,size(X_data,2),size(X_data,2));
W(numclasses+1:end,:) = X_data;

omega_y = find(~isnan(Y_data));
omega_x = find(~isnan(X_data));
omega_x_len = length(omega_x);
omega_y_len = length(omega_y);

best_mAP = 0;
best_mAUC = 0;

W_nan = W;
W(isnan(W)) = 0;
[U, Sigma, V] = svds([W; ones(1,size(W,2))],1);

% -------------------------------------------------------------------------
% Set the step size 'tau' and the candidate set of the trade-off parameter 'mu' for FPC
% -------------------------------------------------------------------------
mus = Sigma(1)*para.red_factor;
while(mus(end) > para.mu_thresh)
	mus = [mus, mus(end)*para.red_factor];
end
tau_z = min(3.8*omega_y_len/(para.lambda), omega_x_len);

% -------------------------------------------------------------------------
% Initialize with rank-1 projection of W with zero on unknown entries
% -------------------------------------------------------------------------
Z = sparse(U*Sigma*V'); clear U V Sigma

tmp = Z(1:end-1,:);
tmp(isnan(W)) = 0;
Z(1:end-1,:) = tmp;

Y = Z(1:numclasses,:);
X = Z(numclasses+1:end-1,:);

iter = 0;
last_err = Inf;
% -------------------------------------------------------------------------
% Main loop of the Fixed Point Continuation algorithm for solving MC-Simplex
% -------------------------------------------------------------------------
for mu = mus 
	err_diff = Inf;
	while (err_diff > para.seuildiffobjMC)
        % ----------------------------------------------------
        % Compute the gradient step
        % ----------------------------------------------------
        if option.shldNrm == 3
            % Use chi-square dist for features
            gX = 1/omega_x_len * (X.^2 +2*X.*X_data -3*X_data.^2)./((X+X_data).^2);
            gX(isnan(gX)) = 0;
        else        
            % Use LS error for features
            gX = 1/omega_x_len * (X-X_data);
        end
        % Use generalized log loss for labels
        if option.bin_type == 1             % -1/1 labels
    		gY = para.lambda/omega_y_len * (-Y_data./(1+exp(para.rou*Y_data.*Y)));
        else                                % 0/1 labels
        	gY = para.lambda/omega_y_len * ((-Y_data)./(1+exp(para.rou*Y_data.*(Y-1))));    
        end
        
        gZ = [gY;gX]; clear gY gX
		gZ(isnan(W_nan)) = 0;       % Zero out gradient for unknown entries
		A = Z - tau_z*[gZ;zeros(1,size(W,2))];

        % ----------------------------------------------------
        % Shrinkage step
        % ----------------------------------------------------
		[U, Sigma, V] = svd(full(A),'econ');
        SigmaD = diag(Sigma); r = sum(SigmaD > tau_z*mu);
        if r == 0
            % 'div...'
            break
        end
		Z = U*diag( max(0,diag(Sigma)-tau_z*mu) ) * V'; clear U Sigma V SigmaD

        % ----------------------------------------------------
        % Project Z back onto feasible region
        % ----------------------------------------------------
		Z(end,:) = 1;
        if option.simPrj
            X_proj = Z(numclasses+1:end-1,:);
            projsimplex(X_proj, featsize);
            Z(numclasses+1:end-1,:) = X_proj;
        end
        % Z(1:numclasses,end-numclasses+1:end) = 2*eye(numclasses)-1;
        
		% ----------------------------------------------------
		% Calculate obj value
        % ----------------------------------------------------
		Y = Z(1:numclasses,:);
		X = Z(numclasses+1:end-1,:);
        if option.bin_type ==1          % -1/1 labels
			label_err = (1/para.rou)*log(1+exp(-para.rou*Y_data.*Y));       % Logistic error for labels
        else                            % 0/1 labels
			label_err = (1/para.rou)*log(1+exp(-para.rou*Y_data.*(Y-1)));   % Logistic error for labels
        end
		if option.shldNrm == 3
            % Use chi-square dist for features
            feature_err = ((X_data - X).^2) ./(X_data + X); % LS error
            feature_err(isnan(feature_err)) = 0;
		else        
            % Use LS error for features
            feature_err = (X_data - X).^2;                  % LS error
        end
        nuc_norm = sum(svd(Z));                             % Nuclear norm
		iter = iter + 1;
		cur_err(iter) = mu*nuc_norm + para.lambda/omega_y_len *sum(label_err(omega_y)) ...
            + 1/omega_x_len *sum(feature_err(omega_x));
		
        % ----------------------------------------------------
        % Check stopping criteria
        % ----------------------------------------------------
        if iter > 20
    		err_diff = min(abs(cur_err(end) - cur_err(end-10:end-1))./cur_err(end-10:end-1));
        else
            err_diff = abs(cur_err(iter) - last_err)/last_err;
        end
		last_err = cur_err(iter);
		fprintf('mu = %d, iters = %d, obj = %f, err_diff = %f\n', mu, iter, last_err,err_diff)
    end
    % ---------------------------------------------------------------------
    % If an optimal solution has been found, update the accuracy
    % ---------------------------------------------------------------------     
	if (r ~= 0)
        AP = zeros(numclasses, 1); AUC = zeros(numclasses, 1);
        for ic = 1:numclasses
            AP(ic) = evaluateAP_VOC(Y_gt(ic,:), Y(ic,trainsize+1:end));
            AUC(ic) = evaluateAUC_ROC(Y_gt(ic,:), Y(ic,trainsize+1:end), 1, -1);
        end
        tst_mAP = mean(AP); tst_mAUC = mean(AUC);
        if tst_mAP > best_mAP
            best_mAP = tst_mAP;
            fprintf('\nMCSimplex: mu = %d, lambda = %f, mAP = %f\n', mu, para.lambda, tst_mAP);
            if tst_mAUC <= best_mAUC
                fprintf('\n');
            end
        end
        if tst_mAUC > best_mAUC
            best_mAUC = tst_mAUC;
            if tst_mAP < best_mAP
                fprintf('\n');
            end
            fprintf('MCSimplex: mu = %d, lambda = %f, mAUC = %f\n\n', mu, para.lambda, tst_mAUC);
        end
        % % get avg. AUROC
        % [tst_auroc] = auroc2(Y(:,trainsize+1:end), Y_gt, Y(:,1:trainsize));
        % [old_auroc] = auroc(Y(:,trainsize+1:end), Y_gt);
        % if tst_auroc > best_auroc
        %     best_auroc = tst_auroc;
        %     fprintf('MCSimplex: mu=%d, lambda = %f, AUROC = %f,AUROC prev = %f\n', mu, lambda, tst_auroc,old_auroc);
        % end
    end
end

end

