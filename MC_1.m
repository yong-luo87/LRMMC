function [Zs, last_errset, best_mAP, best_mAUC] = MC_1(W, Y_gt, set, option, para)
% function [best_mAP, best_mAUC, Z] = MC_1(W, Y_gt, lambda, red_factor, mu_thresh)
% -------------------------------------------------------------------------
% Matrix Completion for Multi-label Classification
% -------------------------------------------------------------------------

lambda = para.lambda;

numclasses = size(Y_gt,1);
trainsize = size(W,2) - size(Y_gt,2);

Y_data = W(1:numclasses,:);
X_data = W(numclasses+1:end,:);

omega_y = find(~isnan(Y_data));
omega_x = find(~isnan(X_data));
omega_y_len = length(omega_y);
omega_x_len = length(omega_x);

best_mAP = 0;
best_mAUC = 0;

W_nan = W;
W(isnan(W)) = 0;
[U, Sigma, V] = svds([W; ones(1,size(W,2))],1);

% -------------------------------------------------------------------------
% Set the step size 'tau' and the candidate set of the trade-off parameter 'mu' for FPC
% -------------------------------------------------------------------------
para.mu_thresh_MC = para.mu_MC;
mus = Sigma(1)*para.red_factor;
while(mus(end) > para.mu_thresh_MC)
	mus = [mus, mus(end)*para.red_factor];
end
tau_z = min(3.8*omega_y_len/(lambda), omega_x_len);

% -------------------------------------------------------------------------
% Initialize with rank-1 projection of W with zero on unknown entries
% -------------------------------------------------------------------------
Z = sparse(U*Sigma*V'); clear U V Sigma

tmp = Z(1:end-1,:);
tmp(isnan(W)) = 0;
Z(1:end-1,:) = tmp; clear tmp

Y = Z(1:numclasses,:);
X = Z(numclasses+1:end-1,:);

iter = 0;
last_err = Inf;
% -------------------------------------------------------------------------
% Main loop of the Fixed Point Continuation algorithm for solving MC-1
% -------------------------------------------------------------------------
for mu = mus
    err_diff = Inf;
    while (err_diff > para.seuildiffobjMC)
        % ----------------------------------------------------
        % Compute the gradient step
        % ----------------------------------------------------
        % Use LS error for features
        gX = 1/omega_x_len * (X-X_data);
        % Use generalized log loss for labels
        para.rou = 1;
        gY = lambda/omega_y_len * (-Y_data./(1+exp(para.rou*Y_data.*Y)));
        
        gZ = [gY;gX]; clear gY gX
        gZ(isnan(W_nan)) = 0; % Zero out gradient for unknown entries
        A = Z - tau_z*[gZ;zeros(1,size(W,2))];
        
        % ----------------------------------------------------
        % Shrinkage step
        % ----------------------------------------------------
        A(isnan(A)) = 0; A(isinf(A)) = 0;
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
        % Z(numclasses+1:end-1,:) = max(0,Z(numclasses+1:end-1,:));
        Z(end,:) = 1;
		
		% ----------------------------------------------------
		% Calculate obj value
        % ----------------------------------------------------
		Y = Z(1:numclasses,:);
        X = Z(numclasses+1:end-1,:);
        feature_err = 0.5*(X_data - X).^2;          % LS error for features
        label_err = log(1+exp(-Y_data.*Y));         % Logistic error for labels
		
        Z(isnan(Z)) = 0; Z(isinf(Z)) = 0;
		nuc_norm = sum(svd(Z));                     % Nuclear norm
		iter = iter + 1;
        % cur_err(iter) = mu*nuc_norm + 1/omega_x_len * sum(feature_err(omega_x)) ...
        %     + lambda/omega_y1_len * sum(label_err(omega_y1)) ...
        %     + para.gamma/omega_y2_len * sum(predict_var(omega_y2));
        cur_errset.N(iter) = mu*nuc_norm;
        cur_errset.X(iter) = 1.0/omega_x_len * sum(feature_err(omega_x));
        cur_errset.Y(iter) = 1.0/omega_y_len * sum(label_err(omega_y));
        cur_err(iter) = cur_errset.N(iter) + cur_errset.X(iter) + lambda*cur_errset.Y(iter);
        clear feature_err label_err
        
        % ----------------------------------------------------
        % Check stopping criteria
        % ----------------------------------------------------
        err_diff = abs(cur_err(iter) - last_err)/last_err;
        last_err = cur_err(iter);
        if option.verbose >= 4
            fprintf('\nmu = %d, iter = %d, obj = %f, err_diff = %f', mu, iter, last_err, err_diff);
        end
    end
    % ---------------------------------------------------------------------
    % If an optimal solution has been found, update the accuracy
    % ---------------------------------------------------------------------
    if (r ~= 0)
        AP = zeros(numclasses, 1); AUC = zeros(numclasses, 1);
        for ic = 1:numclasses
            % AP(ic) = evaluateAP_VOC(Y_gt(ic,:), Y2(ic,:));
            % AUC(ic) = evaluateAUC_ROC(Y_gt(ic,:), Y2(ic,:), 1, -1);
            valid_index = (set.nbU+1):(set.nbU+para.size_valid);
            AP(ic) = evaluateAP_VOC(Y_gt(ic, valid_index), Z(ic, set.nbL+valid_index));
            AUC(ic) = evaluateAUC_ROC(Y_gt(ic, valid_index), Z(ic, set.nbL+valid_index), 1, -1);
        end
        tst_mAP = mean(AP); tst_mAUC = mean(AUC);
        if tst_mAP > best_mAP
            best_mAP = tst_mAP;
            if option.mode == 1
                clear Zs; Zs = Z;
            end
            if option.verbose >= 3
                fprintf('\nMC1: mu = %d, lambda = %f, mAP = %f\n', mu, lambda, tst_mAP);
                if tst_mAUC <= best_mAUC
                    fprintf('\n');
                end
            end
        end
        if tst_mAUC > best_mAUC
            best_mAUC = tst_mAUC;
            if option.verbose >= 3
                if tst_mAP < best_mAP
                    fprintf('\n');
                end
                fprintf('MC1: mu = %d, lambda = %f, mAUC = %f\n\n', mu, lambda, tst_mAUC);
            end
        end
        % % ----------------------------------------------------
        % % Get avg. AUROC
        % % ----------------------------------------------------
        % [tst_auroc] = auroc2(Y(:,trainsize+1:end), Y_gt, Y(:,1:trainsize));
        % [old_auroc] = auroc(Y(:,trainsize+1:end), Y_gt);
        % if tst_auroc > best_auroc
        %     best_auroc = tst_auroc;
        %     fprintf('MC1: mu=%d, lambda = %f, AUROC = %f,AUROC prev = %f\n', mu, lambda, tst_auroc,old_auroc);
        % end
    end
end
last_errset.N = cur_errset.N(iter);
last_errset.X = cur_errset.X(iter);
last_errset.Y = cur_errset.Y(iter);
if option.mode == 0
    Zs = Z;
end

end

