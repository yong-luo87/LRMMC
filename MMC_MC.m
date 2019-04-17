function [B_new, obj_new]= MMC_MC(B, P, A, set, option, para)
% -------------------------------------------------------------------------
% Update B when P is fixed
% mu*||B||_* + ||B*P-A||_F^2 + eta*||P||_F^2
% -------------------------------------------------------------------------

[N, D] = size(A);
% -------------------------------------------------------------------------
% Set the continuation sequence
% -------------------------------------------------------------------------
[U, Sigma, V] = svds(double(B), 1);
para.mu_threshold = para.mu;
mus = Sigma(1)*para.red_factor;
while(mus(end) > para.mu_threshold)
    mus = [mus, mus(end)*para.red_factor];
end

% -------------------------------------------------------------------------
% Set the step size
% -------------------------------------------------------------------------
tau = N / (2.0*max(eig(P*P'))); % tau belongs to (0, 1/lambda_max(P*P'))

% -------------------------------------------------------------------------
% Initialize with rank-1 projection of old B
% -------------------------------------------------------------------------
B = sparse(U*Sigma*V'); clear U V Sigma

% -------------------------------------------------------------------------
% FPC to optimize B for a fixed P
% -------------------------------------------------------------------------
iter = 0;
last_err = inf;
for mu = mus
    err_diff = Inf;
    while(err_diff > para.seuildiffobjIterIn)
        % ---------------------------------------------------------
        % Gradient descent
        % ---------------------------------------------------------
        M = B - (2.0/N) * tau * (B*P - A) * P';
        
        % ---------------------------------------------------------
        % Shirinkage step
        % ---------------------------------------------------------
        [U, Sigma, V] = svd(full(M), 'econ');
        SigmaD = diag(Sigma); r = sum(SigmaD > tau*mu);
        if r == 0
            % fprintf('r=0......\n');
            break;
        end
        Sigma_shrink = diag(max(0, diag(Sigma) - tau*mu));
        B = U * Sigma_shrink * V'; clear U Sigma SigmaD Sigma_shrink V
        
        % ---------------------------------------------------------
        % calculate the value of equation: mu*||B||_* + ||B*P-A||_F^2
        % ---------------------------------------------------------
        nuc_norm = sum(svd(B));
        penalty_term = (1.0/N) * norm(B*P-A, 'fro')^2;
        iter = iter + 1;
        cur_err(iter) = mu*nuc_norm + penalty_term;
        
        % ---------------------------------------------------------
        % Check stopping criteria
        % ---------------------------------------------------------
        err_diff = abs(cur_err(iter) - last_err) / last_err;
        last_err = cur_err(iter);
        if option.verbose >= 3
            fprintf('mu = %d, iter = %d, cur_err = %d, err_diff = %d\n', mu, iter, last_err, err_diff);
        end
    end
end
obj_new = cur_err(iter);
B_new = B;

end

