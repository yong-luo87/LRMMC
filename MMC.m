function outB = MMC(featureData, set, option, para)
% -------------------------------------------------------------------------
% Multiview Matrix Completion, mu*||B||_* + ||B*P-A||_F^2 + eta*||P||_F^2
% -------------------------------------------------------------------------

A = featureData; 
[N, D] = size(A);
clear featureData
% -------------------------------------------------------------------------
% Initialize B with the DSE result
% -------------------------------------------------------------------------
% [B, ~] = eigs(double(A*A'), para.nbK);

% -------------------------------------------------------------------------
% Initialize B with a rank-1 approximation of A
% -------------------------------------------------------------------------
% [U, Sigma, V]  = svds(double(A), 1);
% B = U * Sigma * V';
% B = B(:, 1:para.nbK); clear U Sigma V

% -------------------------------------------------------------------------
% Initialize B with a random matrix
% -------------------------------------------------------------------------
rand('seed', 123);
B = rand(set.nbTrain+set.nbTest, para.nbK);

% -------------------------------------------------------------------------
% Learn the optimal pattern matrix B for MMC
% -------------------------------------------------------------------------
iter = 0;
last_err = Inf;
err_diff = Inf;
while (err_diff > para.seuildiffobjIterOut || isnan(err_diff))
    % ----------------------------------------------------------
    % Update the projection matrix P for fixed pattern matrix B
    % ----------------------------------------------------------
    P = (1.0/N*(B'*B) + para.eta*eye(para.nbK)) \ (1.0/N*(B'*A));
    
    % ----------------------------------------------------------
    % Update the pattern matrix B using FPC for fixed projection matrix P
    % ----------------------------------------------------------
    iter = iter + 1;
    [B, obj_FPC] = MMC_MC(B, P, A, set, option, para);
    cur_err(iter) = obj_FPC + para.eta * norm(P,'fro')^2; clear obj_FPC
    
    % ----------------------------------------------------------
    % Check stopping criteria
    % ----------------------------------------------------------
    err_diff = abs(cur_err(iter) - last_err ) / last_err;
    if option.verbose >= 3
        fprintf('\niter_out = %d, cur_err = %d, last_err = %d err_diff = %d\n\n', ...
            iter, cur_err(iter), last_err, err_diff);
    else
        fprintf('iter_out = %d, cur_err = %d, last_err = %d err_diff = %d\n', ...
            iter, cur_err(iter), last_err, err_diff);
    end
    last_err = cur_err(iter);
end
outB = B;

end

