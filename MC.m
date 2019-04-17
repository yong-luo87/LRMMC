function [Z, obj] = MC(W, trainLabelsU, testLabels, set, option, para)
% -------------------------------------------------------------------------
% Matrix Completion implementation
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Complete the matrix via FPC (Fixed Point Continuation)
% -------------------------------------------------------------------------
if option.completeOnlyU
    switch option.alg
        case 'MC_1'
            [Z, obj] = MC_1(W, trainLabelsU, set, option, para);
        case 'MC_Pos'
            [Z, obj] = MC_Pos(W, trainLabelsU, option, para);
        case 'MC_Simplex'
            [Z, obj] = MC_Simplex(W, trainLabelsU, option, para);
    end
end
if option.completeOnlyT
    switch option.alg
        case 'MC_1'
            [Z, obj] = MC_1(W, testLabels, set, option, para);
        case 'MC_Pos'
            [Z, obj] = MC_Pos(W, testLabels, option, para);
        case 'MC_Simplex'
            [Z, obj] = MC_Simplex(W, testLabels, option, para);
    end
end
if ~option.completeOnlyU && ~option.completeOnlyT
    switch option.alg
        case 'MC_1'
            [Z, obj] = MC_1(W, ([trainLabelsU testLabels]), set, option, para);
        case 'MC_Pos'
            [Z, obj] = MC_Pos(W, ([trainLabelsU testLabels]), option, para);
        case 'MC_Simplex'
            [Z, obj] = MC_Simplex(W, ([trainLabelsU testLabels]), option, para);
    end
end

end

