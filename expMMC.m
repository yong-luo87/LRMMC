function [APu, APt, APl, PERFu, PERFt, PERFl] = ...
    expMMC(singleTrainFea, singleTestFea, trainLabels, testLabels, set, option, para)
% -------------------------------------------------------------------------
% Experiment of MMC for Multi-label Classification
% -------------------------------------------------------------------------

singleTrainFeaL = cell(set.nbV,1); singleTrainFeaU = cell(set.nbV,1);
% -------------------------------------------------------------------------
% Create a label/unlabeled split
% -------------------------------------------------------------------------
[indexL, indexU, trainLabelsL, trainLabelsU] = dataSplit(trainLabels, set, option, para);
set.nbL = length(indexL); set.nbU = length(indexU);
for v = 1:set.nbV
    singleTrainFeaL{v} = singleTrainFea{v}(indexL, :);
    singleTrainFeaU{v} = singleTrainFea{v}(indexU, :);
end

% -------------------------------------------------------------------------
% Concatenate all the features
% -------------------------------------------------------------------------
trainFeaL = []; trainFeaU = []; testFea = [];
for v = 1:set.nbV
    trainFeaL = [trainFeaL singleTrainFeaL{v}];
    trainFeaU = [trainFeaU singleTrainFeaU{v}];
    testFea = [testFea singleTestFea{v}];
end
clear singleTrainFeaL singleTrainFeaU singleTestFea

% -------------------------------------------------------------------------
% Combine the different views using MMC
% -------------------------------------------------------------------------
allFea = [trainFeaL; trainFeaU; testFea]; clear trainFeaL trainFeaU testFea
newAllFea = MMC(allFea, set, option, para); clear allFea
trainFeaL = newAllFea(1:set.nbL, :);
trainFeaU = newAllFea(set.nbL+1:set.nbTrain, :);
testFea = newAllFea(set.nbTrain+1:end, :); clear newAllFea

% -------------------------------------------------------------------------
% Construct the un-completed matrix
% -------------------------------------------------------------------------
if option.completeOnlyU
    W = [trainLabelsL' nan(size(trainLabelsU))'; trainFeaL' trainFeaU'];
end
if option.completeOnlyT
    W = [trainLabelsL' nan(size(testLabels))'; trainFeaL' testFea'];
end
if ~option.completeOnlyU && ~option.completeOnlyT
    W = [trainLabelsL' nan(size([trainLabelsU; testLabels]))'; trainFeaL' ([trainFeaU; testFea])'];
end

% -------------------------------------------------------------------------
% Predict with matrix completion
% -------------------------------------------------------------------------
[Z, err] = MC(W, trainLabelsU', testLabels', set, option, para);
Y = Z(1:set.nbP,:); ERR.N = err.N; ERR.X = err.X; ERR.Y = err.Y; clear Z err
probLbls = Y(:,1:set.nbL)';
if option.completeOnlyU
    probUnls = Y(:,set.nbL+1:end)';
end
if option.completeOnlyT
    probTest = Y(:,set.nbL+1:end)';
end
if ~option.completeOnlyU && ~option.completeOnlyT
    probUnls = Y(:,set.nbL+1:set.nbTrain)';
    probTest = Y(:,set.nbTrain+1:end)';
end

% -------------------------------------------------------------------------
% Evaluate the performance
% -------------------------------------------------------------------------
fprintf('Evaluating ... ');
probLbls = 1./(1.0+exp(-probLbls));
[APl, PERFl] = evaluate(probLbls, trainLabelsL, 0.5);
if option.completeOnlyU
    probUnls = 1./(1.0+exp(-probUnls));
    [APu, PERFu] = evaluate(probUnls, trainLabelsU, 0.5);
    APt = []; PERFt = nan;
end
if option.completeOnlyT
    probTest = 1./(1.0+exp(-probTest));
    [APt, PERFt] = evaluate(probTest, testLabels, 0.5);
    APu = []; PERFu = nan;
end
if ~option.completeOnlyU && ~option.completeOnlyT
    probUnls = 1./(1.0+exp(-probUnls));
    probTest = 1./(1.0+exp(-probTest));
    [APu, PERFu] = evaluate(probUnls, trainLabelsU, 0.5);
    [APt, PERFt] = evaluate(probTest, testLabels, 0.5);
end
disp('Finished!');

end

