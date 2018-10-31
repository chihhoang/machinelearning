function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01 0.03 0.1 0.3 1 3 10 30];
min_error = 999999;

% Use the result (from the below code) directly 
% to avoid long training during submission
% C = 1;
% sigma = 0.1;

for i = 1:length(steps)
    C_ = steps(i);
    for j = 1:length(steps)
        sigma_ = steps(j);
        % Train the SVM with gaussian kernel
        model = svmTrain(X, y, C_, @(x1, x2) gaussianKernel(x1, x2, sigma_));
        % Predict the labels on the cross validation set
        predictions = svmPredict(model, Xval);
        current_error = mean(double(predictions ~= yval));
        % If the current error is less than the minimum error so far
        % update C and sigma
        if current_error < min_error
            min_error = current_error;
            C = C_;
            sigma = sigma_;
        endif
    end
end

% =========================================================================

end
