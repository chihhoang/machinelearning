function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
feature_num = size(X,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % iterative solution
    % delta = zeros(1, feature_num);

    % for j = 1:feature_num
    %     for i = 1:m
    %         delta(j) = delta(j) + (X(i,:)*theta - y(i)) * X(i, j);
    %     end
    % end

    % % update thetas
    % for j = 1:feature_num
    %     theta(j) = theta(j) - alpha/m*delta(j);
    % end

    % vectorization solution
    delta = X' * (X*theta - y);
    theta = theta - alpha/m*delta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    % printf('cost: %d\n', J_history(iter));
end

end
