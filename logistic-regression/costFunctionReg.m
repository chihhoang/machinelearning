function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X * theta);
% J = costFunction(theta, X, y)(1) + lambda/;

% TODO: Analyze dimension and finish gradient
J = 1 / m * (-y'*log(h) - (1 - y')*log(1 - h)) + lambda/2/m*theta.^2;
disp(size(h));
disp(size(J));
first_x = X(:,1);
disp(size(X));
disp(size(first_x'));

h0 = sigmoid(X * theta);
grad(1) = grad(1) + 1/m*first_x'*(h(1,:) - y(1,:));

grad() = grad + 1/m*X(2:m,:)'*(h(2:m,:) - y(2:m)) + lambda/m*theta;





% =============================================================

end
