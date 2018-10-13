function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% X has size 5000 x 400, need to add a 1s column for a1
a1 = [ones(m, 1) X];

% Compute values in the hidden layer
a2 = sigmoid(a1 * Theta1');

% Add a 1s column to a2
a2 = [ones(size(a2), 1) a2];

% Compute values in the output layer
a3 = sigmoid(a2 * Theta2');

% max function returns 2 values
% The first one is max values of each row, which we don't need
% The second one is that first index of the max value, which we want
% Save our outputs in p
[max_values, p] = max(a3, [], 2);

% =========================================================================


end
