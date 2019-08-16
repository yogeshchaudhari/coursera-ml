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


sigma = X * theta;
prediction = sigmoid(sigma);
errors = prediction - y;
gradientRegulator = (lambda/m) * theta';
grad = (1/m * (errors' * X)) + gradientRegulator;

firstGradient = (1/m * sum(errors * X(1)));

grad(1) = firstGradient;

pos = -y' * log(prediction);
neg = (1-y)' * log(1 - prediction);
theta(1) = 0;
costRegulator = (lambda/(2*m)) * sum(theta.^2);
J = 1/m * (pos - neg) + costRegulator;



% =============================================================

end
