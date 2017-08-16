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


% Regularization terms
reg_term = (lambda/(2*m)) * (theta(2:end)' * theta(2:end))
gra_reg_tem = (lambda/m) * theta(2:end)

% Hipotesis
h_x = sigmoid(X * theta)

% Cost for positive a negative labels
cost_one = -y' * log(h_x)
cost_zero =  (1 - y)' * log(1 - h_x)

% Total Cost
J = 1/m * (cost_one - cost_zero)  + reg_term

%Gradient Descent
grad_one = 1/m * ((h_x - y)'*X(:,1))'
grad_else = 1/m * ((h_x - y)'*X(:,2:end))' + gra_reg_tem
grad = [ grad_one;grad_else]





% =============================================================

end
