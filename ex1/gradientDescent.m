
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % # of parameters
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
%  V = zeros(n, 1);
%  for i = 1:m

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
%        V = V + (X(i, :) * theta - y(i)) * (X(i, :)');
%    end
%    theta = theta - (alpha * V / m);

    % ============================================================
	min1 = 0
	min2 = 0
	for i = 1:m
		min1 = min1 + (theta(1) + theta(2)*X(i,2) - y(i))*X(i,1);
		min2 = min2 + (theta(1) + theta(2)*X(i,2) - y(i))*X(i,2);
		end

	theta(1) = theta(1) - (alpha/m)*min1;
	theta(2) = theta(2) - (alpha/m)*min2;
	theta
	


    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
