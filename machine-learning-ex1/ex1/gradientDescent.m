function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	H1=0;
    H2=0;
	for i=1:m
	H1=H1+((X(i,:)*theta )-y(i))*X(i,1);
	H2=H2+((X(i,:)*theta )-y(i))*X(i,2);
	i+=1;
	end

    temp0=theta(1)-(alpha/m)*H1;%the c is wrtong watch the video to refigure it ourt
	them1=theta(2)-(alpha/m)*H2;
	theta(1)=temp0;
	theta(2)=them1;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
