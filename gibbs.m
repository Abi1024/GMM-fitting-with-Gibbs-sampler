%Abiyaz Chowdhury, Stevens Institute of Technology PhD admissions
%assignment for Prof. Perez-Cruz on fitting a GMM with Gibbs sampling
%Jan 2017
clc;
clear all;
close all;

%GMM actual component parameters. if the components have similar means 
%with high variance, the algorithm is unlikely to perform well.
p = [0.5 0.3 0.2];
mu = [0 10 20];
var = [1 2 3];
p_actual = p;
mu_actual = mu;
var_actual = var;

%size of observed data, and number of iterations. try increasing either for
%better results, though at the expense of longer running time
num_data = 200;
iterations = 10000;

%generating the data
M = [];
for i = 1:3
   M = [M (randn(1,floor(num_data*p(i)))*sqrt(var(i))+mu(i))];
end

%storage variables
responsibilities = zeros(1,3); %responsibilities for each data point
group = zeros(1,size(M,2)); %store current assignment of data point to component
count = zeros(1,3); %store number of points currently assigned to a component
running_sum = zeros(1,3); %store sum of data in a given component
running_sum_squares = zeros(1,3); %store sum of squares of data in a given component

%initialize algorithm by randomly assigning data to components
for i = 1:size(M,2)
    group(i) = randi([1 3],1,1);
    count(group(i)) = count(group(i)) + 1;
    running_sum(group(i)) = running_sum(group(i)) + M(i);
    running_sum_squares(group(i)) = running_sum_squares(group(i)) + M(i)*M(i);
end

%GMM fitting with Gibbs sampling
for x = 1:iterations
    id = mod(x,size(M,2))+1;    %iterate through data in random order
    %remove the data point from the cluster
    if (count(group(id)) > 10)
        count(group(id)) = count(group(id))-1; 
        running_sum(group(id)) = running_sum(group(id))-M(id);
        running_sum_squares(group(id)) = running_sum_squares(group(id)) - M(id)*M(id);
        %update the GMM parameters
        for j = 1:3  
            mu(j) = running_sum(j)/count(j);
            p(j) = count(j)/size(M,2);
            var(j) = running_sum_squares(j)/count(j) - mu(j)*mu(j);
            responsibilities(j) = normpdf(M(id),mu(j),sqrt(var(j)))*p(j);   
        end
        %sample a value of z, given the existing assignments and parameters
        responsibilities = responsibilities/sum(responsibilities);
        [toto , draw_samples] = histc(rand(1 , 1), [0 cumsum(responsibilities)]);
        %use the sampled value of z to assign the data point to a new component
        group(id) = draw_samples;
        count(group(id)) = count(group(id))+1;
        running_sum(group(id)) = running_sum(group(id))+M(id);
        running_sum_squares(group(id)) = running_sum_squares(group(id)) + M(id)*M(id); 
        if (mod(x,1000) == 0)
            fprintf('Iterations: %d\n',x);
            disp(strrep(['p: (' sprintf(' %f,', p) ')'], ',)', ')'))
            disp(strrep(['mean: (' sprintf(' %f,', mu) ')'], ',)', ')'))
            disp(strrep(['var: (' sprintf(' %f,', var) ')'], ',)', ')'))
            fprintf('\n');
        end
    end
end

%display solution
fprintf('\nFinal result:\n')
disp(strrep(['p: (' sprintf(' %f,', p) ')'], ',)', ')'))
disp(strrep(['mean: (' sprintf(' %f,', mu) ')'], ',)', ')'))
disp(strrep(['var: (' sprintf(' %f,', var) ')'], ',)', ')'))

fprintf('\n\nActual parameters:\n')
disp(strrep(['p: (' sprintf(' %f,', p_actual) ')'], ',)', ')'))
disp(strrep(['mean: (' sprintf(' %f,', mu_actual) ')'], ',)', ')'))
disp(strrep(['var: (' sprintf(' %f,', var_actual) ')'], ',)', ')'))
