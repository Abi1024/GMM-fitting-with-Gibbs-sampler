%Abiyaz Chowdhury, Stevens Institute of Technology PhD admissions
%assignment for Prof. Perez-Cruz on Gibbs sampling
%Jan 2017
clc;
clear all;
close all

%GMM actual parameters
p = [0.5 0.3 0.2];
mu = [0 50 100];
var = [1 4 1];

num_data = 200;
iterations = 10;

%generating the data
M = [];
for i = 1:3
   M = [M (randn(1,floor(num_data*p(i)))*sqrt(var(i))+mu(i))];
end

M

%random initial parameters
responsibilities = zeros(1,3);
group = zeros(1,size(M,2));
count = zeros(1,3);
running_sum = zeros(1,3);
running_sum_squares = zeros(1,3);
min_group = 5;

for i = 1:size(M,2)
    group(i) = randi([1 3],1,1);
    count(group(i)) = count(group(i)) + 1;
    running_sum(group(i)) = running_sum(group(i)) + M(i);
    running_sum_squares(group(i)) = running_sum_squares(group(i)) + M(i)*M(i);
end

group

%GMM fitting with Gibbs sampling
for x = 1:iterations
    for id = 1:size(M,2)
        count(group(id)) = count(group(id))-1;
        running_sum(group(id)) = running_sum(group(id))-M(id);
        running_sum_squares(group(id)) = running_sum_squares(group(id)) - M(id)*M(id);
        for j = 1:3
            responsibilities(j) = normpdf(M(id),mu(j),sqrt(var(j)))*p(j);         
        end
        mu;
        responsibilities = responsibilities/sum(responsibilities);
        [toto , draw_samples] = histc(rand(1 , 1), [0 cumsum(responsibilities)]);
        draw_samples;
        group(id) = draw_samples;
        count(group(id)) = count(group(id))+1;
        running_sum(group(id)) = running_sum(group(id))+M(id);
        running_sum_squares(group(id)) = running_sum_squares(group(id)) + M(id)*M(id);
    end
    for j = 1:3
         mu(j) = running_sum(j)/count(j);
         p(j) = count(j)/size(M,2);
         var(j) = running_sum_squares(j)/count(j) - mu(j)*mu(j);
    end
    
 
end

%solution
p
mu
var