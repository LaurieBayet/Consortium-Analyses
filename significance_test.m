

function [p, distribution, test_acc, acc] = significance_test(results_struct, n_iter, test_type)
%% find the significance of decoding accuracy
% takes in the results struct from folding wrappers and performs a
% permutation test to find the p value for our classification accuracy 

%% set number of iterations if it isnt provided
if isempty(n_iter)
    n_iter = min(10000, factorial(results_struct.conditions));
end

%% find accuracy for the results struct


if isfield(results_struct, 'accuracy_matrix')
    
    if ~isempty(strfind(results_struct.test_type, 'Participant'))
        acc = nanmean(results_struct.accuracy_matrix,4);
        test_acc = nanmean(acc(:));
    else
        acc = nanmean(results_struct.accuracy_matrix,4);
        acc = nanmean(acc,5);
        test_acc = nanmean(acc(:));
    end
else
    test_acc = mean([mean(results_struct.accuracy(1).subsetXsubj) mean(results_struct.accuracy(2).subsetXsubj)]);
end



%% find accuracy distribution
distribution = [];
for iter = 1:n_iter
    
    % do classification
    iter_results = test_type(results_struct);
    
    % get classification accuracy 
    if length(results_struct.conditions) == 2 % if we gave two conditions
        perm_accuracy = mean([mean(iter_results.accuracy(1).subsetXsubj) mean(iter_results.accuracy(2).subsetXsubj)]);
    else % if we have more than 2 conditions (will have results matrix rather than vector)        
        if ~isempty(strfind(results_struct.test_type, 'Participant'))
            perm_acc = nanmean(results_struct.accuracy_matrix,4);
            perm_accuracy = nanmean(acc(:));
        else
            perm_acc = nanmean(results_struct.accuracy_matrix,4);
            perm_acc = nanmean(perm_acc,5);
            perm_accuracy = nanmean(perm_acc(:));
        end
        
        
    end
    
    distribution = [distribution perm_accuracy];

end

%% calculate p

if n_iter == factorial(length(results_struct.conditions))
    p = sum(distribution >= test_acc)/n_iter;
else
    p = (sum(distribution >= test_acc) + 1)/(n_iter + 1);
end


end
