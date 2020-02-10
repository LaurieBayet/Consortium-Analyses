
function p = significance_test(results_struct, n_iter, test_type)
%% find the significance of decoding accuracy
% takes in the results struct from folding wrappers and performs a
% permutation test to find the p value for our classification accuracy 

%% set number of iterations if it isnt provided
if isempty(n_iter)
    n_iter = min(10000, factorial(results_struct.conditions));
end

%% find accuracy for the results struct


if isfield(results_struct, 'accuracy_matrix')       
    results_struct_accuracy = results_struct.accuracy_matrix;
        
    % we do this loop because within subjects and participant level
    % have a different amount of dimensions in their accuracy matrix
    for mean_dim = 1:ndims(results_struct.accuracy_matrix)
        results_struct_accuracy = nanmean(results_struct_accuracy, mean_dim);
    end
   
else
    result_struct_accuracy = mean([mean(results_struct.accuracy(1).subsetXsubj) mean(results_struct.accuracy(2).subsetXsubj)]);
end



%% find accuracy distribution
iter_accuracy = [];
for iter = 1:n_iter
    fprintf('Performing iteration %g \n', iter);
    % do classification
    iter_results = test_type(results_struct);
    
    % get classification accuracy 
    if isfield(iter_results, 'accuracy_matrix')
        accuracy = iter_results.accuracy_matrix;
        
        for mean_dim = 1:ndims(results_struct.accuracy_matrix)
            accuracy = nanmean(accuracy, mean_dim);
        end
        
    else 
        accuracy = mean([mean(iter_results.accuracy(1).subsetXsubj) mean(iter_results.accuracy(2).subsetXsubj)]);     
    end
    
    iter_accuracy = [iter_accuracy accuracy];

end

%% calculate p

if n_iter == factorial(length(results_struct.conditions))
    p = sum(iter_accuracy >= results_struct_accuracy)/n_iter;
else
    p = (sum(iter_accuracy >= results_struct_accuracy) + 1)/(n_iter + 1);
end


end
