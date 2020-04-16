

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
    results_struct_participant_accuracy = [];
    for sub = 1:results_struct.incl_subj
        results_struct_participant_accuracy = [results_struct_participant_accuracy nanmean(nanmean(results_struct.accuracy_matrix(:,:,sub)))];
    end
    result_struct_accuracy = mean(results_struct_participant_accuracy);
else
    result_struct_accuracy = mean([mean(results_struct.accuracy(1).subsetXsubj) mean(results_struct.accuracy(2).subsetXsubj)]);
end



%% find accuracy distribution
iter_accuracy = [];
for iter = 1:n_iter
    
    % do classification
    iter_results = test_type(results_struct);
    
    % get classification accuracy 
    if length(results_struct.conditions) == 2 % if we gave two conditions
        accuracy = mean([mean(iter_results.accuracy(1).subsetXsubj) mean(iter_results.accuracy(2).subsetXsubj)]);
    else % if we have more than 2 conditions (will have results matrix rather than vector)
        participant_accuracy = [];
        for sub = 1:results_struct.incl_subj
            participant_accuracy = [participant_accuracy nanmean(nanmean(iter_results.accuracy_matrix(:,:,sub)))];
        end
        accuracy = mean(participant_accuracy);
    end
    
    iter_accuracy = [iter_accuracy accuracy];

end

%% calculate p

if n_iter == factorial(length(results_struct.conditions))
    p = sum(iter_accuracy >= result_struct_accuracy)/n_iter;
else
    p = (sum(iter_accuracy >= result_struct_accuracy) + 1)/(n_iter + 1);
end


end
