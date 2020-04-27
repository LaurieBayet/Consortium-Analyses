function allsubj_results = permute_nfold_classify_ParticipantLevel(results_struct)
%% nfold_classify_ParticipantLevel_permutationTest takes a results struct and performs
% n-fold cross-validation for n subjects to classify individual
% participants' average response patterns, but with permuted labels. 
% All parameters are taken from the results struct


% Arguments:
% results_struct: results struct from previous decoding


%% Prep some basic parameters
n_subj = length(results_struct.incl_subjects);
n_sets = size(results_struct.subsets,1);
n_feature = length(results_struct.incl_features);
try n_cond = length(unique(results_struct.conditions)); catch, n_cond = length(results_struct.conditions); end


%% Begin the n-fold process: Select one test subj at a time from MCPA struct
for s_idx = 1:n_subj
    if results_struct.verbose
        fprintf('Running %g feature subsets for Subject %g / %g',n_sets,s_idx,n_subj);
        fprintf('\n');
    end
    tic;
    
    %% Run over feature subsets
    temp_set_results_cond = nan(n_cond,n_sets,n_feature);
    
    %% Folding & Dispatcher
    [group_data, group_labels, subj_data, subj_labels] = split_test_and_train(s_idx,...
        results_struct.conditions,...
        results_struct.patterns,...
        results_struct.event_types,...
        results_struct.final_dimensions,...
        results_struct.dimensions, [], []);

    %% permute the group labels
    num_labels = length(group_labels);
    permuted_idx = randperm(num_labels)';
    group_labels = group_labels(permuted_idx);
        
    %% Run classifier and compare output with correct labels
    for set_idx = 1:min(n_sets,results_struct.max_sets)
        %% Progress reporting bit (not important to function. just sanity)
        % Report at every 5% progress
        if results_struct.verbose
            status_jump = floor(n_sets/20);
            if ~mod(set_idx,status_jump)
                fprintf(' .')
            end
        end

        % Select the features for this subset
        set_features = results_struct.subsets(set_idx,:);
        
        
        %% classify
        % call differently based on if we do RSA or not
        % if we do pairwise comparison, the result test_labels will be a 3d
        % matrix with the dimensions: predicted label x correct label x
        % index of comparison. The output 'comparisons' will be the
        % conditions that were compared and can either be a 2d cell array or a
        % matrix of integers. If we don't do pairwise comparisons, the
        % output 'test_labels' will be a 1d cell array of predicted labels.
        % The output 'comparisons' will be a 1d array of the correct
        % labels.
        
        % RSA
        if strcmp(func2str(results_struct.test_handle),'rsa_classify')
            [test_labels, comparisons] = results_struct.test_handle(...
                group_data(:,set_features,:), ...
                group_labels,...
                subj_data(:,set_features,:),...
                subj_labels,...
                results_struct.opts_struct);

        else
            [test_labels, comparisons] = results_struct.test_handle(...
                group_data(:,set_features), ...
                group_labels,...
                subj_data(:,set_features),...
                subj_labels,...
                results_struct.opts_struct);
        end

        %% Record results 
                
        if size(test_labels,2) > 1 % test labels will be a column vector if we don't do pairwise
            if s_idx==1 && set_idx == 1, allsubj_results.accuracy_matrix = nan(8,8,min(n_sets,results_struct.max_sets),n_subj); end

            % if not nans, see how comparisons is saved. 
            % If its a cell array, then we need to find where the values in that array correspond to the condition location in event_types
            % Then find the subject accuracy
            if iscell(comparisons) 
                subj_acc = nanmean(strcmp(test_labels(:,1,:), test_labels(:,2,:)));
                nan_idx = cellfun(@(x) any(isnan(x)), test_labels(:,1,:), 'UniformOutput', false);
                subj_acc(:,:,[nan_idx{1,:,:}]) = nan;

                comparisons = cellfun(@(x) find(strcmp(x,results_struct.event_types)),comparisons);
            else
                subj_acc = nanmean(strcmp(test_labels(:,1,:), test_labels(:,2,:)));
                nan_idx = cellfun(@(x) any(isnan(x)), test_labels(:,1,:), 'UniformOutput', false);
                subj_acc(:,:,[nan_idx{1,:,:}]) = nan;
            end
          
            % Then loop through comparisons and save accuracy to the results struct
            for comp = 1:size(comparisons,1)
                if size(comparisons,2)==1
                    allsubj_results.accuracy_matrix(comparisons(comp,1),:,set_idx,s_idx) = subj_acc(comp);
                else
                    allsubj_results.accuracy_matrix(comparisons(comp,1),comparisons(comp,2),set_idx,s_idx) = subj_acc(comp);
                end
            end
        else
            for cond_idx = 1:n_cond
                temp_acc = cellfun(@strcmp,...
                    subj_labels(strcmp(strjoin(string(results_struct.conditions{cond_idx}),'+'),subj_labels)),... % known labels
                    test_labels(strcmp(strjoin(string(results_struct.conditions{cond_idx}),'+'),subj_labels))...% classifier labels
                    );
                
                temp_set_results_cond(cond_idx,set_idx,set_features) = nanmean(temp_acc);
            end
            for cond_idx = 1:n_cond
                allsubj_results.accuracy(cond_idx).subsetXsubj(:,s_idx) = nanmean(temp_set_results_cond(cond_idx,:,:),3);
                allsubj_results.accuracy(cond_idx).subjXfeature(s_idx,:) = nanmean(temp_set_results_cond(cond_idx,:,:),2);
            end
        end
        
    end % end set_idx loop
end % end s_idx loop

end




