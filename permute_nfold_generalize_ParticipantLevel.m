function allsubj_results = permute_nfold_generalize_ParticipantLevel(results_struct)
%% nfold_generalize_ParticipantLevel takes an MCP struct and performs


%% Load MCP struct if necessary
if isstring(results_struct) || ischar(results_struct)
    results_struct = load(results_struct);
    
end

%% Prep some basic parameters
n_subj = length(results_struct.incl_subjects);
n_sets = size(results_struct.subsets,1);
n_feature = length(results_struct.incl_features);
% n_events = max(arrayfun(@(x) max(sum(x.fNIRS_Data.Onsets_Matrix)),MCP_struct));
n_cond = length(unique(results_struct.conditions));
groups = unique(results_struct.cond_key(:,2));
n_group = length(groups);



%% Begin the n-fold process: Select one test subj at a time from MCPA struct
for s_idx = 1:n_subj
    if results_struct.verbose
        fprintf('Running %g feature subsets for Subject %g / %g',n_sets,s_idx,n_subj);
    end
    tic
   
    %% Run over feature subsets
    temp_set_results_cond = nan(n_group,n_sets,n_feature);
    
    if isempty(results_struct.test_marks)
        % If the marks to use in test set are not already specified,
        % then throw an error and quit. TO DO: throw an warning instead
        % and randomly select half of each superordinate category.
        error('Please specify ''test_marks''');
    else
        % If the condition names to use in the test are specified,
        % then create the list of superordinate groups
        % (event_groups), the test conditions (test_events), and
        % the training conditions (train_events)
        event_groups = cellfun(@(x) results_struct.cond_key{strcmp(x,results_struct.cond_key(:,1)),2},results_struct.event_types, 'UniformOutput',false);
        test_events = cellfun(@(x) any(strcmp(x,results_struct.test_marks)),results_struct.event_types);
        train_events = cellfun(@(x) all(~strcmp(x,results_struct.test_marks)),results_struct.event_types);
    end


    %% Folding & Dispatcher: Here's the important part
    % Right now, the data have to be treated differently for 2
    % conditions vs. many conditions. In MCPA this is because 2
    % conditions can only be compared in feature space (or, hopefully,
    % MNI space some day). If there are a sufficient number of
    % conditions (6ish or more), we abstract away from feature space
    % using RSA methods. Then classifier is trained/tested on the RSA
    % structures. This works for our previous MCPA studies, but might
    % not be appropriate for other classifiers (like SVM).
    
    [group_data, group_labels, subj_data, subj_labels] = split_test_and_train(s_idx,...
        results_struct.conditions,...
        results_struct.patterns,...
        event_groups,...
        results_struct.final_dimensions,...
        results_struct.dimensions,...
        test_events,...
        train_events);

    %% permute the group labels
    num_labels = length(group_labels);
    permuted_idx = randperm(num_labels)';
    group_labels = group_labels(permuted_idx);
    
    for set_idx = 1:n_sets
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
            if strcmp(func2str(results_struct.test_handle),'pairwise_rsa_test')
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
            
            %% Compare the labels output by the classifier to the known labels
            if size(test_labels,2) > 1 % test labels will be a column vector if we don't do pairwise
            
                if s_idx==1 && set_idx == 1, allsubj_results.accuracy_matrix = nan(n_cond,n_cond,min(n_sets,results_struct.max_sets),n_subj); end
                
                if iscell(comparisons)
                    subj_acc = nanmean(strcmp(test_labels(:,1,:), test_labels(:,2,:)));
                    comparisons = cellfun(@(x) find(strcmp(x,unique(event_groups))),comparisons);
                else
                    subj_acc = nanmean(strcmp(test_labels(:,1,:), test_labels(:,2,:)));
                end
                
                for comp = 1:size(comparisons,1)
                    if size(comparisons,2)==1
                        allsubj_results.accuracy_matrix(comparisons(comp,1),:,set_idx,s_idx) = subj_acc(comp);
                    else
                        allsubj_results.accuracy_matrix(comparisons(comp,1),comparisons(comp,2),set_idx,s_idx) = subj_acc(comp);
                    end
                end    
                
            else                
                for group_idx = 1:n_group
                    temp_acc = cellfun(@strcmp,...
                        subj_labels(strcmp(string(groups{group_idx}),subj_labels)),... % known labels
                        test_labels(strcmp(string(groups{group_idx}),subj_labels))...% classifier labels
                        );

                    temp_set_results_cond(group_idx,set_idx,set_features) = nanmean(temp_acc);
                end
                for group_idx = 1:n_group
                    allsubj_results.accuracy(group_idx).subsetXsubj(:,s_idx) = nanmean(temp_set_results_cond(group_idx,:,:),3);
                    allsubj_results.accuracy(group_idx).subjXfeature(s_idx,:) = nanmean(temp_set_results_cond(group_idx,:,:),2);
                end
            end
        
    end %set_idx loop
    %% Progress reporting
    if results_struct.verbose
        fprintf(' %0.1f mins\n',toc/60);
    end
end % s_idx loop

%% Visualization
if results_struct.verbose
    if n_sets > 1 && length(results_struct.conditions)==2
        
        figure
        errorbar(1:size(allsubj_results.accuracy.cond1.subjXfeature,2),mean(allsubj_results.accuracy.cond1.subjXfeature),std(allsubj_results.accuracy.cond1.subjXfeature)/sqrt(size(allsubj_results.accuracy.cond1.subjXfeature,1)),'r')
        hold;
        errorbar(1:size(allsubj_results.accuracy.cond2.subjXfeature,2),mean(allsubj_results.accuracy.cond2.subjXfeature),std(allsubj_results.accuracy.cond2.subjXfeature)/sqrt(size(allsubj_results.accuracy.cond2.subjXfeature,1)),'k')
        title('Decoding Accuracy across all features: Red = Cond1, Black = Cond2')
        set(gca,'XTick',[1:length(results_struct.incl_features)])
        set(gca,'XTickLabel',results_struct.incl_features)
        hold off;
        
        figure
        errorbar(1:size(allsubj_results.accuracy.cond1.subjXfeature,1),mean(allsubj_results.accuracy.cond1.subjXfeature'),repmat(std(mean(allsubj_results.accuracy.cond1.subjXfeature'))/sqrt(size(allsubj_results.accuracy.cond1.subjXfeature,2)),1,size(allsubj_results.accuracy.cond1.subjXfeature,1)),'r')
        hold;
        errorbar(1:size(allsubj_results.accuracy.cond2.subjXfeature,1),mean(allsubj_results.accuracy.cond2.subjXfeature'),repmat(std(mean(allsubj_results.accuracy.cond2.subjXfeature'))/sqrt(size(allsubj_results.accuracy.cond2.subjXfeature,2)),1,size(allsubj_results.accuracy.cond2.subjXfeature,1)),'k')
        title('Decoding Accuracy across all subjects: Red = Cond1, Black = Cond2')
        set(gca,'XTick',[1:results_struct.incl_subjects])
        hold off;
        
    end
end

end