function allsubj_results = permute_model_based_classify_ParticipantLevel(results_struct)
%% rsa_classify_IndividualSubjects takes an MCP struct and performs
% RSA classification for n subjects to classify individual
% participants' average response patterns using a semantic model. This wrapper assumes that
% features will be averaged within-participants to produce a single
% participant-level observation. Thus the training set is constrained to
% the number of participants minus 1. Several parameters can be changed,
% including which functions are used to generate features and what
% classifier is trained. See Arguments below:
%
% Arguments:
% MCP_struct: either an MCP-formatted struct or the path to a Matlab file
% (.mat or .mcp) containing the MCP_struct.
% semantic_model: The model we want to use for classification
% incl_channels: channels to include in the analysis. Default: all channels
% incl_subjects: index of participants to include. Default: all participants
% baseline_window: [onset, offset] in seconds. Default [-5,0]
% time_window: [onset, offset] in seconds. Default [2,6]
% conditions: cell array of condition names / trigger #s. Default: {1,2}
% summary_handle: function handle (or char of function name) to specify how
% time-x-channel data should be summarized into features. Default: nanmean
% setsize: number of channels to analyze (for subset analyses) Default: all
% test_handle: function handle for classifier. Default: mcpa_classify
% opts_struct: contains additional classifier options. Default: empty struct
% verbose: logical flag to report status updates and results. Default: true

% dimensions for the accuracy matrix in results struct is cond x cond x
% channel subset x subject



%% Prep some basic parameters
n_subj = length(results_struct.incl_subjects);
n_sets = size(sets,1);
n_feature = length(results_struct.incl_channels);
try n_cond = length(unique(results_struct.conditions)); catch, n_cond = length(results_struct.conditions); end


%% Begin the n-fold process: Select one test subj at a time from MCPA struct
for s_idx = 1:length(results_struct.incl_subjects)
    if results_struct.verbose
        fprintf('Running %g feature subsets for Subject %g / %g',n_sets,s_idx,n_subj);
    end
    tic;
    
    %% Run over feature subsets
    temp_set_results_cond = nan(n_cond,n_sets,n_feature);
    
    %% Folding & Dispatcher: Here's the important part
    % Right now, the data have to be treated differently for 2
    % conditions vs. many conditions. In MCPA this is because 2
    % conditions can only be compared in feature space (or, hopefully,
    % MNI space some day). If there are a sufficient number of
    % conditions (6ish or more), we abstract away from feature space
    % using RSA methods. Then classifier is trained/tested on the RSA
    % structures. This works for our previous MCPA studies, but might
    % not be appropriate for other classifiers (like SVM).
       
    [~, group_labels, subj_data, subj_labels] = split_test_and_train(s_idx,...
        results_struct.conditions,...
        results_struct.patterns,...
        results_struct.event_types,...
        final_dimensions,...
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
        set_features = sets(set_idx,:);

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
                results_struct.semantic_model,...
                group_labels,...
                subj_data(:,set_features,:,:),...
                subj_labels,...
                results_struct.opts_struct);

        else
            [test_labels, comparisons] = results_struct.test_handle(...
                results_struct.semantic_model, ...
                group_labels,...
                subj_data(:,set_features),...
                subj_labels,...
                results_struct.opts_struct);
        end
           
       %% Record results .
        if size(test_labels,2) > 1 % test labels will be a column vector if we don't do pairwise
            if s_idx==1 && set_idx == 1, allsubj_results.accuracy_matrix = nan(n_cond,n_cond,min(n_sets,results_struct.max_sets),n_subj); end

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
                  
        %% Progress reporting
        if results_struct.verbose
            fprintf(' %0.1f mins\n',toc/60);
        end
    end % end set_idx loop
end % end subject loop

end
