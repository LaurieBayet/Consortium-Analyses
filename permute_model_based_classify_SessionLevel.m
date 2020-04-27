function allsubj_results = permute_model_based_classify_SessionLevel(results_struct)
%% function to classify sessions from individual participants
% n-fold within subjects cross-validation for n subjects with m sessions
% to classify individual sessions' average response patterns.


% This wrapper allows the user to choose if
% features will be averaged within-participants to produce a single
% participant-level observation or if individual events will be preserved.
% If the featuers are averaged within-participants, the training set
% is constrained to the number of participants minus 1. Otherwise the
% training set will be (participants - 1) * (number of instances of an
% object).

% Several parameters can be changed,
% including which functions are used to generate features and what
% classifier is trained. See Arguments below:
%
% Arguments:
% MCP_struct: either an MCP-formatted struct or the path to a Matlab file
% (.mat or .mcp) containing the MCP_struct.
% incl_features: features to include in the analysis. Default: all features
% incl_subjects: index of participants to include. Default: all participants
% baseline_window: [onset, offset] in seconds. Default [-5,0]
% time_window: [onset, offset] in seconds. Default [2,6]
% conditions: cell array of condition names / trigger #s. Default: {1,2}
% summary_handle: function handle (or char of function name) to specify how
% time-x-feature data should be summarized into features. Default: nanmean
% setsize: number of features to analyze (for subset analyses) Default: all
% test_handle: function handle for classifier. Default: mcpa_classify
% opts_struct: contains additional classifier options. Default: empty struct
% verbose: logical flag to report status updates and results. Default: true
% norm_data: option to perform some for of feature scaling. Defualt: false
% norm_withinSessions: if we do norm data, this allows the user to choose
% if we norm within individual sessions or across a participants session.
% It is only valid to norm across sessions for
% nfold_classify_ParticipantLevel. Default: true
% norm_function: function handle for how to scale the data. Default:
% minMax_scale
% minMax: what min and max to set the data to. Default: [0,1]
% summarize_dimensions: what dimensions and what order to summarize the
% dimensions of mcpa patterns by. Default behavior is to average dimensions
% in the order of this cell array. Default: {'instance', 'time'}



%% Prep some basic parameters
n_subj = length(results_struct.incl_subjects);
n_sets = size(sets,1);
n_feature = length(results_struct.incl_features);
s = size(results_struct.patterns);
n_sessions = s(end-1);
try n_cond = length(unique(results_struct.conditions)); catch, n_cond = length(results_struct.conditions); end

%% now begin the fold

for s_idx = 1:length(MCP_struct)
    
    if results_struct.verbose
        fprintf('Running %g feature subsets for Subject %g / %g \n',n_sets,s_idx,n_subj);
    end
    tic;
    
    % might want to get rid of this part later - purpose is to subset out
    % the participant for this round of CV
    if length(size(results_struct.patterns)) == 5
        subject_patterns = results_struct.patterns(:,:,:,:,s_idx);
    else
        subject_patterns = results_struct.patterns(:,:,:,s_idx);
    end
    
    %% do we have more than one session to classify on?
    % with current summarize dimensions, we can only proceed with
    % classification for 1 session if we are using KNN, SVM, or logistic
    % regression
    if n_sessions == 1
        %find something else for test train to be
        fold_dim = ndims(subject_patterns);
        num_data = size(subject_patterns,fold_dim);
        
        if ~isfield(results_struct, 'test_percent') || isempty(results_struct.test_percent)
            test_percent = .2;
        else
            test_percent = results_struct.test_percent;
        end
        num_in_fold = num_data*test_percent;
        num_folds = num_data/num_in_fold;
        one_session = true;
    else
        num_folds = length(MCP_struct(s_idx).Experiment.Runs);
        one_session = false;
    end
    
    for folding_idx = 1:num_folds
        %% Run over feature subsets
        temp_set_results_cond = nan(n_cond,n_sets,n_feature);
        
        if one_session
            fold_idx_end = folding_idx * num_in_fold;
            fold_idx_start = fold_idx_end - num_in_fold + 1;
            fold = fold_idx_start:fold_idx_end;
        else
            fold = folding_idx;
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
        
        [~, group_labels, subj_data, subj_labels] = split_test_and_train(fold,...
            results_struct.conditions,...
            subject_patterns,...
            results_struct.event_types,...
            final_dimensions,...
            results_struct.dimensions, [], []);
                                                                            
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

            % RSA
            if strcmp(func2str(results_struct.test_handle),'rsa_classify')
                [test_labels, comparisons] = results_struct.test_handle(...
                    results_struct.semantic_model, ...
                    group_labels,...
                    subj_data(:,set_features,:),...
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

            %% Record results 

            if size(test_labels,2) > 1 % test labels will be a column vector if we don't do pairwise
                if s_idx==1 && set_idx == 1 && folding_idx == 1, allsubj_results.accuracy_matrix = nan(n_cond,n_cond,min(n_sets,results_struct.max_sets),n_sessions,n_subj); end

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
                        allsubj_results.accuracy_matrix(comparisons(comp,1),:,set_idx,folding_idx,s_idx) = subj_acc(comp);
                    else
                        allsubj_results.accuracy_matrix(comparisons(comp,1),comparisons(comp,2),set_idx,folding_idx,s_idx) = subj_acc(comp);
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
                    allsubj_results.accuracy(cond_idx).subjXsession(s_idx,folding_idx) = nanmean(temp_set_results_cond(cond_idx,:,:),3);
                end
            end


        end % end set_idx
    %% Progress reporting
    if results_struct.verbose
        fprintf(' %0.1f mins\n',toc/60);
    end
    end % end session loop
    
end % end subject loop


end % end function
            

            