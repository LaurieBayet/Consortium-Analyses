function allsubj_results = nfold_classify_ParticipantLevel(MCP_struct,varargin)
%% nfold_classify_ParticipantLevel takes an MCP struct and performs
% n-fold cross-validation for n subjects to classify individual
% participants' average response patterns.


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


%% Load MCP struct if necessary
if isstring(MCP_struct) || ischar(MCP_struct)
    MCP_struct = load(MCP_struct,'-mat');
    varname = fieldnames(MCP_struct);
    MCP_struct = eval(['MCP_struct.' varname{1}]);
end


%% Parse out the input data
p = parse_inputs(MCP_struct, varargin{:});

%% Setting up the combinations of channel subsets
% Create all possible subsets. If setsize is equal to the total number of
% channels, there will only be one 'subset' which is the full channel
% array. If setsize is less than the total number of channels, there will
% be n-choose-k subsets to analyze.
%
% The size of the subsets can grow extremely quickly with the size of
% incl_channels. Consequently, there is a default max of 1000000 sets,
% which can be customized. If the total number of sets is larger than the
% max number of allowed sets, the list of sets will be subsampled.

% Determine how many sets will be generated. Can use this later for warning
% messages or other branching. Sets variable turns into a huge memory hog.

unmapped_sets = find_sets(p.Results);
sets = map_values(p, unmapped_sets);


%% norm check - do we want to scale individual participant data?
if p.Results.norm_data
    MCP_struct = scale_individuals(MCP_struct, p.Results);
end

%% Build MCPA struct for all subjects in the MCP
% Step 1: Epoching the data by time window

mcpa_struct = MCP_to_MCPA(MCP_struct,...
    p.Results.incl_subjects,...
    p.Results.incl_channels,...
    p.Results.time_window,...
    p.Results.baseline_window);

%% summarize MCPA struct
% Step 2: Apply the desired function (e.g., @nanmean) for summarizing time
% window data. You can write custom functions to deal with time- and
% channel-domain data however you want. Default behavior is to apply the
% function along the first dimension of the MCPA pattern (instance) and then the second dimension (time),
% but this can also be changed.

%% first decide how we want to concatenate or average over our dimensions
% intermediary step: see if the user specified the summarizing dimensions. If not,
% recommend what dimensions to average over

if ~isempty(p.Results.summarize_dimensions) || ~isfield(p.Results, 'summarize_dimensions')
    summarize_dimensions = p.Results.summarize_dimensions;
else
    isWithinSubjects = false;
    warning('summarize_dimensions not specified. Consulting recommend_dimensions.')
    
    [summarize_dimensions, ~] = recommend_dimensions(p.Results, isWithinSubjects);
    
    fprintf('Summarizing dimensions with %s:\n',func2str(p.Results.summary_handle))
    fprintf('%s ',summarize_dimensions{:})
    fprintf('\n')
end

% then see if the user specified the final dimensions the data should take
% before going into classification
if ~isempty(p.Results.final_dimensions) || ~isfield(p.Results, 'final_dimensions')
    final_dimensions = p.Results.final_dimensions;
else
    isWithinSubjects = false;
    warning('final_dimensions not specified. Consulting recommend_dimensions.')
    
    [~, final_dimensions] = recommend_dimensions(p.Results, isWithinSubjects);
   
    fprintf('The format the data will be in when it enters the classifier wrapper is: %s', final_dimensions{:});
    fprintf('\n')
end

%% then do the summarizing
if p.Results.verbose
    disp('Summarizing MCPA patterns with dimensions:');
    disp(strjoin(mcpa_struct.dimensions,' x '));
    disp(strjoin(cellfun(@num2str, num2cell(size(mcpa_struct.patterns)),'UniformOutput',false),' x '));
end

mcpa_summ = summarize_MCPA_Struct(p.Results.summary_handle,...
    mcpa_struct,...
    summarize_dimensions);

if p.Results.verbose
    disp('MCPA patterns have been summarized to:')
    disp(strjoin(mcpa_summ.dimensions,' x '));
    disp(strjoin(cellfun(@num2str, num2cell(size(mcpa_summ.patterns)),'UniformOutput',false),' x '));
end

%% Prep some basic parameters
n_subj = length(p.Results.incl_subjects);
n_sets = size(sets,1);
n_chan = length(p.Results.incl_channels);
try n_cond = length(unique(p.Results.conditions)); catch, n_cond = length(p.Results.conditions); end


%% Set up the results structure which includes a copy of MCPA_pattern

allsubj_results = create_results_struct(false,...
    mcpa_summ,...
    MCP_struct,...
    p,...
    sets,...
    n_subj,...
    n_sets,...
    n_chan,...
    n_cond);

stack = dbstack;
current_folding_function = stack.name;
allsubj_results.test_type = current_folding_function;

%% Begin the n-fold process: Select one test subj at a time from MCPA struct
for s_idx = 1:length(mcpa_summ.incl_subjects)
    
    if p.Results.verbose
        fprintf('Running %g feature subsets for Subject %g / %g',n_sets,s_idx,n_subj);
    end
    tic;
    
    %% Run over channel subsets
    temp_set_results_cond = nan(n_cond,n_sets,n_chan);
    
    %% Folding & Dispatcher: Here's the important part
    % Right now, the data have to be treated differently for 2
    % conditions vs. many conditions. In MCPA this is because 2
    % conditions can only be compared in channel space (or, hopefully,
    % MNI space some day). If there are a sufficient number of
    % conditions (6ish or more), we abstract away from channel space
    % using RSA methods. Then classifier is trained/tested on the RSA
    % structures. This works for our previous MCPA studies, but might
    % not be appropriate for other classifiers (like SVM).
    
    
    [group_data, group_labels, subj_data, subj_labels] = split_test_and_train(s_idx,...
        p.Results.conditions,...
        mcpa_summ.patterns,...
        mcpa_summ.event_types,...
        final_dimensions,...
        mcpa_summ.dimensions);
        
      
    %% Run classifier and compare output with correct labels
    for set_idx = 1:min(n_sets,p.Results.max_sets)
        %% Progress reporting bit (not important to function. just sanity)
        % Report at every 5% progress
        if p.Results.verbose
            status_jump = floor(n_sets/20);
            if ~mod(set_idx,status_jump)
                fprintf(' .')
            end
        end
        % Select the channels for this subset
        set_chans = sets(set_idx,:);

        %% classify
        % call differently based on if we do RSA or not
        
        % RSA
        if strcmp(func2str(p.Results.test_handle),'pairwise_rsa_test')
            [test_labels, comparisons] = p.Results.test_handle(...
                group_data(:,set_chans,:), ...
                group_labels,...
                subj_data(:,set_chans,:),...
                p.Results.opts_struct);

        else
            [test_labels, comparisons] = p.Results.test_handle(...
                    group_data(:,set_chans), ...
                    group_labels,...
                    subj_data(:,set_chans),...
                    p.Results.opts_struct);
        end

           
       %% Record results 
        if size(test_labels,2) > 1 % test labels will be a column vector if we don't do pairwise
            if s_idx==1 && set_idx == 1, allsubj_results.accuracy_matrix = nan(n_cond,n_cond,min(n_sets,p.Results.max_sets),n_subj); end

            if iscell(comparisons)
                subj_acc = strcmp(test_labels(:,1),comparisons(:,1));
                comparisons = cellfun(@(x) find(strcmp(x,mcpa_summ.event_types)),comparisons); 
            else
                subj_acc = test_labels(:,1)==comparisons(:,1);
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
                subj_labels(strcmp(strjoin(string(p.Results.conditions{cond_idx}),'+'),subj_labels)),... % known labels
                temp_test_labels(strcmp(strjoin(string(p.Results.conditions{cond_idx}),'+'),subj_labels))...% classifier labels
                );

                temp_set_results_cond(cond_idx,set_idx,set_chans) = nanmean(temp_acc);

                allsubj_results.accuracy(cond_idx).subsetXsubj(:,s_idx) = nanmean(temp_set_results_cond(cond_idx,:,:),3);
                allsubj_results.accuracy(cond_idx).subjXchan(s_idx,:) = nanmean(temp_set_results_cond(cond_idx,:,:),2);
            end
        end
                  
        %% Progress reporting
        if p.Results.verbose
            fprintf(' %0.1f mins\n',toc/60);
        end
    end % end set_idx loop
end % end subject loop

%% Visualization
if p.Results.verbose
    if n_sets > 1 && length(p.Results.conditions)==2
        
        figure
        errorbar(1:size(allsubj_results.accuracy(1).subjXchan,2),mean(allsubj_results.accuracy(1).subjXchan),std(allsubj_results.accuracy(1).subjXchan)/sqrt(size(allsubj_results.accuracy(1).subjXchan,1)),'r')
        hold;
        errorbar(1:size(allsubj_results.accuracy(2).subjXchan,2),mean(allsubj_results.accuracy(2).subjXchan),std(allsubj_results.accuracy(2).subjXchan)/sqrt(size(allsubj_results.accuracy(2).subjXchan,1)),'k')
        title('Decoding Accuracy across all channels: Red = Cond1, Black = Cond2')
        set(gca,'XTick',[1:length(p.Results.incl_channels)])
        set(gca,'XTickLabel',p.Results.incl_channels)
        hold off;
        
        figure
        errorbar(1:size(allsubj_results.accuracy(1).subjXchan,1),mean(allsubj_results.accuracy(1).subjXchan'),repmat(std(mean(allsubj_results.accuracy(1).subjXchan'))/sqrt(size(allsubj_results.accuracy(1).subjXchan,2)),1,size(allsubj_results.accuracy(1).subjXchan,1)),'r')
        hold;
        errorbar(1:size(allsubj_results.accuracy(2).subjXchan,1),mean(allsubj_results.accuracy(2).subjXchan'),repmat(std(mean(allsubj_results.accuracy(2).subjXchan'))/sqrt(size(allsubj_results.accuracy(2).subjXchan,2)),1,size(allsubj_results.accuracy(2).subjXchan,1)),'k')
        title('Decoding Accuracy across all subjects: Red = Cond1, Black = Cond2')
        set(gca,'XTick',[1:p.Results.incl_subjects])
        hold off;
        
    end
end




end
