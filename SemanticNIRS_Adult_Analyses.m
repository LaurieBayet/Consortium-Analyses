%% Semantic Classification of fNIRS Data in Adults and Infants
% https://osf.io/rj38w
% Script for pre-registered analyses of adult data

%% Prep workspace for the analysis
% Change to working directory where .mcp files are stored
%cd('~/Desktop/Pton_adult/Pton_adult_repairNames');

% Import the MCP file and clean up duplicate trials (a bug specific to the
% present experiment, not generally necessary)
load('SemanticNIRS_Adult_dataset1_21-Nov-2019.mcp','-mat');
MCP_data = clean_shimadzu_trials(MCP_data);

% Load the Consortium tools into the workspace
%addpath(genpath('~/Desktop/original_decoding_scripts'));

%% Update the condition names
% cond_names lists the stimuli in order by their trigger number in the
% stimulus matrix (s or aux).
cond_names = {'baby','book','bottle','cat','dog','hand','shoe','spoon'};
for old_mark = 1:length(cond_names)
    MCP_data = MCP_relabel_stimuli(MCP_data,old_mark,cond_names{old_mark},0);
end

%% create transformation matrices
path_to_tddatabase = ['/Users/annaph/Desktop/GRIDSEARCH_REAL/data/'];
for subj_idx = 1:16
    for session_idx = 1:4
         try
            % if converting to Brodmann's, write out file path to where
            % probe location data for that participant is stores
            pathname = ['/Users/annaph/Desktop/GRIDSEARCH_REAL/data/adult_POS_data/dec',num2str(subj_idx,'%02.0f'),'/decoding_',num2str(session_idx),'/'];
            MCP_data(subj_idx) = create_transformation_matrix(MCP_data(subj_idx), false, pathname, 'POS.mat', path_to_tddatabase, session_idx);    
        catch
            % this only happens when the participant has 3 sessions
            % instead of 4, so I just ignore it
            warning(['Failed to convert subject ' num2str(subj_idx) ' session ' num2str(session_idx) ' into Brodmanns Areas.'])
            continue;
        end
    end
end
%% PREVIEW OF THE DATA USING MCPA FUNCTION
% This section is only for plotting the mean HRFs and visualizing the
% overall similarity of the stimulus classes. It's not necessary to do this
% for classification. (It happens automatically in the analyses below.)

% Extract events into MCPA struct and graph the grand-average HRF over time
% for each condition. Then summarize the data by taking the time-window
% average for each condition, and plot a similarity matrix

MCPA_data = MCP_to_MCPA(...
    MCP_data,...                % MCP data struct
    1:16,...                      % Subjects to include (blank indicates all)
    1:47,... 
    1:139,...% Channels to include (blank indicates all)
    [-5,30],...                 % Analysis time window specified in seconds
    [-5,0]...                   % Baseline time window specified in seconds
    );

% Plot the time-windowed data (HRF) for each condition, averaged across all
% subjects and all channels.
n_cond = length(MCPA_data.event_types);
figure;
suptitle('Adult Consortium Data')
for i = 1:n_cond
    subplot(ceil(n_cond/2),2,i)
    plot(MCPA_data.time_window,smooth(squeeze(nanmean(nanmean(MCPA_data.patterns(:,i,:,:),4),3))))
    hold off;
    xlabel('time from stim onset [sec]');
    title(MCP_data(1).Experiment.Conditions(i).Name);
end

% Summarize the MCPA data by reducing the first dimension (time) using
% nanmean. This is the same procedure for generating features that will be
% used in the classifier, but it is performed automatically prior to
% classification. Here it's done manually just for visualization purposes.
summarize_dimensions = {'repetition', 'time'}; 
MCPA_data_summ = summarize_MCPA_Struct(@nanmean,...
    MCPA_data,...
    summarize_dimensions);


model_mat = nan(size(MCPA_data_summ.patterns,1),size(MCPA_data_summ.patterns,1),size(MCPA_data_summ.patterns,3),size(MCPA_data_summ.patterns,4));
    for i = 1: (size(MCPA_data_summ.patterns,3)*size(MCPA_data_summ.patterns,4))
        model_mat(:,:,i) = corr(MCPA_data_summ.patterns(:,:,i)', 'type', 'spearman');
    end
group_corr = nanmean(model_mat,3);
group_corr = nanmean(group_corr,4);

figure;
imagesc(group_corr)
xticklabels(cond_names)
yticklabels(cond_names)
title('full sample RSA matrix')
caxis([-0.5,0.5])
colorbar('hot')
[i, j, ~] = find(~isnan(group_corr));
text(i-.3,j,num2str(round(group_corr(~isnan(group_corr)),2)));

%% HYPOTHESIS 1: PRESERVATION OF HIERARCHICAL CATEGORY REPRESENTATIONS
%% Analysis 1A: Category Level, n-fold between-subjects classification
clear opts
opts.pairwise = true;
opts.similarity_space = true;
nfold_category_results = model_based_classify_ParticipantLevel(...
    MCP_data,...    
    model,...                                   % MCP data struct
    'baseline_window',[-5,0],...                % Baseline window to average and subtract from the time window
    'time_window',[3,8],...                     % Time window to analyze (in sec)
    'conditions',cond_names,...                 % Conditions to include (the named ones, not m1-m8)
    'summary_handle',@nanmean,...               % Which function to use to summarize data to features
    'test_handle',@rsa_classify,...            % Which classifier to call (also can have opts_struct)
    'verbose',true,...
    'norm_data', false,...
    'opts_struct', opts);

%%
 Plot the mean accuracy for each pairwise comparison
figure;
mean_acc = mean(nfold_category_results.accuracy_matrix,4);
overall_acc = nanmean(mean_acc(:));
imagesc(mean_acc')
xticklabels(cond_names)
yticklabels(cond_names)
caxis([0,1])
colorbar('hot')
title(['Analysis 1a, overall accuracy: ' num2str(overall_acc)]);
[i, j, ~] = find(~isnan(mean_acc));
text(i-.3,j,num2str(round(mean_acc(~isnan(mean_acc)),2)));

%% Analysis 1B: Exemplar Level, n-fold generalization between subjects

% Duplicate the MCP data because we will make some changes to the stimuli
MCP_data_object = MCP_data;

% Import the exact stimulus orders for each subject from an external table
% so that we can discriminate exemplars, which did not have unique trigger
% IDs (e.g., baby1 and baby2 are both trigger 1 'baby')
object_names = load('adult_stim_order.mat');

% Loop through the old trigger / mark IDs and replace them one at a time
for old_mark = 1:length(cond_names)
    old_name = cond_names{old_mark};
    fprintf('Replacing %s:\n',cond_names{old_mark});
    
    % Loop through the subjects and replace triggers one at a time
    for s_idx = 1:length(MCP_data_object)
        fprintf('Subj: %s... ', MCP_data_object(s_idx).Subject.Subject_ID);
        
        % This function is specific to our dataset (saved locally) and
        % screens out the relevant stimulus category from the list
        new_names = get_names_from_orders(MCP_data_object(s_idx), old_mark, object_names.orders(:,s_idx), cond_names);
        
        % The MCP_relabel_stimuli can take a specific condition (baby) or
        % trigger number (1) and replace it with a different label OR
        % replace each instance of that condition with a vector of specific
        % changes (vector must be same length as the number of instances).
        try
            MCP_data_object(s_idx) = MCP_relabel_stimuli(MCP_data_object(s_idx),old_mark,new_names,0);
            fprintf('OK!\n');
        catch
            onsets = MCP_data_object(s_idx).fNIRS_Data.Onsets_Matrix(:,old_mark);
            fprintf('Failed. Found %g onsets but %g replacements.\n',sum(onsets),length(new_names));
        end
    end
    
    % Removes the category-level condition that was just replaced by the
    % exemplars.
    MCP_data_object = MCP_delete_stimuli(MCP_data_object,old_mark,0);
end

% Build the key which aligns all of the exemplars (e.g., baby1 & baby2) with their category labels
object_key = [{MCP_data_object(1).Experiment.Conditions.Name};...
    cellfun(@(x) x(1:(end-1)),{MCP_data_object(1).Experiment.Conditions.Name},'UniformOutput',false)]';

% Cycle through all possible combinations of one exemplar per category in
% training and the other exemplar per category in testing. There are 256
% such possible combinations for 8 categories, so they're denoted by the
% binary sequence of 0 to 255 as 8 bits, with 0 for the first exemplar and
% 1 for the second exemplar.
combo_result_mat_exemplar = nan(256,length(MCP_data_object));

for combo_idx = 0:255
    
    % Choose the eight test items based on the binary math
    test_set = object_key([1:2:16]+bitget(combo_idx,1:8),1);
    % And report them on screen
    fprintf('Holding out test set %g: ',combo_idx+1);
    fprintf('%s ',test_set{:})
    fprintf('\n');
    
    % Run the generalization
    results = nfold_generalize_ParticipantLevel(...
        MCP_data_object,...
        'baseline_window',[-5,0],...                % Baseline window to average and subtract from the time window
        'time_window',[3,8],...                     % Time window to analyze (in sec)
        'summary_handle',@nanmean,...               % Which function to use to summarize data to features
        'test_handle',@rsa_classify,...            % Which classifier to call (also can have opts_struct)
        'cond_key',object_key,...                   % This key specifies the relationships between exemplars and their categories
        'test_marks',test_set,...                   % List of specific exemplars to hold out for the test set (all others used in training)
        'verbose',false,...
        'opts_struct', opts,...
        'setsize', 47,...
        'incl_channels', 1:139,...
        'incl_features', 1:47);
    
    % Save participant-level mean accuracy for all pairwise comparisons
    results_acc = squeeze(nanmean(nanmean(results.accuracy_matrix,2),1))';
    combo_result_mat_exemplar(combo_idx+1,:) = results_acc;
end

figure();
% Plot a histogram of the accuracies for each generalization trial
subplot(1,2,1);
hist(mean(combo_result_mat_exemplar,2),0.3:.05:0.7)
title(sprintf('Accuracy in each stimulus combination analysis 1b: mean=%0.2f',mean(mean(combo_result_mat_exemplar))))
xlabel('Accuracy')
ylabel('Frequency')
% Plot the average accuracy (across all generalizations) for each subject
subplot(1,2,2);
plot(mean(combo_result_mat_exemplar,1),'o-')
title('Subject-wise accuracy across all stimulus combos analysis 1b')
xlabel('Subject')
ylabel('Mean Accuracy')


%% Analysis 1C: Group Level, Set up and run leave-4-conditions-out generalization

% Update the add condition-to-group mapping key
group_key = {'baby','book','bottle','cat','dog','hand','shoe','spoon';
    'anim','inan','inan','anim','anim','anim','inan','inan'}';
anim_combos = nchoosek(find(strcmp(group_key(:,2),'anim')),2);
inan_combos = nchoosek(find(strcmp(group_key(:,2),'inan')),2);

classif_opts = struct;
classif_opts.corr_stat = 'spearman';
classif_opts.exclusive = true;

combo_result_mat_group = nan(size(anim_combos,1)*size(inan_combos,1),length(MCP_data));
combo_idx=1;
for anim_idx = 1:length(anim_combos)
    for inan_idx = 1:length(inan_combos)
        test_set = ...
            [group_key(anim_combos(anim_idx,:),1);...
            group_key(inan_combos(inan_idx,:),1)];
        fprintf('Holding out test set %g: ',combo_idx);
        fprintf('%s ',test_set{:})
        fprintf('\n');
        results = nfold_generalize_ParticipantLevel(...
            MCP_data,...
            'baseline_window',[-5,0],...                % Baseline window to average and subtract from the time window
            'time_window',[3,8],...                     % Time window to analyze (in sec)
            'summary_handle',@nanmean,...               % Which function to use to summarize data to features
            'test_handle',@mcpa_classify,...            % Which classifier to call (also can have opts_struct)
            'test_marks',test_set,...                   % List of specific exemplars to hold out for the test set (all others used in training)
            'cond_key',group_key,...                    % This key specifies the relationships between exemplars and their categories
            'opts_struct',classif_opts,...
            'verbose',false,...
            'setsize', 47,...
            'incl_channels', 1:139,...
            'incl_features', 1:47);
        results_acc = mean(...
            reshape([results.accuracy.subsetXsubj],...
            16,length(results.accuracy)),2)';
        combo_result_mat_group(combo_idx,:) = results_acc;
        combo_idx=combo_idx+1;
    end
end

figure();
subplot(1,2,1);
hist(mean(combo_result_mat_group,2),0.3:.05:0.7)
title(sprintf('Accuracy in each stimulus combination analysis 1c: mean=%0.2f',mean(mean(combo_result_mat_group))))
xlabel('Accuracy')
ylabel('Frequency')
subplot(1,2,2);
plot(mean(combo_result_mat_group,1),'o-')
title('Subject-wise accuracy across all stimulus combos analysis 1c')
xlabel('Subject')
ylabel('Mean Accuracy')